#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
from datetime import datetime
import pymeshlab
import random
import math
import torch.nn
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from src.utils.sh_utils import eval_sh
from src.utils.transforms import project_points
from src.utils.vis_util import get_colors_from_cmap
from src.utils.extra import *
import torch.distributions as D
import cv2
from tqdm import trange
from skimage import measure
import taichi as ti

ti.init(arch=ti.cuda, device_memory_fraction=0.8)


def dilate_mask(mask, kernel_size=11):
    kernel = torch.ones(
        (kernel_size, kernel_size), dtype=torch.float32, device=mask.device
    )
    mask = mask.float()
    mask = torch.nn.functional.conv2d(
        mask.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=kernel_size // 2,
    )
    mask = mask.squeeze(0).squeeze(0)
    mask = mask > 0
    return mask


def get_nocs_grid(bones_rest, res, ratio=[1.0, 1.0, 1.0]):
    heads = bones_rest.heads
    tails = bones_rest.tails
    keypts = to_numpy(torch.cat([heads[:1], tails], dim=0))
    cano_min = np.min(keypts, axis=0)
    cano_max = np.max(keypts, axis=0)
    center = (cano_max + cano_min) / 2
    center += np.array([0, 0, -0.03])

    x_ratio, y_ratio, z_ratio = ratio
    res_scaled = res / np.array([x_ratio, y_ratio, z_ratio])
    res_scaled = res_scaled.astype(np.int32)
    d, h, w = (
        math.floor(res_scaled[2]),
        math.floor(res_scaled[1]),
        math.floor(res_scaled[0]),
    )
    grid_points = create_skinning_grid(d, h, w)
    grid_colors = (grid_points.clone() + 1) / 2
    scale = np.linalg.norm(cano_max - cano_min) / 2
    scale = np.array([[scale * z_ratio, scale * y_ratio, scale * x_ratio]]).astype(
        np.float32
    )
    grid_points = grid_points * scale + center

    center = to_tensor(center)
    scale = to_tensor(scale)

    grid_points = grid_points.reshape(d, h, w, 3)
    grid_colors = grid_colors.reshape(d, h, w, 3)
    return grid_points, grid_colors, center, scale


def get_nocs_colors(xyz, grid_colors, grid_center, grid_scale):
    device = xyz.device
    grid_colors = grid_colors.permute(3, 0, 1, 2).unsqueeze(0).to(device)
    grid_colors.expand(1, -1, -1, -1, -1)

    xyz_norm = (xyz - grid_center.to(device)) / grid_scale.to(device)
    xyz_norm = xyz_norm.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    nocs_coord = torch.nn.functional.grid_sample(
        grid_colors,
        xyz_norm,
        align_corners=True,
        mode="bilinear",
        padding_mode="zeros",
    )
    nocs_coord = nocs_coord.squeeze(0).squeeze(-1).squeeze(-1).T
    return nocs_coord


def get_points_outside_mask(camera, points, mask, keypoints=None, dilate=False):
    K = camera.K
    extr = camera.extr
    if len(K.shape) == 3:
        K = K[0]
    if len(extr.shape) == 3:
        extr = extr[0]

    if len(mask.shape) == 4:
        mask = mask[0]

    # Dilate Mask
    if dilate:
        mask = dilate_mask(mask[..., 0]).unsqueeze(-1).int()

    p2d = project_points(points[None], K, extr[:3, :4])[0]

    pts_x = torch.clamp(p2d[..., 0], 0, mask.shape[1] - 1).int()
    pts_y = torch.clamp(p2d[..., 1], 0, mask.shape[0] - 1).int()
    mask = ~mask.bool()
    mask_value = mask[pts_y, pts_x]

    # If any of the keypoint is outside the mask, ignore mask_value
    if keypoints is not None:
        k2d = project_points(keypoints[None], K, extr[:3, :4])[0]
        pts_x = torch.clamp(k2d[..., 0], 0, mask.shape[1] - 1).int()
        pts_y = torch.clamp(k2d[..., 1], 0, mask.shape[0] - 1).int()
        keypt_mask_value = mask[pts_y, pts_x]
        if torch.any(keypt_mask_value):
            mask_value = torch.zeros_like(mask_value)

    """ Visualization debug code
    mask_value = torch.zeros_like(mask_value)
    p2d = p2d[mask_value[..., 0].bool()]
    mask = mask.repeat(1, 1, 3)
    img = to_numpy(mask) * 255
    img = np.ascontiguousarray(img).astype(np.uint8)
    
    ## visualize these points using OpenCV
    for idx in range(p2d.shape[0]):
        x = int(p2d[idx][0])
        y = int(p2d[idx][1])
        img = cv2.circle(img, (x, y), 1, (0, 255, 0), 1)
    dump_image(img, './proj_points.png')
    """

    return mask_value


def offsets_from_voxel_grid(xyz, grid_center, grid_scale, grid_weights):
    device = xyz.device
    grid_weights = grid_weights.permute(3, 0, 1, 2).unsqueeze(0).to(device)
    grid_weights.expand(1, -1, -1, -1, -1)
    xyz_norm = (xyz - grid_center.to(device)) / grid_scale.to(device)
    xyz_norm = xyz_norm.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    offset = torch.nn.functional.grid_sample(
        grid_weights,
        xyz_norm,
        align_corners=True,
        mode="bilinear",
        padding_mode="zeros",
    )
    offset = offset.squeeze(0).squeeze(-1).squeeze(-1).T
    return offset


def skinning_weights_from_voxel_grid(xyz, grid_center, grid_scale, grid_weights):
    device = xyz.device
    grid_weights = grid_weights.permute(3, 0, 1, 2).unsqueeze(0).to(device)
    grid_weights.expand(1, -1, -1, -1, -1)
    xyz_norm = (xyz - grid_center.to(device)) / grid_scale.to(device)
    xyz_norm = xyz_norm.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    weights = torch.nn.functional.grid_sample(
        grid_weights,
        xyz_norm,
        align_corners=True,
        mode="bilinear",
        padding_mode="zeros",
    )
    skin_wts = weights.squeeze(0).squeeze(-1).squeeze(-1).T
    # skin_wts = torch.nn.functional.softmax(5 * skin_wts, dim=-1)

    skin_wts = skin_wts / skin_wts.sum(dim=-1, keepdim=True)

    # ToDO: add a check that weights sum is never 0
    # assert torch.any(skin_wts.sum(dim = -1) == 0.0)

    """ Debugging Visualization
    colors = visualize_skin_weights(skin_wts)
    dump_points(xyz.reshape(-1, 3), 'test_cano.ply', colors)
    colors = visualize_skin_weights(self.grid_weights.reshape(-1, self.grid_weights.shape[-1]))

    dump_points(self.grid_points.reshape(-1, 3), 'test_grid.ply', colors)
    breakpoint()
    """
    return skin_wts


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_symmetric(L):
    cov = torch.zeros((L.shape[0], 3, 3), dtype=L.dtype, device=L.device)
    cov[:, 0, 0] = L[:, 0]
    cov[:, 0, 1] = L[:, 1]
    cov[:, 0, 2] = L[:, 2]
    cov[:, 1, 1] = L[:, 3]
    cov[:, 1, 2] = L[:, 4]
    cov[:, 2, 2] = L[:, 5]

    cov[:, 1, 0] = L[:, 1]
    cov[:, 2, 0] = L[:, 2]
    cov[:, 2, 1] = L[:, 4]
    return cov


def build_rotation(r, device="cuda"):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r, device="cuda"):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=device)
    R = build_rotation(r, device=device)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def render_gaussians(
    posed_means,
    posed_cov,
    cano_means,
    cano_features,
    cano_opacity,
    camera,
    bg_color,
    colors_precomp=None,
    sh_degree=3,
    tf=None,
    device=torch.device("cuda"),
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            posed_means, dtype=posed_means.dtype, requires_grad=True, device=device
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(camera.fovx * 0.5)
    tanfovy = math.tan(camera.fovy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1,
        viewmatrix=camera.world_view_transform.to(device),
        projmatrix=camera.full_proj_transform.to(device),
        sh_degree=sh_degree,
        campos=camera.camera_center.to(device),
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = cano_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.

    if colors_precomp is None:
        colors_precomp = calculate_colors_from_sh(
            posed_means, cano_features, cano_means, camera, sh_degree, tf
        )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=posed_means,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=None,
        rotations=None,
        cov3D_precomp=posed_cov,
    )

    rendered_image = torch.permute(rendered_image, (1, 2, 0))

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def calculate_colors_from_sh(
    posed_means, cano_features, cano_means, camera, sh_degree, tf
):
    shs_view = cano_features.transpose(1, 2).view(-1, 3, (sh_degree + 1) ** 2)
    camera_center = camera.camera_center.repeat(cano_features.shape[0], 1)

    if tf is not None:
        cam_inv = torch.einsum(
            "nij, nj->ni", torch.linalg.inv(tf), homo(camera_center)
        )[..., :3]
        dir_pp_inv = cano_means - cam_inv
        dir_pp_normalized = dir_pp_inv / dir_pp_inv.norm(dim=1, keepdim=True)
    else:
        dir_pp = posed_means - camera_center
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)

    sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    return colors_precomp

def density_update(model, pred, extent, global_step, bg_color, mask_to_prune=None):
    update_optimizer = False
    with torch.no_grad():
        if mask_to_prune is not None:
            model.prune_points(mask_to_prune)
            torch.cuda.empty_cache()
            update_optimizer = True

            cprint(f"Removing pts : {mask_to_prune.sum()}", "green")
        else:
            viewspace_point_tensor = pred["viewspace_points"]
            visibility_filter = pred["visibility_filter"]
            radii = pred["radii"]
            cameras_extent = extent

            # Densification
            if global_step < model.opts.densify_until_step:
                # Keep track of max radii in image-space for pruning
                model.max_radii2D[visibility_filter] = torch.max(
                    model.max_radii2D[visibility_filter], radii[visibility_filter]
                )

                model.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if (
                    global_step > model.opts.densify_from_step
                    and global_step % model.opts.densification_interval == 0
                ):
                    size_threshold = (
                        model.opts.size_threshold
                        if global_step > model.opts.opacity_reset_interval
                        else None
                    )

                    clean_outliers = global_step == model.opts.remove_outliers_step
                    model.densify_and_prune(
                        model.opts.densify_grad_threshold,
                        model.opts.min_opacity_threshold,
                        cameras_extent,
                        size_threshold,
                        clean_outliers,
                    )
                    print("gaussians changed to ", model.get_xyz.shape[0])
                    update_optimizer = True

                if global_step % model.opts.opacity_reset_interval == 0 or (
                    (bg_color == "white")
                    and global_step == model.opts.densify_from_step
                ):
                    if global_step != 0:
                        model.reset_opacity()
                        update_optimizer = True
    return update_optimizer


def update_learning_rate(optimizer, model, global_step):
    """Learning rate scheduling per step"""
    for param_group in optimizer.param_groups:
        if param_group["name"] == "xyz":
            lr = model.xyz_scheduler_args(global_step)
            param_group["lr"] = lr
            return optimizer


def get_contact_map(pt1, pt2, chunk=1024):
    contact_map = torch.zeros((pt1.shape[0],), dtype=torch.float32, device=pt1.device)
    for i in range(0, pt1.shape[0], chunk):
        contact_map[i : i + chunk] = torch.cdist(pt1[i : i + chunk], pt2).min(1)[0]
    return contact_map


def get_contact_dist(pt1, pt2):
    device = pt1.device

    ti_pt1 = ti.ndarray(shape=(pt1.shape[0], 3), dtype=ti.f32)
    ti_pt2 = ti.ndarray(shape=(pt2.shape[0], 3), dtype=ti.f32)
    ti_contact_map = ti.ndarray(shape=pt1.shape[0], dtype=ti.f32)
    ti_contact_indices = ti.ndarray(shape=pt1.shape[0], dtype=ti.f32)

    ti_pt1.from_numpy(to_numpy(pt1))
    ti_pt2.from_numpy(to_numpy(pt2))

    @ti.kernel
    def calculate_distances(
        ti_pt1: ti.types.ndarray(ndim=2),
        ti_pt2: ti.types.ndarray(ndim=2),
        ti_contact_map: ti.types.ndarray(ndim=1),
        ti_contact_indices: ti.types.ndarray(ndim=1),
    ):
        for i in range(pt1.shape[0]):
            min_dist = 1e9
            for j in range(pt2.shape[0]):
                dist = 0.0
                for k in range(3):
                    dist += (ti_pt1[i, k] - ti_pt2[j, k]) ** 2
                dist = ti.sqrt(dist)
                if dist < min_dist:
                    min_dist = dist
                    ti_contact_indices[i] = j
            ti_contact_map[i] = min_dist

    calculate_distances(ti_pt1, ti_pt2, ti_contact_map, ti_contact_indices)
    contact_map = attach(to_tensor(ti_contact_map.to_numpy()), device)
    contact_indices = attach(to_tensor(ti_contact_indices.to_numpy()), device)
    return contact_map, contact_indices


def update_mask_based_on_outliers(xyz, prob=0.5, neighbors=128):
    pts_path = "./noisy.ply"
    dump_points(xyz, pts_path)
    mask = remove_outliers(pts_path, prob, neighbors)
    return mask


def remove_outliers(pts_path, prob, neighbors):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(pts_path)
    ms.compute_selection_point_cloud_outliers(propthreshold=prob, knearest=neighbors)
    return ms.current_mesh().vertex_selection_array()


def get_cmap(pt1, pt2, c_thresh=0.004, cmap_type="gray"):
    dist, indices = get_contact_dist(pt1, pt2)
    dist = torch.clamp(dist.clone(), 0, c_thresh) / c_thresh
    dist = 1 - dist
    colors = get_colors_from_cmap(to_numpy(dist), cmap_name=cmap_type)[..., :3]
    colors = attach(to_tensor(colors), pt1.device)
    return dist, indices, colors
