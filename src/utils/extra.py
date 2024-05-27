import os
import glob
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from termcolor import colored
from collections import defaultdict
import dataclasses
import trimesh
from src.utils.structures import *


def detach(tensor):
    return tensor.detach().cpu()


def attach(tensor, device=torch.device("cpu")):
    return tensor.to(device)


def is_tensor(var):
    return isinstance(var, torch.Tensor)


def is_numpy(var):
    return isinstance(var, np.ndarray)


def is_list(var):
    return isinstance(var, list)


def is_dict(var):
    return isinstance(var, dict)


def is_dataclass(var):
    return dataclasses.is_dataclass(var)


def is_string(var):
    return isinstance(var, str)


def is_dict(var):
    return isinstance(var, dict)


def dict_to_tensor(var, dtype=torch.float32, device=torch.device("cpu")):
    for key, value in var.items():
        if value is not None:
            var[key] = to_tensor(value, dtype, device)
    return var


def to_tensor(var, dtype=torch.float32, device=torch.device("cpu")):
    if is_dataclass(var):
        for key, value in var.__dict__.items():
            if value is not None:
                setattr(var, key, to_tensor(value, dtype, device))
    elif is_numpy(var):
        try:
            if len(var.shape) > 0:
                var = attach(torch.tensor(var, dtype=dtype), device)
        except:
            pass
    elif is_list(var):
        try:
            var = attach(torch.tensor(var, dtype=dtype), device)
        except:
            pass
    elif is_tensor(var):
        try:
            var = attach(var, device)
        except:
            pass
    elif is_dict(var):
        try:
            var = dict_to_tensor(var, dtype, device)
        except:
            pass
    return var


def to_numpy(var):
    if is_dataclass(var):
        for key, value in var.__dict__.items():
            if value is not None:
                setattr(var, key, to_numpy(value))
    elif is_tensor(var):
        var = detach(var).numpy()
    elif is_list(var):
        if is_tensor(var[0]):
            var = detach(var).numpy()
    elif is_dict(var):
        for key, value in var.items():
            if value is not None:
                var[key] = to_numpy(value)
    return var


def directory_check(dir_path):
    if os.path.exists(dir_path):
        print(colored(f"Loading !! {dir_path} ", "green"))
    else:
        print(colored(f"Directory not found!! Stopping process!! ", "red"))
        exit()


def concat_img_array(im1, img2):
    if is_tensor(im1):
        img = torch.cat((im1, img2), dim=0)
    elif is_numpy(im1):
        img = np.concatenate((im1, img2), axis=0).astype(np.uint8)
    return img


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def dump_points(points, path, colors=None):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    pc = trimesh.PointCloud(points)
    if colors is not None:
        pc.visual.vertex_colors = colors[..., :4]
    pc.export(path)


def load_pts(pts_path):
    pts = trimesh.load(pts_path, process=False, maintain_order=True)
    return pts.vertices


def dump_mesh(path, verts, faces, normals=None, colors=None):
    if isinstance(verts, torch.Tensor):
        verts = to_numpy(verts)
        faces = to_numpy(faces)
        if normals:
            normals = to_numpy(normals)
        if colors:
            colors = to_numpy(colors)
    mesh = trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_normals=normals, vertex_colors=colors
    )
    mesh.export(path)


def dump_image(img, path=None, return_img=False):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.max() <= 1.0:
        img = img * 255
    img = img.astype(np.uint8)
    if return_img:
        return img
    if path is None:
        raise Exception("Save path is None!!")

    img = Image.fromarray(img)
    img.save(path)


def cprint(text, color="green"):
    print(colored(text, color))


def visualize_skin_weights(skin_weights):
    if isinstance(skin_weights, torch.Tensor):
        skin_weights = to_numpy(skin_weights)

    cmap = plt.get_cmap("tab20c")
    num_bones = skin_weights.shape[-1]
    color_values = np.linspace(0, 1, num_bones)
    color_map = cmap(color_values)
    indices = np.argmax(skin_weights, axis=-1)
    colors = color_map[indices]
    return colors


def convert_to_batch(object_list):
    dict_keys = object_list[0].__dict__.keys()
    batch_dict = defaultdict(list)

    for obj in object_list:
        for key in dict_keys:
            batch_dict[key].append(getattr(obj, key))

    for key in dict_keys:
        if isinstance(batch_dict[key][0], torch.Tensor):
            batch_dict[key] = torch.stack(batch_dict[key], dim=0)
        elif isinstance(batch_dict[key][0], np.ndarray):
            batch_dict[key] = np.stack(batch_dict[key], axis=0)
        elif isinstance(batch_dict[key][0], list):
            batch_dict[key] = np.array(batch_dict[key])
    return dict(batch_dict)


def find_best_checkpoint(check_dir, sort_by="epoch"):  # loss, epoch
    all_checkpoints = glob.glob(os.path.join(check_dir, "*.ckpt"))

    if len(all_checkpoints) == 0:
        errno = f"no checkpoint found at {check_dir}"
        raise FileNotFoundError(errno)
    else:
        epochs = []
        converged_losses = []
        converged_losses_list = []
        steps = []
        for chk_path in all_checkpoints:
            epoch = chk_path.split("/")[-1].split("epoch=")[-1].split("-")[0]
            epochs.append(epoch)
            step = chk_path.split("/")[-1].split("step=")[-1].split("-")[0]
            steps.append(step)
            converged_loss = chk_path.split("/")[-1].split(".ckpt")[0].split("=")[-1]
            converged_losses_list.append(converged_loss)
            converged_losses.append(np.array(converged_loss).astype(np.float64))

        if sort_by == "loss":
            min_loss_idx = np.argmin(converged_losses)
            min_loss = min(converged_losses)
            converged_losses = np.array(converged_losses, dtype=np.float64)
            mask = np.array(converged_losses) == min_loss
            steps_array = np.array(steps)
            max_step = max(steps_array[mask])
            idx = steps_array.tolist().index(max_step)
            best_epoch = epochs[idx]
            best_step = steps[idx]
            min_loss_str = converged_losses_list[min_loss_idx]
            checkpoint_name = (
                f"epoch={best_epoch}-step={best_step}-loss={min_loss_str}.ckpt"
            )
            return os.path.join(check_dir, checkpoint_name)
        elif sort_by == "epoch":
            max_epoch = max(epochs)
            for ckpt_path in all_checkpoints:
                if f"epoch={max_epoch}" in ckpt_path:
                    return ckpt_path


def homo(points):
    return torch.nn.functional.pad(points, (0, 1), value=1)


def create_voxel_grid(grid_boundary, resolution):
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x
    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    return grid_points


def create_skinning_grid(d, h, w, device="cpu"):
    x_range = (
        (torch.linspace(-1, 1, steps=w, device=device))
        .view(1, 1, 1, w)
        .expand(1, d, h, w)
    )  # [1, H, W, D]
    y_range = (
        (torch.linspace(-1, 1, steps=h, device=device))
        .view(1, 1, h, 1)
        .expand(1, d, h, w)
    )  # [1, H, W, D]
    z_range = (
        (torch.linspace(-1, 1, steps=d, device=device))
        .view(1, d, 1, 1)
        .expand(1, d, h, w)
    )  # [1, H, W, D]
    grid = (
        torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3, -1).permute(0, 2, 1)
    )

    return grid[0]


def dump_video(frames, out_path):
    frame_size = (frames[0].shape[1], frames[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(out_path, fourcc, 30.0, frame_size)

    for image in frames:
        out.write(image[..., ::-1].astype(np.uint8))
    out.release()
    print("video is saved " + out_path)
