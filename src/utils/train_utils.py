import hydra
import torch
import numpy as np
import trimesh
import math
from omegaconf import OmegaConf
from src.utils.extra import to_tensor, cprint, dump_points, visualize_skin_weights
from pysdf import SDF


def default_collate_fn(data):
    return dict(data[0])


def setup_dataloaders(
    dataset_instance, batch_size, num_workers, split, shuffle=True, pin_memory=True
):
    data = hydra.utils.instantiate(dataset_instance, _recursive_=False, split=split)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=default_collate_fn,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    return data, loader


def load_modules(module, opts, mode, model, checkpoint):
    num_gaussians = get_num_gaussians_from_checkpoint(checkpoint)
    model.opts.num_gaussians = num_gaussians

    opts.model = model
    config_dict = {
        "_target_": module,
        "opts": opts,
        "mode": mode,
    }

    conf = OmegaConf.create(config_dict)

    module_instance = hydra.utils.instantiate(conf, _recursive_=False)
    module_instance = module_instance.load_from_checkpoint(
        checkpoint_path=checkpoint, opts=opts, mode=mode
    )
    return module_instance

def init_mano_weights(points, data, neighbors=20, filter_grid=True):
    mano_verts = data["verts"]
    weights = data["weights"]
    mano_faces = data["face"]

    mask = None
    if filter_grid:
        sdf_func = SDF(mano_verts, mano_faces)
        sdf_value = sdf_func(points.detach().cpu().numpy())
        threshold = -0.02
        mask = sdf_value > threshold
        dump_points(points[mask], "./grid_points_filtered.ply")

    mano_verts = to_tensor(data["verts"])

    colors = visualize_skin_weights(weights)
    dump_points(mano_verts, "./mano_verts.ply", colors=colors)

    ## 20 bones
    mano_to_ours = [13, 14, 14, 15, 0, 1, 2, 3, 0, 4, 5, 6, 0, 10, 11, 12, 0, 7, 8, 9]

    init_weights = weights[..., mano_to_ours]
    dist = torch.cdist(points, mano_verts)
    knn = dist.topk(neighbors, largest=False)
    indices = knn[1]
    weights = np.mean(init_weights[indices], axis=1)

    if filter_grid:
        ## augment vertices of MANO with grid points
        # outside_points = points[~mask]
        extra_weights = np.zeros((weights.shape[0], 1))
        weights = np.concatenate([weights, extra_weights], axis=-1)
        outside_weights = weights[sdf_value < threshold]
        outside_weights[:, :] = 0
        outside_weights[:, -1] = 1
        weights[sdf_value < threshold] = outside_weights

    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights, mask


def sample_gaussians_on_mano(mano_data, sample_size):
    sample_size = sample_size * 20
    mesh = trimesh.Trimesh(
        mano_data["verts"], mano_data["face"], process=False, maintain_order=True
    )
    for _ in range(5):
        mesh = mesh.subdivide()
    points = np.asarray(mesh.vertices[:sample_size])
    noise = np.random.normal(0, 0.003, points.shape)
    points = points + noise
    points_colors = torch.rand(points.shape)
    return to_tensor(points), to_tensor(points_colors)


def sample_gaussians_on_bones_func(bones_rest, sample_size):
    ## Make covariance matrix with heads-tails as the major axis
    heads = bones_rest.heads
    tails = bones_rest.tails

    ## Sample points on bones
    bones_mid = (heads + tails) / 2
    dist = torch.linalg.norm(tails - heads, dim=1, keepdim=True)
    scale = torch.cat([dist / 5, dist / 4, dist / 4], dim=-1)
    # scale = torch.cat([dist / 6, dist / 4, dist / 6], dim=-1)
    # scale = torch.cat([dist / 8, dist / 5, dist / 10], dim=-1)
    scale = torch.diag_embed(scale)

    rot = bones_rest.transforms[:, :3, :3]
    cov = torch.eye(3).unsqueeze(0).repeat(bones_mid.shape[0], 1, 1)
    cov = scale @ cov @ scale.transpose(1, 2)
    cov = rot @ cov @ rot.transpose(1, 2)
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(bones_mid, cov)
    points_bones = mvn.sample((sample_size,)).reshape(-1, 3)

    ## Sample points on joints
    bones_mid = heads
    dist = torch.linalg.norm(tails - heads, dim=1, keepdim=True)
    scale = torch.cat([dist / 6, dist / 4, dist / 6], dim=-1)
    scale = torch.diag_embed(scale)

    rot = bones_rest.transforms[:, :3, :3]
    cov = torch.eye(3).unsqueeze(0).repeat(bones_mid.shape[0], 1, 1)
    cov = scale @ cov @ scale.transpose(1, 2)
    cov = rot @ cov @ rot.transpose(1, 2)
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(bones_mid, cov)
    points_heads = mvn.sample((sample_size // 2,)).reshape(-1, 3)

    points = torch.cat([points_bones, points_heads], dim=0)
    points_colors = torch.rand(points.shape)
    return points, points_colors


def load_models(model, dataset, checkpoint, model_type, clean_gaussians=False):
    if model_type == "hand":
        return load_hand_model(model, dataset, checkpoint, clean_gaussians)
    elif model_type == "object":
        return load_object_model(model, dataset, checkpoint, clean_gaussians)


def mask_model_state(model_state, mask):
    num_pts = model_state["_xyz"].shape[0]

    for key, value in model_state.items():
        if value.shape[0] == num_pts:
            model_state[key] = value[mask]
        else:
            raise ValueError("Model state shape mismatch")
    return model_state


def get_num_gaussians_from_checkpoint(ckpt_path):
    weights, params = load_checkpoint(ckpt_path)
    return params["num_gaussians"]


def remove_nans_from_checkpoint(checkpoint):
    ## If there are any NaNs remove those NaNs in the positions
    state_dict = checkpoint["state_dict"]
    nan_mask = torch.zeros(
        (state_dict["model._xyz"].shape[0]),
        dtype=torch.bool,
        device=state_dict["model._xyz"].device,
    )
    for key in state_dict.keys():
        value = state_dict[key]
        if len(value.shape) == 2:
            tmp_mask = torch.isnan(value).any(dim=-1)
        elif len(value.shape) == 3:
            tmp_mask = torch.isnan(value).any(dim=-1).any(dim=-1)
        elif len(value.shape) == 4:
            tmp_mask = torch.isnan(value).any(dim=-1).any(dim=-1).any(dim=-1)

        assert tmp_mask.shape[0] == nan_mask.shape[0]
        nan_mask = torch.logical_or(nan_mask, tmp_mask)

    for key in state_dict.keys():
        state_dict[key] = state_dict[key][~nan_mask]

    ## Update the num Gaussians
    checkpoint["extra_params"]["num_gaussians"] = state_dict["model._xyz"].shape[0]
    return checkpoint


def load_checkpoint(ckpt_path, device=torch.device("cpu")):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = remove_nans_from_checkpoint(ckpt)
    model_weights = ckpt["state_dict"]
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)

    if "extra_params" in ckpt:
        extra_params = ckpt["extra_params"]
    else:
        extra_params = {}
    return model_weights, extra_params


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def sample_on_bones(heads, tails, samples=10, range=(0.1, 0.9)):
    t_vals = torch.rand(heads.shape[0], samples) * (range[1] - range[0]) + range[0]
    t_vals = t_vals.to(heads.device)
    samples = (
        heads[:, None, :] + (tails[:, None, :] - heads[:, None, :]) * t_vals[:, :, None]
    )
    return samples
