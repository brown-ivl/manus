import os
import sys
import numpy as np
import argparse
import torch
import trimesh
from tqdm import tqdm
from natsort import natsorted

sys.path.append(os.getcwd())
from src.utils.gaussian_utils import get_cmap
from src.utils.vis_util import get_colors_from_cmap
from src.utils.extra import *
from src.utils.train_utils import *


def get_parser():
    parser = argparse.ArgumentParser(description='Process MANO contacts')
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--subject_name', type=str, required=True)
    parser.add_argument('--object_exp_name', type=str, required=True)
    parser.add_argument('--object_exp_dir', type=str, required=True)
    parser.add_argument('--grasp_path', type=str, required=True)
    parser.add_argument('--results', action='store_true')
    parser.add_argument('--harp', action='store_true')
    return parser


def get_mano_contacts(mano_rest, mano_pose, o_out, frame_id, cmap_type, results_dir, dist=None):
    if dist is not None:
        cmap = get_colors_from_cmap(to_numpy(dist), cmap_name=cmap_type)[..., :3]
        cmap = attach(to_tensor(cmap), dist.device)
    else:
        dist, indices, cmap = get_cmap(to_tensor(mano_pose), to_tensor(o_out), cmap_type=cmap_type)

    dump_mano_results(mano_rest, cmap, frame_id, results_dir)
    return dist, cmap


def dump_mano_results(rest_mano, cmap, frame_id, results_dir):
    cmap = to_numpy(cmap)
    cmap = np.concatenate([cmap, np.ones_like(cmap)[..., :1]], axis=-1)
    cmap = cmap * 255
    path = os.path.join(results_dir, f"{frame_id}.ply")
    dump_mesh(path, rest_mano.vertices, rest_mano.faces, colors=cmap)


def main():
    args = get_parser().parse_args()
    grasp_dir = '/'.join(args.grasp_path.split('/')[:-1])
    root_dir = '/'.join(args.grasp_path.split('/')[:-3])

    exp_root_dir = '/'.join(args.exp_dir.split('/')[:-4])

    # object_ckpt_dir = os.path.join(exp_root_dir, 'object', args.subject_name, args.object_exp_name,
    #                                "checkpoints")

    object_ckpt_dir = os.path.join(args.object_exp_dir, "checkpoints")
    object_ckpt_path = find_best_checkpoint(object_ckpt_dir)
    object_ckpt = load_checkpoint(object_ckpt_path)
    object_pts = object_ckpt[0]['_xyz'].numpy()

    if args.harp:
        all_mano_files = natsorted(glob.glob(os.path.join(grasp_dir, "harp/*", "*.obj")))
    else:
        all_mano_files = natsorted(glob.glob(os.path.join(grasp_dir, "mano/mesh", "*[!_kp].ply")))

    if args.harp:
        mano_rest_path = \
            glob.glob(os.path.join(root_dir, "evals", f'{args.object_exp_name}_action', "harp/*", "*.obj"))[0]
    else:
        if args.results:
            mano_rest_path = os.path.join(root_dir, "mano_rest.ply")
        else:
            mano_rest_path = glob.glob(os.path.join(root_dir, "evals", f'{args.object_exp_name}_action',  "mano/mesh", "*[!_kp].ply"))[0]

    mano_rest = trimesh.load(mano_rest_path, process=False, maintain_order=True)

    dir_name = 'acc_results' if args.results else 'acc_eval'
    out_dir_name = 'harp' if args.harp else 'mano'
    acc_results_dir = os.path.join(args.exp_dir, f"results/eval_results/{out_dir_name}/{dir_name}/")
    dir_name = 'results' if args.results else 'gt_eval'
    eval_results_dir = os.path.join(args.exp_dir, f"results/eval_results/{out_dir_name}/{dir_name}/")

    os.makedirs(acc_results_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)

    cmap_type = 'plasma' if args.results else 'gray'

    acc_dist = []
    for path in tqdm(all_mano_files):
        if args.harp:
            frame_id = int(path.split('/')[-2])
        else:
            frame_id = int(path.split('/')[-1].split('.')[0])
        mano_pose = trimesh.load(path, process=False, maintain_order=True)
        local_mano_pose = mano_pose
        local_mano_rest = mano_rest
        subdiv_iter = 2 if  args.harp else 3
        for _ in range(subdiv_iter):
            local_mano_pose = local_mano_pose.subdivide()
            local_mano_rest = local_mano_rest.subdivide()

        local_mano_pose = local_mano_pose.vertices
        local_mano_rest = local_mano_rest

        dist, cmap = get_mano_contacts(local_mano_rest, local_mano_pose, object_pts, frame_id, cmap_type, eval_results_dir, None)
        acc_dist.append(dist)
        sum_dist = torch.stack(acc_dist).sum(dim=0)
        _, _ = get_mano_contacts(local_mano_rest, local_mano_pose, object_pts, frame_id, cmap_type, acc_results_dir, sum_dist)

    if args.results:
        render_type = 'canonical'
        params_path = "./data/camera_paths/cano_camera.pkl"
    else:
        params_path = os.path.join(root_dir, "evals", f'{args.object_exp_name}_action', "gt_cam.pkl")
        # params_path = os.path.join(root_dir, "calib.evals", "optim_params.txt")
        render_type = 'gt_eval'
        blend_file = os.path.join(root_dir, "evals", f'{args.object_exp_name}_action', "gt_cam.blend")

    cmd_path = f'bash scripts/render_mano.sh {acc_results_dir} {render_type} {params_path} {blend_file}'
    os.system(cmd_path)

if __name__ == '__main__':
    main()
