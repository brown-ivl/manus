"""
Use inverse kinematics to obtain joint angles from given 3D keypoints
"""
import sys
sys.path.append(".")

import json
import trimesh
import re
import os
import logging
import torch
from src.utils.parser import add_common_args
from src.IK.skeleton import KinematicChain
from src.utils.general import create_dir, get_chosen_frames
from natsort import natsorted
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
add_common_args(parser)
parser.add_argument("--input_dir", default="keypoints_3d")
parser.add_argument("--output_dir", default="joint_angles")
parser.add_argument("--use_prev", action="store_true", help="Whether to use previous frames predictions as next frames initialization")
parser.add_argument("--max_cpu", default=0, type=int)
parser.add_argument("--gpu", action="store_true", help="Whether to use GPU acceleration")
parser.add_argument("--use_filtered", action="store_true", help="Whether to use only filtered keypoints (binned)")
parser.add_argument("--use_common_bone_len", action="store_true")
parser.add_argument("--rest_pose_path", required=True)
args = parser.parse_args()

if args.max_cpu > 0:
    torch.set_num_threads(args.max_cpu)

if args.gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

logging.info(f"Using {device}")

with open(args.rest_pose_path, "r") as f:
    bones = json.load(f)

chain = KinematicChain(bones["bones"], device=device)
angles = torch.zeros((len(bones["bones"]), 3), device=device)

if 'session' in args.seq_path:
    args.seq_path = re.sub("/.*?_session_","/", args.seq_path)
    
base_path = os.path.join(args.root_dir, args.seq_path)

keypoints_dir = os.path.join(base_path, "pose_dumps", args.input_dir)
keypoints_path = natsorted(glob(f"{keypoints_dir}/*.json"))
keypoints = []

if args.use_filtered:
    keypoints_path = get_chosen_frames(keypoints_path, os.path.join(base_path, "pose_dumps", "chosen_frames.json"))
    
for keypoint_path in keypoints_path:
    with open(keypoint_path) as f:
        keypoints.append(json.load(f))
        
print("Found {} frames".format(len(keypoints)), "at {}".format(keypoints_dir))

assert len(keypoints) != 0, "Atleast one frame needed"

output_dir = os.path.join(base_path, "pose_dumps", args.output_dir)
create_dir(output_dir)

ik_vis_dir = os.path.join(base_path, "pose_dumps", "ik_results")
create_dir(ik_vis_dir)

keypoints = torch.tensor(keypoints, device=device)

if args.use_common_bone_len:
    with open(os.path.join(args.root_dir, "bone_lens.json")) as f:
        lens = json.load(f)
        for i in range(len(chain.bones)):
            chain.bones[f"bone_{i}"]["len"] = lens[i]
else:
    chain.update_bone_lengths(keypoints)

with open(os.path.join(base_path, "pose_dumps", "bone_lens.json"), "w") as f:
    lens = [chain.bones[f"bone_{i}"]["len"] for i in range(len(chain.bones))]
    json.dump(lens, f)
    

angles = torch.zeros((21, 3), device=device)

trans_init = torch.zeros(3, device=device)
angles_init = angles.flatten()

threshold=1e-7

for frame in tqdm(range(0, keypoints.shape[0], args.stride)):
    target = keypoints[frame, :, :3]
    to_use = ~torch.isclose(keypoints[frame, :, 3], torch.tensor(0.0))
    
    trans_params, angle_params = chain.IK(target, to_use, constraint=True, limit=True, lr=1e-2, trans_init=trans_init, angles_init=angles_init, threshold=threshold)
    # chain.plot_skeleton(trans_params, angle_params, target)

    ik_keyp, heads, tails = chain.forward(trans_params, angle_params)
    
    if args.use_prev:
        trans_init = trans_params
        angles_init = angle_params
        
    ## Dump Keypoints to file
    name = os.path.basename(keypoints_path[frame]).replace(".json", "")
    pc = trimesh.PointCloud(vertices=target.detach().cpu().numpy())
    _ = pc.export(os.path.join(ik_vis_dir, f"{name}_kp.ply"))
    
    pc = trimesh.PointCloud(vertices=ik_keyp.detach().cpu().numpy())
    _ = pc.export(os.path.join(ik_vis_dir, f"{name}_ik.ply"))

    with open(
        os.path.join(output_dir, os.path.basename(keypoints_path[frame])), "w"
    ) as f:
        json.dump({
            "translation": trans_params.tolist(),
            "angles": angle_params.reshape(-1, 3).tolist()
        }, f)