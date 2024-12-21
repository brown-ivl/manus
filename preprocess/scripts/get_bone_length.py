import sys
sys.path.append(".")
"""
Gets the bone lengths by averaging out the derived bone lengths across all actions
"""
import os
import logging
import json
import torch
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from src.utils.general import get_chosen_frames
from src.IK.skeleton import KinematicChain
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--root_dir", required=True)
parser.add_argument("--input_dir", default="keypoints_3d")
parser.add_argument("--seq_dir", default="actions")
parser.add_argument("--use_filtered", action="store_true", help="Whether to use only filtered keypoints (binned)")
parser.add_argument("--rest_pose_path", required=True)
args = parser.parse_args()

with open(args.rest_pose_path, "r") as f:
    bones = json.load(f)
    
chain = KinematicChain(bones["bones"])

keypoints = []
base_dir = os.path.join(args.root_dir, args.seq_dir)
actions = glob(base_dir + '/*')

if os.path.exists(os.path.join(args.root_dir, "bone_lens.json")):
    print("Skipping Bone Length estimation!!! ")
    exit(0)

for action_path in tqdm(actions, desc="Loading keypoints"):
    action = os.path.basename(action_path)
    keypoints_dir = os.path.join(base_dir, action, "pose_dumps", args.input_dir)
    keypoints_path = natsorted(glob(f"{keypoints_dir}/*.json"))

    if args.use_filtered:
        keypoints_path = get_chosen_frames(keypoints_path, os.path.join(base_dir, action, "pose_dumps", "chosen_frames.json"))

    for keypoint_path in keypoints_path:
        with open(keypoint_path) as f:
            keypoints.append(json.load(f))
            
assert len(keypoints) != 0, "Atleast one frame needed"

keypoints = torch.tensor(keypoints)
chain.update_bone_lengths(keypoints)

with open(os.path.join(args.root_dir, "bone_lens.json"), "w") as f:
    lens = [chain.bones[f"bone_{i}"]["len"] for i in range(len(chain.bones))]
    json.dump(lens, f)
    
import trimesh 
import numpy as np
mean_keypoints = np.mean(keypoints.numpy(), axis = 0)[..., :3].reshape(-1, 3)
pc = trimesh.PointCloud(mean_keypoints)
pc.export(os.path.join(args.root_dir, "bone_keypoints.ply"))
