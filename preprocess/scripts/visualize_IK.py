import os
import cv2
import glob
import json
import re
import torch
import logging
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(".")
from src.utils import params as param_utils
from src.utils import visualize
from src.utils.extra import detach
from src.utils.general import create_dir
from src.IK.skeleton import KinematicChain
from src.utils.parser import add_common_args
from natsort import natsorted
from src.utils.reader import Reader
from argparse import ArgumentParser

import ipdb

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--camera_ids", "-c", nargs="+", default=["all", "brics-sbc-021_cam1"])
parser.add_argument("--joint_angles", default="joint_angles", type=str)
parser.add_argument("--separate_calib", action="store_true")
parser.add_argument("--keypoints_3d", default="keypoints_3d", type=str)
parser.add_argument("--output_dir", default="pose_dumps/result", type=str)
parser.add_argument("--rest_pose_path", required=True, type=str)
parser.add_argument("--fps", default=30, type=int)
parser.add_argument("--params_path", required =True, type=str)
parser.add_argument("--raw_video", required= True, type = str)
add_common_args(parser)
args = parser.parse_args()

if 'session' in args.seq_path:
    args.seq_path = re.sub("/.*?_session_","/", args.seq_path)
    
base_path = os.path.join(args.root_dir, args.seq_path)

# Get list of camera_ids that we have to visualize for
params = param_utils.read_params(args.params_path)
if "all" in args.camera_ids:
    cam_names = params[:]["cam_name"]
else:
    cam_names = list(set(args.camera_ids))

# Get list of keypoint files and joint angle files
kyp_dir = os.path.join(base_path, "pose_dumps", args.keypoints_3d)
ja_dir = os.path.join(base_path, "pose_dumps", args.joint_angles)
ja_files = natsorted(glob.glob(f"{ja_dir}/*.json"))

# Load kinematic chain
with open(args.rest_pose_path, "r") as f:
    bones = json.load(f)

chain = KinematicChain(bones["bones"])
with open(os.path.join(base_path, "pose_dumps", "bone_lens.json")) as f:
    bone_lengths = json.load(f)
for i in range(len(chain.bones)):
    chain.bones[f"bone_{i}"]["len"] = bone_lengths[i]

# Get the frames to project
with open(os.path.join(base_path, "pose_dumps", "chosen_frames.json")) as f:
    chosen_frames = json.load(f)

# Create a mapping from cam_name to params
cam2params = {}
found = False
for cam_name in cam_names:
    # Search for params of given camera_id
    for param in params:
        if param["cam_name"] == cam_name:
            found = True
            cam2params[cam_name] = param
            break

    assert found, f"Parameters for camera {cam_name} not found"

# Create reader for loading images
reader = Reader("video", args.raw_video)
idx = 0
for frames, frame_idx in tqdm(reader(chosen_frames), total=len(chosen_frames)):
    for cam_name in cam_names:
        image = frames[cam_name]

        intr, dist = param_utils.get_intr(cam2params[cam_name])
        extr = param_utils.get_extr(cam2params[cam_name])
        proj = intr @ extr

        if args.undistort:
            dist_intr, _ = param_utils.get_undistort_params(intr, dist, (image.shape[1], image.shape[0]))
            image = param_utils.undistort_image(intr, dist_intr, dist, image)
            proj = dist_intr @ extr

        # Load keypoint files
        kyp_file = os.path.join(kyp_dir, f"{frame_idx:08d}.json")
        with open(kyp_file) as f:
            kyp = np.asarray(json.load(f))
        with open(ja_files[idx]) as f:
            ja = json.load(f)

        kyp_result = visualize.plot_keypoints_2d(kyp[kyp[:,3] > 1e-6,:3], image, chain, proj, plot_bones=False)

        joints, _, _ = chain.forward(torch.tensor(ja["translation"]),
                            torch.tensor(ja["angles"])[chain.dof].flatten(),
                            constraint=True)

        ja_result = visualize.plot_keypoints_2d(detach(joints), image, chain, proj)

        final = np.concatenate([
            np.concatenate([kyp_result, ja_result], axis=0),
        ], axis=1)
        final = cv2.resize(final, (int(final.shape[1] / 2), int(final.shape[0] / 2)))


        kyp_output_dir = os.path.join(base_path, args.output_dir, cam_name,  "keypoints_3d")
        ja_output_dir = os.path.join(base_path, args.output_dir, cam_name, "joint_angles")
        final_output_dir = os.path.join(base_path, args.output_dir, cam_name,  "final")
        os.makedirs(kyp_output_dir, exist_ok = True)
        os.makedirs(ja_output_dir, exist_ok = True)
        os.makedirs(final_output_dir, exist_ok = True)

        cv2.imwrite(os.path.join(kyp_output_dir, f"{frame_idx:08d}.jpg"), kyp_result)
        cv2.imwrite(os.path.join(ja_output_dir, f"{frame_idx:08d}.jpg"), ja_result)
        cv2.imwrite(os.path.join(final_output_dir, f"{frame_idx:08d}.jpg"), final)

    idx += 1
