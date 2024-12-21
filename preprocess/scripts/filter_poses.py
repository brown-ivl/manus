"""
Filter out bad poses based upon some simple heuristics
"""
import sys
sys.path.append(".")

import numpy as np
import json
import os
import re
import argparse
from glob import glob
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", "-r", required=True, type=str)
parser.add_argument("--seq_path", "-s", required=True, type=str)
parser.add_argument("--bin_size", type=int, default=5)
parser.add_argument("--start_frame_for_grasp", type=int, default=200)
parser.add_argument("--ignore_missing_tip", action="store_true", help="Should a missing fingertip be allowed")

args = parser.parse_args()

if 'session' in args.seq_path:
    args.seq_path = re.sub("/.*?_session_","/", args.seq_path)

base_path = os.path.join(args.root_dir, args.seq_path)
keypoints3d_dir = os.path.join(base_path, "pose_dumps", "keypoints_3d")
kyps_files = list(sorted(glob(f"{keypoints3d_dir}/*.json")))

if 'grasps' in args.seq_path:
    new_kyps_files = []
    for path in kyps_files:
        frame_id = int(path.split('/')[-1].split('.')[0])
        if frame_id < args.start_frame_for_grasp:
            continue
        new_kyps_files.append(path)
    kyps_files = deepcopy(new_kyps_files)

chosen_frames = []

# Indices of fingers
# If none of the keypoints are present for any finger, skip the frame
finger_idx = [list(range(2, 5))] + [ list(range(i, i+4)) for i in range(5, 18, 4) ]

# Indices of finger tips
# If any of them are missing, skip the frame
tip_idx = [4, 8, 12, 16, 20]

for i in range(0, len(kyps_files), args.bin_size):
    # Read back the files
    kyps_3d = []
    start_frame_id = int(os.path.basename(kyps_files[i]).split(".")[0])
    for j in range(args.bin_size):
        if i+j >= len(kyps_files):
            break
        with open(kyps_files[i+j], "r") as f:
            kyps_3d.append(np.asarray(json.load(f)))

    kyps_3d = np.stack(kyps_3d)

    # Remove frames which have complete finger missing
    to_use = np.ones(kyps_3d.shape[0], dtype=bool)
    for idx in finger_idx:
        to_use = np.logical_and(to_use, np.any(kyps_3d[:,idx,3], axis=1))
        
        
    # Remove frames which have any of the finger tips missing
    if not args.ignore_missing_tip:
        to_use = np.logical_and(to_use, np.all(kyps_3d[:,tip_idx,3], axis=1))
        if not np.any(to_use):
            continue
    
    # Find frame with maximum number of detected keypoints
    unfound_count = kyps_3d.shape[1] * np.ones(kyps_3d.shape[0])
    unfound_count[to_use] = np.count_nonzero(np.isclose(kyps_3d[to_use,:,3], 0), axis=1)
    chosen_frame = kyps_files[i+np.argmin(unfound_count)]
    chosen_frame_id = int(os.path.basename(chosen_frame).split(".")[0])
    
    # if (kyps_3d[0][:, -1].sum() == 21.0):
    chosen_frames.append(chosen_frame_id)
    
chosen_path = os.path.join(base_path, "pose_dumps", "chosen_frames.json")
with open(chosen_path, "w") as f:
    json.dump(chosen_frames, f, indent=2)
   
print("=================================================================================") 
print(f"Saved chosen frames to {chosen_path}")
print(f"Number of frames: {len(chosen_frames)}")
print("=================================================================================") 