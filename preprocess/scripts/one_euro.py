import json
import os
import numpy as np
import torch
import sys
sys.path.append(".")

import logging
from src.IK.skeleton import KinematicChain
from src.IK.one_euro_filter import OneEuroFilter
from src.utils.extra import detach
from src.utils.general import create_dir, get_chosen_frames
import trimesh
from argparse import ArgumentParser
from glob import glob
from natsort import natsorted
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--root_dir", required=True)
parser.add_argument("--seq_path", required=True)
parser.add_argument("--rest_pose_path", required=True)
parser.add_argument("--output_dir", default="keypoints_3d_filtered")
parser.add_argument("--fcmin", default=0.08, type=float)
parser.add_argument("--beta", default=1.0, type=float)
parser.add_argument("--dcutoff", default=1.0, type=float)
parser.add_argument("--mode", default="joint_angles", type=str)

args = parser.parse_args()

base_path = os.path.join(args.root_dir, args.seq_path)
if args.mode == 'keypoints':
    input_dir = "keypoints_3d"
elif args.mode == 'joint_angles':
    input_dir = "joint_angles"
    
input_dir = os.path.join(base_path, 'pose_dumps', input_dir)

input_paths = natsorted(glob(f"{input_dir}/*.json"))

timestamps = []
if args.mode == "keypoints":
    with open(os.path.join(base_path, "pose_dumps", "chosen_frames.json"), "r") as f:
        chosen_frames = json.load(f)
    keypoints = []
    for frame_id in chosen_frames:
        kpt_path = os.path.join(input_dir, f"{frame_id:08d}.json")
        timestamps.append(frame_id)
        with open(kpt_path) as f:
            keypoints.append(json.load(f))
    keypoints = np.asarray(keypoints)
    
    assert keypoints.shape[0] > 1, "Atleast two frames needed"

    filtered_keypoints = keypoints.copy()
    filtered_keypoints[0] = keypoints[0]
    prev_keypoints = filtered_keypoints[0]
    prev_found = -np.ones(filtered_keypoints.shape[1]) # Keep track of keypoint when last keyframe was seen
    prev_found[~np.isclose(keypoints[0,:,3], 0)] = timestamps[0]

    e_filter = OneEuroFilter(prev_found, prev_keypoints[:,:3], np.zeros((prev_keypoints.shape[0], 3)), min_cutoff=args.fcmin, beta=args.beta, d_cutoff=args.dcutoff)

    output_dir = os.path.join(base_path, 'pose_dumps', args.output_dir)
    create_dir(output_dir)

    with open( os.path.join(output_dir, f"{timestamps[0]:08d}.json"), "w") as f:
        json.dump(filtered_keypoints[0].tolist(), f)

    for frame in tqdm(range(1, keypoints.shape[0])):
        # Find keypoints which are detected in this frame
        present = ~np.isclose(keypoints[frame,:,3], 0)

        # Find keypoints which are seen for first time
        first = np.logical_and(present, prev_found < 0)

        # Update previous seen timestamps
        prev_found[present] = timestamps[frame]
        filtered_keypoints[frame,present,:3] = e_filter(prev_found, keypoints[frame,:,:3])[present]

        # Update keypoints for first time seen keypoints
        filtered_keypoints[frame,first] = keypoints[frame,first]
        
        with open( os.path.join(output_dir, f"{timestamps[frame]:08d}.json"), "w") as f:
            json.dump(filtered_keypoints[frame].tolist(), f)
            
        pc = trimesh.PointCloud(vertices=filtered_keypoints[frame, :, :3])
        pc.export(os.path.join(output_dir, f"{timestamps[frame]:08d}.ply"))
    

elif args.mode == "joint_angles":
    angles = []
    trans = []
    for ja_path in input_paths:
        timestamps.append(int(ja_path.split('/')[-1].split('.')[0]))
        with open(ja_path) as f:
            ja = json.load(f)
            angles.append(ja['angles'])
            trans.append(ja['translation'])
    trans = np.asarray(trans)
    angles = np.asarray(angles)
    jas = np.concatenate([trans[:, None, :], angles], axis = 1)
    
    # filtered_keypoints = keypoints.copy()
    # filtered_keypoints[0] = keypoints[0]
    # prev_keypoints = filtered_keypoints[0]
    prev_found = np.ones(jas.shape[1]) # Keep track of keypoint when last keyframe was seen
    prev_found[:] = timestamps[0]
    start_timestamp = timestamps[0]
    final_timestamp = timestamps[-1]
    prev_jas = jas[0] 
    e_filter = OneEuroFilter(prev_found, prev_jas[:,:3], np.zeros((prev_jas.shape[0], 3)), min_cutoff=args.fcmin, beta=args.beta, d_cutoff=args.dcutoff)

    output_dir = os.path.join(base_path, 'pose_dumps', "joint_angles_filtered")
    create_dir(output_dir)
    
    filtered_jas = jas.copy()
    # filtered_jas.append([jas[0], f"{timestamps[0]:08d}"])
    filtered_jas[0] = jas[0]

    for frame in tqdm(range(1, jas.shape[0])):
        # Update previous seen timestamps
        prev_found[:] = timestamps[frame]
        filtered_jas[frame,:,:3] = e_filter(prev_found, jas[frame,:,:3])
        
        cur_timestamp = timestamps[frame]
        if cur_timestamp > 0.7 * final_timestamp:
            if e_filter.min_cutoff == args.fcmin:
                e_filter.min_cutoff -= 0.06
                print("current fcmin", e_filter.min_cutoff)
   
    with open(args.rest_pose_path, "r") as f:
        bones = json.load(f) 
        
    chain = KinematicChain(bones["bones"])
    
    ik_vis_dir = os.path.join(base_path, "pose_dumps", 'ik_results_filtered')
    os.makedirs(ik_vis_dir, exist_ok=True)
    
    with open(os.path.join(base_path, "pose_dumps", "bone_lens.json")) as f:
        bone_lengths = json.load(f)
        
    for i in range(len(chain.bones)):
        chain.bones[f"bone_{i}"]["len"] = bone_lengths[i] 
        
    for idx, frame_id in enumerate(timestamps):
        out_path = os.path.join(output_dir, f"{frame_id:08d}.json")
        angles = filtered_jas[idx][1:]
        trans = filtered_jas[idx][:1][0]
        
        out_dict = {
            "translation": trans.tolist(), 
            'angles': angles.tolist()
        }
        
        ik_keyp, _, _ = chain.forward(torch.tensor(out_dict["translation"]),
                            torch.tensor(out_dict["angles"])[chain.dof].flatten(),
                            constraint=True)
        
        pc = trimesh.PointCloud(vertices=ik_keyp.detach().cpu().numpy())
        _ = pc.export(os.path.join(ik_vis_dir, f"{frame_id}_ik.ply"))
        
        with open(out_path, "w") as f:
            json.dump(out_dict, f)
        