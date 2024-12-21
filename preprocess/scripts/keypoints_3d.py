import os
import sys
import json
import shutil
import argparse
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
from glob import glob
import trimesh

# sys.path.append("./pose-estimation")
import src.utils.params as param_utils

sys.path.append("./EasyMocap")
from myeasymocap.operations.triangulate import SimpleTriangulate

# -------------------- Arguments -------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True, type=str)
parser.add_argument("--root_dir", required=True, type=str)
parser.add_argument("--seq_path", required=True, type=str)
parser.add_argument("--params_path", required=True, type=str)
parser.add_argument("--stride", required=True, type=int)
parser.add_argument("--conf_thresh", default=0.2, type=float)
args = parser.parse_args()

base_path = os.path.join(args.root_dir, args.seq_path)
out_dir = os.path.join(args.out_dir, args.seq_path, 'pose_dumps')

params = param_utils.read_params(args.params_path)
cam_names = params[:]["cam_name"]
## Skip views in triangulation which are bad
skip_views = [
            # "brics-sbc-003_cam0",
            # "brics-sbc-003_cam1",
            # "brics-sbc-004_cam0",
            # "brics-sbc-004_cam1",
            # "brics-sbc-005_cam1",
            # "brics-sbc-008_cam0",
            # "brics-sbc-008_cam1",
            # "brics-sbc-009_cam0",
            # "brics-sbc-009_cam1",
            # "brics-sbc-010_cam0",
            "brics-sbc-014_cam1",
            "brics-sbc-015_cam1",
            # "brics-sbc-017_cam1",
            "brics-sbc-019_cam1",
            "brics-sbc-020_cam0",
            "brics-sbc-020_cam1"
            ]

cam_names = cam_names.tolist()
use_idx = [cam_names.index(cam) for cam in cam_names if cam not in skip_views]

# Gets the projection matrices and distortion parameters
projs = []
intrs = []
dists = []
rot = []
trans = []
for i in range(len(params)):
    extr = param_utils.get_extr(params[i])
    intr, dist = param_utils.get_intr(params[i])
    r, t = param_utils.get_rot_trans(params[i])

    rot.append(r)
    trans.append(t)

    intrs.append(intr.copy())

    projs.append(intr @ extr)
    dists.append(dist)

# creates the 3d keypoint directory
keypoints3d_dir = os.path.join(out_dir, "keypoints_3d")
os.makedirs(keypoints3d_dir, exist_ok = True)

# creates the 2d keypoint directory
keypoints2d_dir = os.path.join(out_dir, "keypoints_2d")

projs = [] 
cameras = {}
for i in range(len(params)):
    extr = param_utils.get_extr(params[i])
    intr, dist = param_utils.get_intr(params[i])
    projs.append(intr @ extr)
projs = np.asarray(projs)
cameras['P'] = projs[use_idx]

frames = os.listdir(os.path.join(keypoints2d_dir, os.listdir(keypoints2d_dir)[0]))

for frame in tqdm(frames):
    keypoints2d = [] 
    
    for cam_name in cam_names:
        if cam_name in skip_views:
            continue
        with open(os.path.join(keypoints2d_dir, cam_name, frame), "r") as f:
            keypoints2d.append(json.load(f))
            
    keypoints2d = np.stack(keypoints2d)
    
    # keypoints3d, residuals = triangulate_joints(np.asarray(keypoints2d), cameras['P'], 
    #                                             conf_thresh_start = 0.7, 
    #                                             processor=simple_processor, 
    #                                             min_cams = 10)
    
    
    # thresh = 15
    # try:
    #     keypoints3d, residuals = triangulate_joints(np.asarray(keypoints2d), cameras['P'], 
    #                                                 conf_thresh_start = 0.5, 
    #                                                 processor=ransac_processor, 
    #                                                 min_cams = 8, 
    #                                                 residual_threshold = thresh, 
    #                                                 min_samples = 8)
        
    #     valid = (residuals < thresh).astype(np.uint8)
    # except:
    #     # Easy Mocap for 3D keypoints
    #     triangulation = SimpleTriangulate("iterative")
    #     keypoints3d = triangulation(keypoints2d, cameras, undistort = False)['keypoints3d']
    #     conf = keypoints3d[..., -1]
    #     valid = (conf > args.conf_thresh).astype(np.uint8)
        
    
    ## Easy Mocap for 3D keypoints
    triangulation = SimpleTriangulate("iterative")
    keypoints3d = triangulation(keypoints2d, cameras, undistort = False)['keypoints3d']
    conf = keypoints3d[..., -1]
    valid = (conf > args.conf_thresh).astype(np.uint8)
    
    keypoints3d = np.concatenate([keypoints3d[..., :3], valid[:,None]], axis=-1)
    
    keypt_file = os.path.join(keypoints3d_dir, f"{frame}")
    print(f"Writing 3D keypoints to {keypt_file}")
    with open(keypt_file, "w") as f:
        json.dump(keypoints3d.tolist(), f)
        
    pc = trimesh.PointCloud(vertices=keypoints3d[..., :3])
    _ = pc.export(os.path.join(keypoints3d_dir, f"{frame}.ply"))
        
    