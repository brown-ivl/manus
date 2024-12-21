import os
import sys
import cv2
import json
import torch
import shutil
import re
import argparse
import tempfile
import platform
import numpy as np
from easydict import EasyDict as edict
from joblib import Parallel, delayed

from tqdm import tqdm
from glob import glob

from src.utils.reader import Reader
import src.utils.params as param_utils
from src.utils.parser import add_common_args

sys.path.append("./AlphaPose_mp")
from alphapose.models import builder
from scripts.keypoints_2d import keypoints_2d
from alphapose.utils.config import update_config


import warnings
warnings.filterwarnings("ignore")

# -------------------- Arguments -------------------- #
parser = argparse.ArgumentParser(description='AlphaPose Keypoints Parser')
parser.add_argument("--separate_calib", action="store_true")
parser.add_argument("--allow_missing_finger", action="store_true")
parser.add_argument("--bin_size", type=int, default=5)
parser.add_argument("--use_optim_params", action="store_true")
parser.add_argument("--all_frames", default=False, action="store_true")
parser.add_argument("--params_path", required=True, type=str)
parser.add_argument("--cfg", default="AlphaPose_mp/configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml",
                    help="config file name")
parser.add_argument("--checkpoint", default="AlphaPose_mp/pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth", 
                    help="checkpoint file name")

add_common_args(parser)
# parser.add_argument("--out_dir", required=True, type=str)
parser.add_argument("--dump_image", action="store_true")
args = parser.parse_args()

# Alpha Pose Arguments
ap_args = argparse.Namespace(
    sp=True, # use single process for pytorch (using single process because multiprocessing breaks :( )
    inputpath='', # image directory
    outputpath='', # output directory
    inputimg='', # image name
    save_img=False, # save resulting image
    checkpoint=args.checkpoint, # alpha pose checkpoint path
    vis=False, # visualize image
    showbox=False, # visualize bounding box
    profile=False, # add speed profiling at screen output
    format='open', # saving output format
    min_box_area=0, # min box area to filter out
    posebatch=64, # pose estimation maximum batch size PER GPU
    eval=False, # save the result json as coco format, using image index(int) instead of image name(str)
    gpus='0', # choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)
    device='', # device
    qsize=1024, # the length of result buffer, where reducing it will lower requirement of cpu memory
    flip=False, # enable flip testing
    debug=False, # print detail information
    vis_fast=False, # use fast rendering
    pose_flow=False, # track humans in video with PoseFlow
    pose_track=False # track humans in video with reid
)

if platform.system() == 'Windows':
    ap_args.sp = True

ap_args.gpus = [int(i) for i in ap_args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
ap_args.device = torch.device("cuda:" + str(ap_args.gpus[0]) if ap_args.gpus[0] >= 0 else "cpu")
ap_args.posebatch = ap_args.posebatch * len(ap_args.gpus)
ap_args.tracking = ap_args.pose_track or ap_args.pose_flow

base_path = os.path.join(args.root_dir, args.seq_path)
image_base = os.path.join(base_path, "synced")

if 'session' in args.seq_path:
    args.seq_path = re.sub("/.*?_session_","/", args.seq_path)
    
    
out_dir = os.path.join(args.out_dir, args.seq_path, 'pose_dumps')

# Create temporary directory for intermediate results 
# EDIT: CREATE OPTION TO SAVE THE IMAGES HERE RATHER THAN SAVING THEM IN TEMP
tmpdir = tempfile.mkdtemp()
tmp_image_pth = os.path.join(tmpdir, "output")
if os.path.exists(tmp_image_pth):
    assert False, "Temporary directory already exists"

os.mkdir(tmp_image_pth)

tmp_ap_keypoints_pth = os.path.join(tmpdir, "keypoints_2d")
if not os.path.exists(tmp_ap_keypoints_pth):
    os.mkdir(tmp_ap_keypoints_pth)

params = param_utils.read_params(args.params_path)
cam_names = params[:]["cam_name"]
cam_names = cam_names.tolist()

cam_param_path = os.path.join(tmpdir, "param/")
try:
    shutil.rmtree(cam_param_path)
except FileNotFoundError:
    pass
os.makedirs(cam_param_path)

# Gets the projection matrices and distortion parameters
projs = []
intrs = []
dist_intrs = []
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
    if args.undistort:
        dist_intrs.append(intr.copy())

    projs.append(intr @ extr)
    dists.append(dist)

# Get files to process
reader = Reader(args.input_type, image_base)

# creates the 2d keypoint directory
keypoints2d_dir = os.path.join(out_dir, "keypoints_2d")
try:
    shutil.rmtree(keypoints2d_dir)
except FileNotFoundError:
    pass
os.makedirs(keypoints2d_dir)
# creates the bounding box directory and camera directories within
bbox_dir = os.path.join(out_dir, "bboxes")
try:
    shutil.rmtree(bbox_dir)
except FileNotFoundError:
    pass
os.makedirs(bbox_dir)

# Load Alpha Pose model
cfg = update_config(args.cfg)
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
print('Loading pose model from %s...' % (args.checkpoint,))
pose_model = pose_model.to(ap_args.device)
pose_model.load_state_dict(torch.load(args.checkpoint, map_location=ap_args.device))

pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)

if (args.all_frames):
    chosen_frames = range(0, reader.frame_count, 1)
else:
    if args.end == -1:
        args.end = reader.frame_count 
    chosen_frames = range(args.start, args.end, args.stride)

projs = [] 
cameras = {}
for i in range(len(params)):
    extr = param_utils.get_extr(params[i])
    intr, dist = param_utils.get_intr(params[i])
    projs.append(intr @ extr)
projs = np.asarray(projs)

print(f"Processing Frames at {keypoints2d_dir}")

# frames = {}
# for idx in chosen_frames:
#     frames[idx] = {}
#     for i in range(len(params)):
#         cam_name = cam_names[i]
#         path = os.path.join(image_base, cam_name, f"{idx:08d}.jpg")
#         img = cv2.imread(path)
#         frames[idx][cam_name] = img 
        

for frames, idx in tqdm(reader(chosen_frames), total=len(chosen_frames)):
    assert len(frames) <= 54, "Total number of images per frame shouldn't exceed 53"
    frame = f"{idx:08d}"
    
    # Copy images
    # for i in range(len(params)):
    def process(i):
        cam_name = cam_names[i]
        image = frames[cam_name]
        output_path = os.path.join(tmp_image_pth, f"{cam_name}.jpg")
        if args.dump_image:
            img_out_dir = os.path.join(args.out_dir, args.seq_path, 'images', 'image', cam_name)
            os.makedirs(img_out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(img_out_dir, f"{frame}.jpg"), image)
        if args.undistort:
            image = param_utils.undistort_image(intrs[i], dist_intrs[i], dists[i], image)
        cv2.imwrite(output_path, image)
    
    Parallel(n_jobs=8)(delayed(process)(i) for i in range(len(params)))
    
    # AlphaPose 2D keypoints
    outdir = os.path.join(tmp_ap_keypoints_pth, frame)
    ap_args.inputpath = tmp_image_pth
    ap_args.outputpath = outdir
    keypoints_2d(ap_args, cfg, pose_model, pose_dataset)
    
    keypoints2d = []
    for cam in cam_names:
        ap_keypoints_path = os.path.join(tmp_ap_keypoints_pth, frame, "sep-json", f"{cam}.json")
        with open(ap_keypoints_path, "r") as f:
            data = json.load(f)
            cam_keypoints = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3).tolist()
            bbox = data['people'][0]['bbox']
            keypoints2d.append(cam_keypoints)

        # save the 2d keypoints as just a list (original pipeline format)
        keypoint2d_path = os.path.join(keypoints2d_dir, cam)
        if not os.path.exists(keypoint2d_path):
            os.mkdir(keypoint2d_path)
        with open(os.path.join(keypoint2d_path, f"{frame}.json"), "w") as f:
                    json.dump(cam_keypoints, f)

        # saves the bounding boxes
        bbox_path = os.path.join(bbox_dir, cam)
        if not os.path.exists(bbox_path):
            os.mkdir(bbox_path)
        with open(os.path.join(bbox_path, f"{frame}.json"), "w") as f:
                    json.dump(bbox, f)

os.system(f"rm -rf {tmpdir}")
