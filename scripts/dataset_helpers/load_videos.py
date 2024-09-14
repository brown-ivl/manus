import os
import sys
from natsort import natsorted
import numpy as np
import glob
sys.path.append(os.getcwd())
from src.utils.reader import Reader
from src.utils import params as param_utils
from src.utils.extra import *

## Make sure you have install manopth
from manopth.manolayer import ManoLayer

from dataclasses import dataclass
from collections import defaultdict
from src.utils.vis_util import plot_points_in_image, project



@dataclass
class Cameras:
    cam_name: np.ndarray
    K: np.ndarray
    distK: np.ndarray
    dist: np.ndarray
    extr: np.ndarray
    
    def __getitem__(self, key):
        if isinstance(key, str):
            idx = np.where(self.cam_name == key)[0][0]
        elif isinstance(key, int):
            idx = key
        else:
            raise TypeError("Key must be either an integer or a string representing the camera name.")

        new_dict = {}
        for key, value in self.__dict__.items():
            new_dict[key] = value[idx]
        return Cameras(**new_dict)

@dataclass
class MANO: 
    pose: np.ndarray
    shape: np.ndarray
    trans: np.ndarray
    scale: np.ndarray
    fno: np.ndarray

    def __getitem__(self, fno):
        idx = np.where(self.fno == fno)[0][0]
        new_dict = {}
        for key, value in self.__dict__.items():
            if value is not None:
                new_dict[key] = value[idx]
            else:
                new_dict[key] = None
        return MANO(**new_dict)

def load_mano_poses(mano_poses): 
    m_dict = defaultdict(list)
    for pose_path in mano_poses: 
        data = np.load(pose_path)
        fno = int(pose_path.split('/')[-1].split('.')[0])
        angle = data['angle']
        shape = data['shape']
        trans = data['trans']
        scale = data['scale']
        m_dict['pose'].append(data['angle'])
        m_dict['shape'].append(data['shape'])
        m_dict['trans'].append(data['trans'])
        m_dict['scale'].append(data['scale'])
        m_dict['fno'].append(fno)

    for k, v in m_dict.items():
        m_dict[k] = np.stack(v, axis=0)
    return MANO(**m_dict)

def load_all_cameras(cam_path, width = 1280, height = 720): 
    cameras = param_utils.read_params(cam_path)
    d_dict = defaultdict(list)
    for idx, cam in enumerate(cameras):
        extr = param_utils.get_extr(cam)
        K, dist = param_utils.get_intr(cam)
        cam_name = cam["cam_name"]
        new_K, roi = param_utils.get_undistort_params(K, dist, (width, height))
        new_K = new_K.astype(np.float32)
        extr = extr.astype(np.float32)
        d_dict['cam_name'].append(cam_name)
        d_dict['dist'].append(dist)
        d_dict['distK'].append(K)
        d_dict['extr'].append(extr)
        d_dict['K'].append(new_K)

    for k, v in d_dict.items():
        d_dict[k] = np.stack(v, axis=0)

    return Cameras(**d_dict)

def main(): 
    ### To get raw videos

    raw_video_dir= "/users/cpokhari/data/datasets/BRICS/BRICS-DATA-02/neural-hands/chandradeep/grasps/2023-10-26_session_bag1_grasp1/synced"
    grasp_path = '/users/cpokhari/data/datasets/MANUS_data/chandradeep/grasps/bags1_grasp1/meta_data.pkl'

    cam_dir = '/'.join(grasp_path.split('/')[:-3])
    grasp_dir = os.path.join(cam_dir, 'grasps')
    grasp_seq = grasp_path.split('/')[-2]
    cam_path = "/users/cpokhari/data/datasets/MANUS_data/chandradeep/calib.object/optim_params.txt"

    ## Take these cams from `optim_params.txt` as while optimizing for camera paramters, we have to through 
    ## some camera views. 
    start_frame_idx = 500 
    ## For most of the scenes, MANO is fitted from 200th frame 
    ## as pose estimation doesn't work well when hand is not visible in all frames
    ## Even from first 200th frame, MANO fitting might not be accurate because of faulty keypoint detection
    ## but that doesn't affect contact quality, because hand is still far from the object. 
    ## When hands are closer to the object, we tried fitting MANO as accurately as possible. 

    idx = 0
    step_size = 2 ## We fit MANO for every second frame
    fno = idx * step_size + start_frame_idx

    cameras = param_utils.read_params(cam_path)
    chosen_cams = cameras[:]['cam_name'].tolist()
    reader = Reader("video", raw_video_dir, selected_cams = chosen_cams, undistort=True, cam_path=cam_path)
    frames, cur_frame = next(reader([fno]))

    ### To get MANO parameters

    ## Use MANO_RIGHT.pkl file for MANO. We support only right hand as of now. 
    mano_layer = ManoLayer(mano_root='data/mano/models', use_pca=True, flat_hand_mean=True, ncomps = 30)
    mano_files = natsorted(glob.glob(os.path.join(grasp_dir, grasp_seq, 'mano/mano_params/*.npz')))
    mano_poses = load_mano_poses(mano_files)
    cameras = load_all_cameras(cam_path)
    mano = to_tensor(mano_poses[fno])

    output_dir = "./mano_vis/"
    os.makedirs(output_dir, exist_ok = True)

    for cam_name in list(frames.keys()): 
        out_path = os.path.join(output_dir, f'{cam_name}.png')
        verts, joints, _, _, _ = mano_layer(mano.pose.unsqueeze(0), mano.shape)
        verts = (verts/1000) * mano.scale + mano.trans   
        joints = (joints/1000) * mano.scale + mano.trans

        f = frames[cam_name]
        cam = cameras[cam_name]
        K = cam.K
        RT = cam.extr
        P = K @ RT[:3, :4]
        pts = project(to_numpy(verts[0]), P[None])[0]
        mano_plot = plot_points_in_image(pts, f[..., :3][..., ::-1]).astype(np.uint8)
        dump_image(mano_plot , out_path)

if __name__ == '__main__': 
    main()