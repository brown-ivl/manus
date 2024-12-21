import sys
import os

def get_2d_keypoints(root_dir, action, params_path, output_dir, which): 
    ##### Get 2D keypoints ######
    #############################
    cmd = f'python scripts/get_2d_keypoints.py -r {root_dir} --seq_path "{which}/{action}" --undistort --start 0 --end -1 --stride 2 --params_path {params_path} --out_dir {output_dir}'
    os.system(cmd)

def get_3d_keypoints(action, output_dir, params_path, which): 
    ##### Get 3D keypoints ######
    #############################
    cmd = f'python scripts/keypoints_3d.py --root_dir {output_dir}/{which} --seq_path {action} --stride 1 --params_path {params_path} --out_dir {output_dir}/{which} --conf_thresh 0.02'
    os.system(cmd)
    cmd = f'python scripts/filter_poses.py -r {output_dir} --seq_path {which}/{action} --bin_size 1'
    os.system(cmd)


def get_bone_length(output_dir, rest_poses_path, which):
    cmd = f'python scripts/get_bone_length.py --root_dir {output_dir} --use_filtered --input_dir keypoints_3d --seq_dir {which} --rest_pose_path {rest_poses_path}'
    os.system(cmd)


def get_poses(output_dir, action, rest_poses_path, which): 
    # Run IK on unfiltered keypoints
    cmd = f'python scripts/get_joint_angles.py --input_dir keypoints_3d --root_dir {output_dir} --seq_path {which}/{action} --use_prev --gpu --max_cpu 8 --use_filtered --rest_pose_path {rest_poses_path} --stride 1 --use_common_bone_len' 
    os.system(cmd)

def one_euro(output_dir, action, rest_poses_path, which): 
    cmd = f'python scripts/one_euro.py --root_dir {output_dir} --seq_path {which}/{action} --mode joint_angles --rest_pose_path {rest_poses_path}' 
    os.system(cmd)

def visualize_ik(raw_dir, output_dir, raw_action, rest_poses_path, params_path, which): 
    action = raw_action.split('_session_')[-1]
    raw_video = os.path.join(raw_dir, which, raw_action, 'synced')
    cmd = f'python scripts/visualize_IK.py --raw_video {raw_video} --root_dir {output_dir} --seq_path {which}/{action} --undistort --params_path {params_path} --rest_pose_path {rest_poses_path} --joint_angles joint_angles_filtered --keypoints_3d keypoints_3d'
    os.system(cmd)

def export_poses(output_dir, which, action, rest_poses_path): 
    rest_blend_path = rest_poses_path.replace('json', 'blend')
    cmd = f'../blender/blender -b {rest_blend_path} -P scripts/export_poses.py -- -r {output_dir} -s {which}/{action} -f 20 --one_euro'
    os.system(cmd)


def main(): 
    root_dir = "/users/cpokhari/data/datasets/BRICS/BRICS-DATA-02/neural-hands/chandradeep/"
    params_path = "/users/cpokhari/data/datasets/MANUS_data/chandradeep/calib.object/optim_params.txt"
    output_dir = "./pose_outputs/"
    raw_action = "2023-10-26_session_bag1_grasp1"
    which = "grasps"

    get_2d_keypoints(root_dir, raw_action, params_path, output_dir, which)

    action = raw_action.split('_session_')[-1]
    get_3d_keypoints(action, output_dir, params_path, which)

    rest_poses_path=os.path.join(os.getcwd(), "rest_poses/subject0.json")
    get_bone_length(output_dir, rest_poses_path, which)
    get_poses(output_dir, action, rest_poses_path, which)
    one_euro(output_dir, action, rest_poses_path, which)
    visualize_ik(root_dir, output_dir, raw_action, rest_poses_path, params_path, which)
    export_poses(output_dir, which, action, rest_poses_path)




if __name__ == '__main__': 
    main()