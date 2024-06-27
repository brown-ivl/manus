#!/bin/bash

SUBJECT_NAME=$1
EXP_NAME=$2
HAND_EXP_DIR=$3
OBJECT_EXP_DIR=$4
OBJECT_EXP_NAME=$5
GRASP_PATH=$6
EXP_DIR=$7
CONTACT_RENDER_TYPE=$8
DATA_DIR=$9


if [[ $CONTACT_RENDER_TYPE == "acc_gt_eval" ]]; then
  GRASP_PATH="${DATA_DIR}/${SUBJECT_NAME}/evals/${OBJECT_EXP_NAME}_action/meta_data.pkl"
#  CAM_PATH="/users/cpokhari/data/users/cpokhari/grasp_data/${SUBJECT_NAME}/calib.evals/optim_params.txt"
  CAM_PATH="${DATA_DIR}/${SUBJECT_NAME}/evals/${OBJECT_EXP_NAME}_action/gt_cam.pkl"
  BKGD_COLOR="black"

else
  PC_PLY_PATH="${OBJECT_EXP_DIR}/init_gaussians.ply"
  CAM_PATH="${EXP_DIR}/results/novel_cam.pkl"
  BKGD_COLOR="white"

  ~/data/users/cpokhari/blender-3.3/blender \
      ./data/blend_files/static.blend \
      -P scripts/process/bl_render.py -b -- $PC_PLY_PATH $CAM_PATH 1.0 0.004 -1.578 0 0

fi

echo "CONTACT_RENDER_TYPE: $CONTACT_RENDER_TYPE"
echo "CAM_PATH: $CAM_PATH"
echo "GRASP_PATH: $GRASP_PATH"

python main.py --config-name COMPOSITE \
    trainer.mode='test' \
    trainer.project='composite' \
    trainer.exp_name=$EXP_NAME  \
    trainer.num_workers=4 \
    output_dir=$EXP_DIR \
    train_dataset.opts.subject=$SUBJECT_NAME \
    opts.object_ckpt_dir="${OBJECT_EXP_DIR}/checkpoints" \
    opts.hand_ckpt_dir="${HAND_EXP_DIR}/checkpoints" \
    test_dataset.opts.frame_sample_rate=2 \
    test_dataset.opts.test_on_canonical_pose=false \
    test_dataset.opts.cano_cam_path='./data/camera_paths/cano_camera.pkl' \
    test_dataset.opts.subject=$SUBJECT_NAME \
    hand_model.opts.skin_weights_init_type='mano_init_voxel' \
    hand_model.opts.sh_degree=3 \
    test_dataset.opts.color_bkgd_aug=$BKGD_COLOR \
    test_dataset.opts.cam_path=$CAM_PATH \
    test_dataset.opts.metadata_path=$GRASP_PATH \
    test_dataset.opts.contact_render_type=$CONTACT_RENDER_TYPE