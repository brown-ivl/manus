#!/bin/bash

SUBJECT_NAME=$1
EXP_NAME=$2
EXP_DIR=$3

BLENDER_PATH=~/data/users/cpokhari/blender-3.3/blender

## Get novel cam path in the Blender
PC_PLY_PATH="${EXP_DIR}/init_gaussians.ply"
OUT_PATH="${EXP_DIR}/results/novel_cam.pkl"

## Get novel cam path in the Blender
$BLENDER_PATH ./data/blend_files/static.blend \
    -P scripts/process/bl_render.py -b -- $PC_PLY_PATH $OUT_PATH 1.2 0.01 -1.578 0 0

# Inference on train/test Set
python main.py --config-name config.yaml \
    --config-path $EXP_DIR \
    hydra.run.dir=$EXP_DIR \
    trainer.mode='test' \
    trainer.project='object' \
    'checkpoint="best"' \
    test_dataset.opts.resize_factor=1.0 \
    test_dataset.opts.frame_sample_rate=1 \
    test_dataset.opts.color_bkgd_aug="white" \
    test_dataset.opts.test_on_train_dataset=false \
    test_dataset.opts.cam_path=$OUT_PATH
