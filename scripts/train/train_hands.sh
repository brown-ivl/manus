#!/bin/bash

SUBJECT=$1
EXP_NAME=$2

DATA_DIR="MANUS_data"
EXP_DIR="outputs/hand/${SUBJECT}/${EXP_NAME}"
ROOT_DIR="${DATA_DIR}/${SUBJECT}/actions_hdf5"

WIDTH=1280
HEIGHT=720

python main.py --config-name HAND_GAUSSIAN \
    output_dir=$EXP_DIR \
    trainer.loggers='[csv]' \
    trainer.mode='train' \
    trainer.project='hand' \
    trainer.exp_name=$EXP_NAME \
    trainer.gpus=[0] \
    trainer.num_workers=4 \
    trainer.accum_iter=1 \
    trainer.pl_vars.max_steps=15000 \
    +trainer.pl_vars.check_val_every_n_epoch=1 \
    train_dataset.opts.num_time_steps=1 \
    train_dataset.opts.subject=$SUBJECT \
    train_dataset.opts.width=$WIDTH \
    train_dataset.opts.height=$HEIGHT \
    train_dataset.opts.root_dir=$ROOT_DIR \
    train_dataset.opts.sequences='all' \
    train_dataset.opts.split_ratio=0.75 \
    model.opts.sample_size=10000 \
    model.opts.sh_degree=3 \
    model.opts.percent_dense=1e-2 \
    model.opts.skin_weights_init_type='mano_init_voxel' \
    model.opts.extra_params_opt_iter_start=1000 \
    model.opts.extra_params_opt_iter_freq=1000 \
    model.opts.densify_grad_threshold=0.00003 \
    model.opts.isotropic_scaling=false \
    model.opts.remove_seg_end=1 \
    model.opts.start_lpips_iter=1000 \
    model.opts.grid_size='[1.0, 0.9, 0.6]' \
    model.opts.grid_offset='[0.01, 0.0, -0.008]' \
    model.opts.grid_res=128 

#bash scripts/train/inference_hand.sh $SUBJECT $EXP_NAME

