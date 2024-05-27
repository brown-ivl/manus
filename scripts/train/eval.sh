#!/bin/bash

SUBJECT_NAME=$1
EXP_NAME=$2
HAND_EXP_DIR=$3
OBJECT_EXP_DIR=$4
OBJECT_EXP_NAME=$5
GRASP_PATH=$6
EXP_DIR=$7

echo "Contacts calculation for MANO!! "
python scripts/process/mano_contacts.py \
   --exp_dir $EXP_DIR \
   --exp_name $EXP_NAME \
   --subject_name $SUBJECT_NAME \
   --object_exp_dir $OBJECT_EXP_DIR \
   --object_exp_name $OBJECT_EXP_NAME \
   --grasp_path $GRASP_PATH

echo "Contacts calculation for HARP!! "
python scripts/process/mano_contacts.py \
   --exp_dir $EXP_DIR \
   --exp_name $EXP_NAME \
   --subject_name $SUBJECT_NAME \
   --object_exp_dir $OBJECT_EXP_DIR \
   --object_exp_name $OBJECT_EXP_NAME \
   --grasp_path $GRASP_PATH --harp

bash scripts/train/run_composite.sh $SUBJECT_NAME $EXP_NAME $HAND_EXP_DIR $OBJECT_EXP_DIR $OBJECT_EXP_NAME $GRASP_PATH $EXP_DIR "gt_eval"
bash scripts/train/run_composite.sh $SUBJECT_NAME $EXP_NAME $HAND_EXP_DIR $OBJECT_EXP_DIR $OBJECT_EXP_NAME $GRASP_PATH $EXP_DIR "acc_gt_eval"
python scripts/process/get_iou_ours.py --exp_dir $EXP_DIR --grasp_path $GRASP_PATH --object_exp_name $OBJECT_EXP_NAME
python scripts/process/get_iou.py --exp_dir $EXP_DIR --grasp_path $GRASP_PATH --object_exp_name $OBJECT_EXP_NAME

#bash scripts/train/run_composite.sh $SUBJECT_NAME $EXP_NAME $HAND_EXP_DIR $OBJECT_EXP_DIR $OBJECT_EXP_NAME $GRASP_PATH $EXP_DIR "results"
#mkdir "${EXP_DIR}results/eval_results/ours/final"
#ffmpeg -i "${EXP_DIR}results/eval_results/ours/results.mp4" -vf fps=10 "${EXP_DIR}results/eval_results/ours/final/final_%04d.png"
# bash scripts/train/run_composite.sh $SUBJECT_NAME $EXP_NAME $HAND_EXP_DIR $OBJECT_EXP_DIR $OBJECT_EXP_NAME $GRASP_PATH $EXP_DIR "nocs"
