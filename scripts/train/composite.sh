#!/bin/bash

SUBJECT_NAME=$1
HAND_EXP_NAME=$2
MODE=$3 ##results/eval

EXP_DIR="outputs"
DATA_DIR="MANUS_data"
ROOT_DIR="${DATA_DIR}/${SUBJECT}"

## Define the objects for which we want to do grasp capture. 
## Note that if you are using "eval" mode, then objs should be `green colored evaluation objects`
OBJS=("books1")

for OBJ_NAME in "${OBJS[@]}"
do
  OBJ_DIR="${ROOT_DIR}/objects/${OBJ_NAME}"
  OBJECT_EXP_NAME="${OBJ_NAME}"
  OBJECT_EXP_DIR="${EXP_DIR}/object/${SUBJECT_NAME}/${OBJECT_EXP_NAME}/"

  HAND_EXP_DIR="${EXP_DIR}/hand/${SUBJECT_NAME}/${HAND_EXP_NAME}"

  ## Note that here 'grasp1' can be 'grasp2' and so on.. if dataset contains it. 
  GRASP_PATH="${DATA_DIR}/${SUBJECT_NAME}/grasps/${OBJECT_EXP_NAME}_grasp1/meta_data.pkl"

  EXP_NAME=$OBJECT_EXP_NAME"--"$HAND_EXP_NAME
  EXP_DIR="${EXP_DIR}/composite/${SUBJECT_NAME}/${EXP_NAME}/"

  if [ ! -d "$EXP_DIR" ]; then
      mkdir -p "${EXP_DIR}/results"
  fi

  if [ "$MODE" == "eval" ]; then
    if [[ "$OBJ_NAME" == *"color"* ]]; then
      bash scripts/train/eval.sh $SUBJECT_NAME $EXP_NAME $HAND_EXP_DIR $OBJECT_EXP_DIR $OBJECT_EXP_NAME $GRASP_PATH $EXP_DIR
    else
      echo "Evaluation can't be performed on this object. Please use the color objects!!"
      exit
    fi
  elif [ "$MODE" == "results" ]; then
    bash scripts/train/run_composite.sh $SUBJECT_NAME $EXP_NAME $HAND_EXP_DIR $OBJECT_EXP_DIR $OBJECT_EXP_NAME $GRASP_PATH $EXP_DIR "results" $DATA_DIR
  fi
done