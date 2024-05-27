# !/bin/bash

PLY_DIR=$1
RENDER_TYPE=$2
PARAMS_PATH=$3
BLEND_FILE=$4

 If render type is canonical
if [ $RENDER_TYPE == "canonical" ]; then
  BLEND_FILE="./data/blend_files/canonical_pose.blend"
fi

~/data/users/cpokhari/blender-3.3/blender \
    $BLEND_FILE \
    -P scripts/process/bl_render_mano.py -b -- \
    $PLY_DIR $RENDER_TYPE
