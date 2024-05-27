# !/bin/bash

BLEND
PLY_DIR=$1

~/data/users/cpokhari/blender-3.3/blender \
    ./data/blend_files/canonical_pose.blend \
    -P scripts/process/bl_render_mano.py -b -- \
    $PLY_DIR
