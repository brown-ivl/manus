# !/bin/bash

~/data/cpokhari/blender-3.3/blender \
    ./data/blend_files/pose1.blend \
    -P scripts/process/export_novel_pose.py  -b
