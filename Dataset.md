# MANUS-Grasps Dataset

Full dataset can be used using these commands. `raw_videos` are the original captured videos from the [BRICS](https://ivl.cs.brown.edu/research/diva.html) capture system. `annotationsV0` are the annotations for the raw videos. Further annotation version would include improved annotations. 
Dataset info is coming soon!!

```
aws s3 cp s3://manus-data/raw_videos/ <path_to_destination> --recursive --no-sign-request
aws s3 cp s3://manus-data/annotationsV0/ <path_to_destination> --recursive --no-sign-request
```

## Dataset Info
We follow the dataset directory structure like this, 

```
├── {SUBJECT}
    ├── actions_hdf5/ 
    ├── bone_keypoints.ply
    ├── bone_lens.json
    ├── calib.actions/ 
    ├── calib.evals/ 
    ├── calib.grasps/ 
    ├── calib.object/ 
    ├── evals/ 
    ├── grasps/ 
    ├── mano_rest.pkl
    ├── mano_rest.ply
    ├── mano_shape.npz
    ├── objects/ 
    └── rest_keypts.ply
```

- To optimize object module, we just requires multi-view images and camera parameters. Camera parameters can be found in `calib.object` folder. `optim_params.txt` file contains the camera parameters. And images can be found inside `object` folder. 
- To optimize hand module, we use multi-view sequences of different hand poses. It can be found in `actions_hdf5` folder. 