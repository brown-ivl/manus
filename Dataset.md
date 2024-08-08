# MANUS-Grasps Dataset

More detailed information coming!!

Full dataset can be used using these commands. `raw_videos` are the original captured videos from the [BRICS](https://ivl.cs.brown.edu/research/diva.html) capture system. `annotationsV1` are the 1st version of annotations for the raw videos. Further annotation version would include improved annotations. 

As of now, we are hosting dataset in the AWS. To downlaod the dataset, please use following commands. 

```
aws s3 cp s3://manus-data/raw_videos/ <path_to_destination> --recursive --no-sign-request
aws s3 cp s3://manus-data/annotationsV0/ <path_to_destination> --recursive --no-sign-request
```

## Dataset Info

#### Raw Videos
There are raw videos for 4 subjects. We showed results for first three subjects, however we release the 4th subject complimentary. Each subject contains the grasp videos from multi-views. 

#### Annotations

```
├── {SUBJECT}
    ├── actions_hdf5/ 
    ├── grasps/ 
    ├── evals/ 
    ├── objects/ 
    ├── bone_keypoints.ply
    ├── bone_lens.json
    ├── calib.actions/ 
    ├── calib.evals/ 
    ├── calib.grasps/ 
    ├── calib.object/ 
    ├── mano_rest.pkl
    ├── mano_rest.ply
    ├── mano_shape.npz
    └── rest_keypts.ply
```

- To optimize object module, we just requires multi-view images and camera parameters. Camera parameters can be found in `calib.object` folder. `optim_params.txt` file contains the camera parameters. And images can be found inside `object` folder. 
- To optimize hand module, we use multi-view sequences of different hand poses. It can be found in `actions_hdf5` folder. 
