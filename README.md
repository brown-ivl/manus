# MANUS: Markerless Grasp Capture using Articulated 3D Gaussians [CVPR 2024]

MANUS is built on top of original Gaussian Splatting codebase and reuses functions from it's codebase heavily. 
Clone the repository using `git clone --recursive <link>`. 

## Setting codebase
We can use conda to setup the python environment like this
```
bash setup_env.sh
```

Apart from the conda env, we use `Blender` to get novel views during test time. You can download the Blender(3.3) and provide the path to the bash file. 

## Codebase Info
- `config` folder contains the config files for trainer, dataset, and different modules. These config parameters can be overridden in the `bash script`. To maintain the config, we use `hyra-core`.
- `src` contains the main code, and `data` contains the essential data required. 
- `main.py` file contains driver code from which everything kickstarts. 
- `submodules` folder contains the differentiable rasterizer and knn provided by original Gaussian-Splatting repo. 

## MANUS-Grasps Dataset
[To be released soon]

## Dataset Info
We follow the dataset directory structure like this, 

```
├── {SUBJECT}
    ├── actions/
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

## Optimization

To optimize object module on our dataset. 
```
bash scripts/train/train_object.sh {SUBJECT}
```

To optimize hand module on our dataset. 
```
bash scripts/train/train_hands_ours.sh {SUBJECT} {EXP_NAME}
```

To composite the scene either for the grasp results of for the evaluation. 
```
bash scripts/composite.sh {SUBJECT} {HAND_EXP_NAME} {results/eval}
```

## Citation
```
@misc{pokhariya2023manus,
      title={MANUS: Markerless Hand-Object Grasp Capture using Articulated 3D Gaussians}, 
      author={Chandradeep Pokhariya and Ishaan N Shah and Angela Xing and Zekun Li and Kefan Chen and Avinash Sharma and Srinath Sridhar},
      year={2023},
      eprint={2312.02137},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
```
This work was supported by NSF CAREER grant #2143576, ONR DURIP grant N00014-23-1-2804, ONR grant N00014-22-1-259, a gift from Meta Reality Labs, and an AWS Cloud Credits award. We would like to thank George Konidaris, Stefanie Tellex and Dingxi Zhang. Additionally, we thank Bank of Baroda for partially funding Chandradeep’s travel expenses.
```

