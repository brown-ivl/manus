## Pose Estimation 

- Install AlphaPose first using instructions `AlphaPoseREADME.md`
- Download the pretrained model  `multi_domain_fast50_dcn_combined_256x192.pth` from the model zoo of the AlphaPose. 
- See `pose.py` for all the steps to estimate the final joint angles
- Make changes to `pose.py` and run `pose.py` for complete optimization of the pose (from 2D keypoints to the IK)