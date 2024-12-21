import numpy as np
import cv2
from src.IK.skeleton import KinematicChain
from typing import Tuple

def project(keypoints3d: np.ndarray, P: np.ndarray):
    """
    Project keypoints to 2D using

    Inputs -
        keypoints3d (N, 3): 3D keypoints
        P (V,3,4): Projection matrices
    Outputs -
        keypoints2d (V, N, 2): Projected 2D keypoints
    """
    hom = np.hstack((keypoints3d, np.ones((keypoints3d.shape[0], 1))))
    projected = np.matmul(P, hom.T).transpose(0, 2, 1)  # (V, N, 2)
    projected = (projected / projected[:, :, -1:])[:, :, :-1]
    return projected

def plot_keypoints_2d(
    joints: np.ndarray,
    image: np.ndarray,
    chain: KinematicChain,
    proj_mat: np.ndarray,
    bone_color: Tuple[int] = (255, 0, 0),
    plot_bones: bool = True,
) -> np.ndarray:
    keypoints_2d = project(joints, np.asarray([proj_mat]))[0]
    res = image.copy()
    joint_radius = min(*image.shape[:2]) // 150
    for keypoint in keypoints_2d:
        cv2.circle(
            res, (int(keypoint[0]), int(keypoint[1])), joint_radius, (0, 0, 255), -1
        )

    if plot_bones:
        for (
            bone,
            parent,
        ) in chain.kintree.items():
            parent_id = parent + 1
            bone_id = int(bone) + 1
            cv2.line(
                res,
                (int(keypoints_2d[bone_id][0]), int(keypoints_2d[bone_id][1])),
                (int(keypoints_2d[parent_id][0]), int(keypoints_2d[parent_id][1])),
                bone_color,
                joint_radius // 2,
            )

    return res