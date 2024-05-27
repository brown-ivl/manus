import numpy as np
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib
from matplotlib.cm import ScalarMappable
from typing import Tuple


def plot_points_in_image(points, image, color=(0, 255, 0), radius=2, thickness=-1):
    image = image.copy()

    for point in points:
        image = cv2.circle(
            image, tuple(point[:2].astype(np.int32)), radius, color, thickness
        )

    return image


def get_colors_from_cmap(scalar_values, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(scalar_values)[..., :3]
    return colors


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
    proj_mat: np.ndarray,
    kintree: dict,
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
        ) in kintree.items():
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
