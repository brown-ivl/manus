import math
import numpy as np
import os
import sys

sys.path.insert(0, os.getcwd())
from src.utils import params as param_utils


def get_scene_extent(cam_centers):
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    radius = diagonal * 1.1
    return radius


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def get_opengl_camera_attributes(
    K, extrins, width, height, zfar=100.0, znear=0.01, resize_factor=1.0
):
    K[..., :2, :] = K[..., :2, :] * resize_factor
    width = int(width * resize_factor + 0.5)
    height = int(height * resize_factor + 0.5)
    fovx = focal2fov(K[0, 0], width)
    fovy = focal2fov(K[1, 1], height)
    extrins = np.concatenate([extrins, np.array([[0, 0, 0, 1]])], axis=0)
    world_view_transform = np.transpose(extrins)
    projection_matrix = np.transpose(
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
    )
    full_proj_transform = np.matmul(world_view_transform, projection_matrix)
    camera_center = np.linalg.inv(world_view_transform)[3, :3]

    out = {
        "width": width,
        "height": height,
        "fovx": fovx,
        "fovy": fovy,
        "K": K,
        "extr": extrins,
        "world_view_transform": world_view_transform,
        "projection_matrix": projection_matrix,
        "full_proj_transform": full_proj_transform,
        "camera_center": camera_center,
    }
    return out


def load_brics_cameras(dir_path, to_skip, calib_dir):
    cam_filepath = os.path.join(dir_path, calib_dir, "optim_params.txt")
    cam_data = param_utils.read_params(cam_filepath)

    P = {}
    for idx, param in enumerate(cam_data):
        cam_name = cam_data["cam_name"][idx]
        if cam_name in to_skip:
            continue
        K, dist = param_utils.get_intr(param)
        new_K, _ = param_utils.get_undistort_params(
            K, dist, (param["width"], param["height"])
        )
        extr = param_utils.get_extr(param)

        P[cam_name] = {
            "K": K,
            "new_K": new_K,
            "dist": dist,
            "extrinsics_opencv": extr,
        }
    return P
