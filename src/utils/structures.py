import numpy as np
from dataclasses import dataclass
import torch
from src.utils.graphics_utils import getProjectionMatrix


@dataclass
class Bones:
    bnames: np.ndarray
    heads: np.ndarray
    tails: np.ndarray
    transforms: np.ndarray
    eulers: np.ndarray = None
    eulers_c: np.ndarray = None
    root_translation: np.ndarray = None
    root_rotation: np.ndarray = None
    kintree: dict = None

    def __getitem__(self, idx):
        new_dict = {}
        for key, value in self.__dict__.items():
            if value is not None:
                new_dict[key] = value[idx]
            else:
                new_dict[key] = None
        return Bones(**new_dict)


@dataclass
class Cameras:
    cam_name: np.ndarray
    K: np.ndarray
    extr: np.ndarray
    fovx: float
    fovy: float
    width: int
    height: int
    world_view_transform: np.ndarray
    projection_matrix: np.ndarray
    full_proj_transform: np.ndarray
    camera_center: np.ndarray

    def __getitem__(self, idx):
        new_dict = {}
        for key, value in self.__dict__.items():
            new_dict[key] = value[idx]
        return Cameras(**new_dict)
