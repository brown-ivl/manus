import math
import numpy as np


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros(4, 4)

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

    fovx = 2 * np.arctan(width / (2 * K[0, 0]))
    fovy = 2 * np.arctan(height / (2 * K[1, 1]))

    breakpoint()
    world_view_transform = torch.tensor(extrins).transpose(0, 1)
    self.projection_matrix = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
    ).transpose(0, 1)
    self.full_proj_transform = (
        self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    self.camera_center = self.world_view_transform.inverse()[3, :3]
