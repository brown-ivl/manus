import math
import sys
import joblib
from mathutils import Vector, Euler
from mathutils import Matrix
import json
import bpy
import os
import numpy as np
import math
import blender_wormholes as bl


def get_camera_parameters_extrinsic():
    bcam = bpy.context.scene.camera
    R_bcam2cv = np.array([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]])
    location = np.array([bcam.matrix_world.decompose()[0]]).T
    R_world2bcam = np.array(bcam.matrix_world.decompose()[
                                1].to_matrix().transposed())
    T_world2bcam = np.matmul(R_world2bcam.dot(-1), location)
    R_world2cv = np.matmul(R_bcam2cv, R_world2bcam)
    T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)
    extr = np.concatenate((R_world2cv, T_world2cv), axis=1)
    return extr


def get_scene_resolution(scene):
    resolution_scale = (scene.render.resolution_percentage / 100.0)
    resolution_x = scene.render.resolution_x * resolution_scale  # [pixels]
    resolution_y = scene.render.resolution_y * resolution_scale  # [pixels]
    return int(resolution_x), int(resolution_y)


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def get_camera_parameters_intrinsic(scene, camera):
    """ Get intrinsic camera parameters: focal length and principal point. """
    # ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera/120063#120063
    focal_length = scene.camera.data.lens  # [mm]
    res_x, res_y = get_scene_resolution(scene)
    cam_data = camera.data
    sensor_size_in_mm = get_sensor_size(cam_data.sensor_fit, cam_data.sensor_width, cam_data.sensor_height)
    sensor_fit = get_sensor_fit(
        cam_data.sensor_fit, scene.render.pixel_aspect_x * res_x, scene.render.pixel_aspect_y * res_y)
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = res_x
    else:
        view_fac_in_px = pixel_aspect_ratio * res_y
    pixel_size_mm_per_px = (
                                   sensor_size_in_mm / focal_length) / view_fac_in_px
    f_x = 1.0 / pixel_size_mm_per_px
    f_y = (1.0 / pixel_size_mm_per_px) / pixel_aspect_ratio
    c_x = (res_x - 1) / 2.0 - cam_data.shift_x * view_fac_in_px
    c_y = (res_y - 1) / 2.0 + (cam_data.shift_y *
                               view_fac_in_px) / pixel_aspect_ratio
    return f_x, f_y, c_x, c_y


def animate_scene(path, scene, camera):
    intrs = []
    extrs = []

    intr = get_camera_parameters_intrinsic(scene, camera)
    extr = get_camera_parameters_extrinsic()
    intrs.append(intr)
    extrs.append(extr)

    all_data = {}
    all_data['intrs'] = intrs
    all_data['extrs'] = extrs
    joblib.dump(all_data, path)


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    ply_path = argv[0]
    cam_traj_path = argv[1]

    sc = bl.core.Scene()
    # bl.utility.set_gpu([gpu], sc.scene)

    rendering_settings = {
        "engine": "CYCLES",
        "use_adaptive_sampling": True,
        "engine_type": "PATH",
        "sample_size": 32,
        "max_bounces": 8,
    }

    image_settings = {
        "resolution": [720, 720],
        "output_path": "",
        "film_transparent": True,
        "file_format": "FFMPEG",
    }

    sc.initialize_rendering_settings(rendering_settings)
    sc.initialize_image_settings(image_settings)

    obj = sc.add_objects(ply_path)
    sc.select_object(obj.name)
    obj.set_origin_to("geometry")
    cam = bl.core.Camera("Camera")
    camera = cam.camera

    animate_scene(cam_traj_path, sc.scene, camera)


if __name__ == '__main__':
    main()
