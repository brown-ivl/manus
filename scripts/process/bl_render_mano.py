import math
import sys
import joblib
from mathutils import Vector, Euler
from mathutils import Matrix
import json
import bpy
import os
import numpy as np
import glob
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

def get_camera_intrinsics_ingp(scene, camera):
    camera_angle_x = camera.data.angle_x
    camera_angle_y = camera.data.angle_y

    # camera properties
    f_in_mm = camera.data.lens  # focal length in mm
    scale = scene.render.resolution_percentage / 100
    width_res_in_px = scene.render.resolution_x * scale  # width
    height_res_in_px = scene.render.resolution_y * scale  # height
    optical_center_x = width_res_in_px / 2
    optical_center_y = height_res_in_px / 2

    # pixel aspect ratios
    size_x = scene.render.pixel_aspect_x * width_res_in_px
    size_y = scene.render.pixel_aspect_y * height_res_in_px
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    # sensor fit and sensor size (and camera angle swap in specific cases)
    if camera.data.sensor_fit == "AUTO":
        sensor_size_in_mm = (
            camera.data.sensor_height
            if width_res_in_px < height_res_in_px
            else camera.data.sensor_width
        )
        if width_res_in_px < height_res_in_px:
            sensor_fit = "VERTICAL"
            camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x
        elif width_res_in_px > height_res_in_px:
            sensor_fit = "HORIZONTAL"
        else:
            sensor_fit = "VERTICAL" if size_x <= size_y else "HORIZONTAL"

    else:
        sensor_fit = camera.data.sensor_fit
        if sensor_fit == "VERTICAL":
            sensor_size_in_mm = (
                camera.data.sensor_height
                if width_res_in_px <= height_res_in_px
                else camera.data.sensor_width
            )
            if width_res_in_px <= height_res_in_px:
                camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x

    # focal length for horizontal sensor fit
    if sensor_fit == "HORIZONTAL":
        sensor_size_in_mm = camera.data.sensor_width
        s_u = f_in_mm / sensor_size_in_mm * width_res_in_px
        s_v = f_in_mm / sensor_size_in_mm * width_res_in_px * pixel_aspect_ratio

    # focal length for vertical sensor fit
    if sensor_fit == "VERTICAL":
        s_u = f_in_mm / sensor_size_in_mm * width_res_in_px / pixel_aspect_ratio
        s_v = f_in_mm / sensor_size_in_mm * width_res_in_px

    camera_intr_dict = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": s_u,
        "fl_y": s_v,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": optical_center_x,
        "cy": optical_center_y,
        "w": width_res_in_px,
        "h": height_res_in_px,
    }
    return camera_intr_dict

def dump_camera_ingp(path, scene, camera):
    end = scene.frame_end
    all_data = []
    for frame in range(0, end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        data = get_camera_intrinsics_ingp(scene, camera)
        data["transform_matrix"] = np.array(camera.matrix_world)[:3, :4].tolist()
        all_data.append(data)
    scene.frame_set(0)

    with open(path, "w") as file:
        json.dump(all_data, file, indent=2)


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
    end = scene.frame_end
    intrs = []
    extrs = []

    for frame in range(0, end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        intr = get_camera_parameters_intrinsic(scene, camera)
        extr = get_camera_parameters_extrinsic()
        intrs.append(intr)
        extrs.append(extr)

    all_data = {}
    all_data['intrs'] = intrs
    all_data['extrs'] = extrs
    scene.frame_set(0)

    # joblib.dump(all_data, path)


def create_shaders(obj):
    mat = bpy.data.materials.new(name="material")
    obj.data.materials.append(mat)

    shader = bl.core.Shaders(mat.name)
    diffuse = shader.get_node('emission')
    mat_out = shader.get_node('Material Output')
    color_attr = shader.nodes.new('ShaderNodeVertexColor')

    shader.links.new(color_attr.outputs[0], diffuse.inputs[0])
    shader.links.new(diffuse.outputs[0], mat_out.inputs[0])


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    ply_dir = argv[0]
    render_type = argv[1]

    sc = bl.core.Scene()
    bl.utility.set_gpu([0], sc.scene)

    rendering_settings = {
        "engine": "CYCLES",
        "use_adaptive_sampling": True,
        "engine_type": "PATH",
        "sample_size": 16,
        "max_bounces": 8,
    }

    image_settings = {
        "resolution": [1080, 1080],
        "output_path": "",
        "film_transparent": True,
        "file_format": "PNG" #if render_type == "canonical" else "FFMPEG",
    }

    sc.initialize_rendering_settings(rendering_settings)
    sc.initialize_image_settings(image_settings)

    cam = bl.core.Camera("Camera")
    camera = cam.camera

    all_objs = glob.glob(ply_dir + "*.ply")
    dir_name = os.path.basename(os.path.dirname(ply_dir))
    out_dir = ply_dir.replace(dir_name, f'{dir_name}_rendered')
    os.makedirs(out_dir, exist_ok=True)

    if render_type == "gt_eval":
        all_objs = [all_objs[-1]]
        sc.scene.frame_start = 0
        sc.scene.frame_end = 16
        sc.scene.render.ffmpeg.format = 'MPEG4'
        sc.scene.render.fps = 30
        # out_path = os.path.join(out_dir, "gt_contacts.mp4")
        cam_dir = '/'.join(ply_dir.split('/')[:-2])
        # dump_camera_ours(os.path.join(cam_dir, "../gt_eval_cam.pkl"), sc.scene, camera)

        ## To match ingp object should be rotated by -90, -180, -90
        # dump_camera_ingp(os.path.join(cam_dir, "../ingp_gt_eval_cam.json"), sc.scene, camera)

        render_single_object(all_objs[0], sc, out_dir)
    elif render_type == "canonical":
        render_dynamic_objects(all_objs, sc, out_dir)

def render_single_object(ply_path, sc, out_path):
    obj = sc.add_objects(ply_path)

    ## To accomodate Instant NGP rendering
    # obj.rotate(math.radians(-90), 0)
    # obj.rotate(math.radians(-180), 1)
    # obj.rotate(math.radians(-90), 2)

    create_shaders(obj.obj)
    sc.select_object(obj.name)
    bpy.ops.object.shade_smooth()
    # bpy.ops.wm.save_as_mainfile(filepath='/users/cpokhari/work/NeuralGrasp/test.blend')
    # exit(0)
    sc.render(path=out_path, animation=True)

def dump_camera_ours(path, scene, camera):
    end = scene.frame_end
    intrs = []
    extrs = []

    for frame in range(0, end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        intr = get_camera_parameters_intrinsic(scene, camera)
        extr = get_camera_parameters_extrinsic()
        intrs.append(intr)
        extrs.append(extr)

    all_data = {}
    all_data['intrs'] = intrs
    all_data['extrs'] = extrs
    scene.frame_set(0)
    joblib.dump(all_data, path)


def render_dynamic_objects(all_objs, sc, out_dir):
    for idx, ply_path in enumerate(all_objs):
        sc.scene.frame_set(idx)
        bpy.context.view_layer.update()
        name = ply_path.split('/')[-1].split('.')[0]
        obj = sc.add_objects(ply_path)

        sc.select_object(obj.name)
        bpy.ops.object.shade_flat()

        create_shaders(obj.obj)

        out_path = os.path.join(out_dir, f'{name}.png')
        sc.render(path=out_path, animation=False)
        sc.delete_objects(obj.name)
        del obj

if __name__ == '__main__':
    main()
