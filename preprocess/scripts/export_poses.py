import bpy
import json
import numpy as np
import glob
import os
import argparse
import pickle
import bmesh  # type: ignore
import sys
# import trimesh # type: ignore
import logging
from mathutils import Vector, Euler # type: ignore
from natsort import natsorted

"""
Majority of the functions taken from https://github.com/facebookresearch/tava
"""

logging.basicConfig(level=logging.INFO)

# Begin animation.py


def get_object_by_type(type: str, index: int = 0):
    """Get an general object in the scene."""
    bpy.ops.object.select_by_type(type=type)
    obj = bpy.context.selected_objects[index]
    return obj


def get_mesh(index: int = 0):
    """Get a mesh in the scene."""
    return get_object_by_type(type="MESH", index=index)


def get_armature(index: int = 0):
    """Get a armature in the scene."""
    return get_object_by_type(type="ARMATURE", index=index)


def triangulate_mesh(mesh_obj):
    """[Inplace] Convert the polygon (maybe tetragon) faces into triangles."""
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode="EDIT")
    mesh = mesh_obj.data
    bm = bmesh.from_edit_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.update_edit_mesh(mesh, True)
    bpy.ops.object.mode_set(mode="OBJECT")


def extract_bone_data_grasp(armature_obj, bone_name: str, rest_space: bool = False):

    matrix_world = armature_obj.matrix_world
    if rest_space:
        bone = armature_obj.data.bones[bone_name]
        matrix = bone.matrix_local
        tail = bone.tail_local
        head = bone.head_local
        return matrix, tail, head, matrix_world
    else:
        bone = armature_obj.pose.bones[bone_name]
        matrix = bone.matrix
        tail = bone.tail
        head = bone.head
        param = armature_obj.pose.bones[bone_name].matrix_basis
        euler = bone.rotation_euler
        return matrix, tail, head, matrix_world, param, euler


def extract_bone_data(armature_obj, bone_name: str, rest_space: bool = False):
    """Extract the rest (canonical) or pose state of a bone.

    :params rest_space: if set True, then return the bone in rest space,
        else return the bone in pose space. Default is False (pose space).
    :returns
        - matrix: [4, 4]. transformation matrix from bone to rest / pose space.
        - tail: [3,]. coordinate of the bone tail in the rest / pose space.
        - head: [3,]. coordinate of the bone head in the rest / pose space.
    """
    matrix_world = armature_obj.matrix_world
    if rest_space:
        bone = armature_obj.data.bones[bone_name]
        matrix = matrix_world @ bone.matrix_local
        tail = matrix_world @ bone.tail_local
        head = matrix_world @ bone.head_local
    else:
        bone = armature_obj.pose.bones[bone_name]
        matrix = matrix_world @ bone.matrix
        tail = matrix_world @ bone.tail
        head = matrix_world @ bone.head
    return matrix, tail, head


def extract_mesh_data(mesh_obj):
    """Extract the mesh data in rest space."""
    verts = np.array([(mesh_obj.matrix_world @ v.co) for v in mesh_obj.data.vertices])
    faces = np.array([poly.vertices for poly in mesh_obj.data.polygons])
    # faces_uvs: (F, 3) LongTensor giving the index into verts_uvs
    #             for each face
    # verts_uvs: (F*3, 2) tensor giving the uv coordinates per vertex
    #             (a FloatTensor with values between 0 and 1).
    faces_uvs = np.array([poly.loop_indices for poly in mesh_obj.data.polygons])
    verts_uvs = np.array(
        [(data.uv.x, data.uv.y) for data in mesh_obj.data.uv_layers.active.data]
    )
    verts = verts.astype(np.float32)
    faces = faces.astype(np.int64)
    verts_uvs = verts_uvs.astype(np.float32)
    faces_uvs = faces_uvs.astype(np.int64)
    return verts, faces, verts_uvs, faces_uvs


def extract_verts_data(mesh_obj, armature_obj):
    """Extract the sequence data of posed verts."""
    init_frame_id = bpy.context.scene.frame_current
    frame_start = int(armature_obj.animation_data.action.frame_range[0])
    frame_end = int(armature_obj.animation_data.action.frame_range[-1])

    verts = []
    for frame_id in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame_id)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm = bmesh.new()
        bm.from_object(mesh_obj, depsgraph)
        bm.verts.ensure_lookup_table()
        verts.append([(mesh_obj.matrix_world @ v.co) for v in bm.verts])
    verts = np.array(verts)

    # reset the frame
    bpy.context.scene.frame_set(init_frame_id)
    return verts.astype(np.float32)


def extract_skeleton_data_grasp(
    num_bones, armature_obj, action, rest_space: bool = False
):

    if rest_space:
        bnames, bnames_parent, matrixs, tails, heads = [], [], [], [], []
        rest_matrix_world = []

        matrixs = np.empty((num_bones, 4, 4))
        tails = np.empty((num_bones, 3))
        heads = np.empty((num_bones, 3))
        rest_matrix_world = np.empty((num_bones, 4, 4))
        bnames = [ '' for _ in range(len(armature_obj.data.bones))]
        bnames_parent = [ '' for _ in range(len(armature_obj.data.bones))]
        for _, bone in enumerate(armature_obj.data.bones):
            bname = bone.name
            bname_parent = None if bone.parent is None else bone.parent.name
            matrix, tail, head, matrix_world = extract_bone_data_grasp(
                armature_obj=armature_obj, bone_name=bname, rest_space=True
            )
            
            idx = int(bname.split('_')[-1])
            bnames[idx] = bname
            bnames_parent[idx] = bname_parent
            matrixs[idx] = matrix
            tails[idx] = tail
            heads[idx] = head
            rest_matrix_world[idx] = matrix_world

        bnames, bnames_parent, matrixs, tails, heads, rest_matrix_world = (
            np.array(bnames),
            np.array(bnames_parent),
            np.array(matrixs),
            np.array(tails),
            np.array(heads),
            np.array(rest_matrix_world),
        )

        return bnames, bnames_parent, matrixs, tails, heads, rest_matrix_world

    else:
        init_frame_id = bpy.context.scene.frame_current
        frame_start = bpy.data.scenes[0].frame_start
        frame_end = bpy.data.scenes[0].frame_end
        num_frames = frame_end - frame_start + 1

        pose_param = np.empty((num_frames, num_bones, 4, 4))
        matrixs = np.empty((num_frames, num_bones, 4, 4))
        tails = np.empty((num_frames, num_bones, 3))
        heads = np.empty((num_frames, num_bones, 3))
        pose_matrix_world = np.empty((num_frames, num_bones, 4, 4))
        eulers = np.empty((num_frames, num_bones , 3))
        for frame_id in range(frame_start, frame_end + 1):
            bpy.context.scene.frame_set(frame_id)
            bpy.context.view_layer.update()

            for _, bone in enumerate(armature_obj.data.bones):
                bname = bone.name
                (
                    matrix,
                    tail,
                    head,
                    matrix_world,
                    param,
                    euler,
                ) = extract_bone_data_grasp(
                    armature_obj=armature_obj, bone_name=bname, rest_space=False
                )
                idx = int(bname.split('_')[-1])

                #  pose_param[-1].append(param)
                input_id = frame_id - frame_start
                pose_param[input_id][idx] = np.array(param)
                matrixs[input_id][idx] = np.array(matrix)
                tails[input_id][idx] = np.array(tail)
                heads[input_id][idx] = np.array(head)
                pose_matrix_world[input_id][idx] = np.array(matrix_world)
                eulers[input_id][idx] = np.array(euler)

        # reset the frame
        bpy.context.scene.frame_set(init_frame_id)
        return matrixs, tails, heads, pose_param, pose_matrix_world, eulers


def extract_skeleton_data(armature_obj, rest_space: bool = False):
    """Extract the skeleton data in rest / pose space."""
    if rest_space:
        bnames, bnames_parent, matrixs, tails, heads = [], [], [], [], []
        for bone in armature_obj.data.bones:
            bname = bone.name
            bname_parent = None if bone.parent is None else bone.parent.name
            matrix, tail, head = extract_bone_data(
                armature_obj=armature_obj, bone_name=bname, rest_space=True
            )

            bnames.append(bname)
            bnames_parent.append(bname_parent)
            matrixs.append(matrix)
            tails.append(tail)
            heads.append(head)
        bnames, bnames_parent, matrixs, tails, heads = (
            np.array(bnames),
            np.array(bnames_parent),
            np.array(matrixs),
            np.array(tails),
            np.array(heads),
        )
        return bnames, bnames_parent, matrixs, tails, heads

    else:
        init_frame_id = bpy.context.scene.frame_current
        frame_start = int(armature_obj.animation_data.action.frame_range[0])
        #  frame_end = int(armature_obj.animation_data.action.frame_range[-1])
        frame_end = bpy.data.scenes[0].frame_end

        matrixs, tails, heads = [], [], []
        for frame_id in range(0, frame_end + 1):
            bpy.context.scene.frame_set(frame_id)
            matrixs.append([])
            tails.append([])
            heads.append([])
            for bone in armature_obj.data.bones:
                bname = bone.name
                matrix, tail, head = extract_bone_data(
                    armature_obj=armature_obj, bone_name=bname, rest_space=False
                )

                matrixs[-1].append(matrix)
                tails[-1].append(tail)
                heads[-1].append(head)
        matrixs, tails, heads = (
            np.array(matrixs),
            np.array(tails),
            np.array(heads),
        )

        #  breakpoint()
        # reset the frame
        bpy.context.scene.frame_set(init_frame_id)
        return matrixs, tails, heads


def extract_all_data(armature_obj):
    """Extract all useful data from the animation."""
    bnames, bnames_parent, rest_matrixs, rest_tails, rest_heads = extract_skeleton_data(
        armature_obj, rest_space=True
    )

    (
        pose_matrixs,
        pose_tails,
        pose_heads,
    ) = extract_skeleton_data(armature_obj, rest_space=False)

    return {
        "bnames": bnames,  # [n_bones,]
        "bnames_parent": bnames_parent,  # [n_bones,]
        "rest_matrixs": rest_matrixs,  # [n_bones, 4, 4]
        "rest_tails": rest_tails,  # [n_bones, 3]
        "rest_heads": rest_heads,  # [n_bones, 3]
        "pose_matrixs": pose_matrixs,  # [n_frames, n_bones, 4, 4]
        "pose_tails": pose_tails,  # [n_frames, n_bones, 3]
        "pose_heads": pose_heads,  # [n_frames, n_bones, 3]
    }


def extract_all_data_grasp(num_bones, armature_obj, action):
    """Extract all useful data from the animation."""
    (
        bnames,
        bnames_parent,
        rest_matrixs,
        rest_tails,
        rest_heads,
        rest_matrix_world,
    ) = extract_skeleton_data_grasp(num_bones, armature_obj, action, rest_space=True)

    (
        pose_matrixs,
        pose_tails,
        pose_heads,
        pose_param,
        pose_matrix_world,
        eulers,
    ) = extract_skeleton_data_grasp(num_bones, armature_obj, action, rest_space=False)

    # root_rotation = pose_param[:, 0, :3, :3]
    # root_translation = pose_param[:, 0, :3, 3]
    pose_param = pose_param[:, :, :3, :3]

    return {
        "bnames": bnames,  # [n_bones,]
        "bnames_parent": bnames_parent,  # [n_bones,]
        "rest_matrixs": rest_matrixs,  # [n_bones, 4, 4]
        "rest_tails": rest_tails,  # [n_bones, 3]
        "rest_heads": rest_heads,  # [n_bones, 3]
        "pose_matrixs": pose_matrixs,  # [n_frames, n_bones, 4, 4]
        "pose_tails": pose_tails,  # [n_frames, n_bones, 3]
        "pose_heads": pose_heads,  # [n_frames, n_bones, 3]
        "pose_params": pose_param,
        "rest_matrix_world": rest_matrix_world,
        "pose_matrix_world": pose_matrix_world,
        "eulers": eulers,
    }


# End animation.py


idx = sys.argv.index("--")
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", "-r", required=True, type=str)
parser.add_argument("--seq_path", "-s", type=str, required=True)
parser.add_argument("--input_dir", default="pose_dumps/joint_angles", type=str)
parser.add_argument("--one_euro", "-o", action="store_true")
parser.add_argument("--frequency", "-f", default=1, type=int)


hand = bpy.data.objects["hand"]

args = parser.parse_args(args=sys.argv[idx + 1 :])

if args.one_euro:
    args.input_dir = "pose_dumps/joint_angles_filtered"
    
root_dir_abs = os.path.abspath(args.root_dir)

base_path = os.path.join(os.getcwd(), args.root_dir, args.seq_path)

with open(os.path.join(args.root_dir, "bone_lens.json")) as f:
    bone_lens = json.load(f)

bpy.ops.object.select_all(action="DESELECT")

hand.select_set(True)
bpy.context.view_layer.objects.active = hand

bpy.ops.object.mode_set(mode="EDIT")

tails = np.zeros((20, 3))
heads = np.zeros((20, 3))
for i in range(20):
    bone = hand.data.edit_bones[f"bone_{i}"]
    heads[i] = np.array(bone.head).copy()
    tails[i] = np.array(bone.tail).copy()

# Set bone lengths
for i in range(20):
    bone = hand.data.edit_bones[f"bone_{i}"]
    dir_vec = np.asarray(tails[i] - heads[i])
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    if bone.parent:
        parent_name = bone.parent.name
        parent_idx = int(parent_name.split("_")[1])
        heads[i] = tails[parent_idx]
    tails[i] = heads[i] + dir_vec * bone_lens[i]

for i in range(20):
    bone = hand.data.edit_bones[f"bone_{i}"]
    bone.head = Vector(heads[i])
    bone.tail = Vector(tails[i])

bpy.ops.object.mode_set(mode="OBJECT")

# Insert keyframes
filenames = natsorted(glob.glob(f"{base_path}/{args.input_dir}/*.json"))
logging.info(f"Found {len(filenames)} frames")

# with open(os.path.join(base_path, "pose_dumps", "usable_frames.json")) as f:
#     usable_frames = json.load(f)
    
# old_filenames = filenames.copy()
# filenames = []
# for file in old_filenames:
#     frame_num = int(os.path.basename(file).split(".")[0])
#     if frame_num in usable_frames:
#         filenames.append(file)
   
logging.info(f"Remaining {len(filenames)} frames after filtering")

# rest_mano_path = os.path.join(args.root_dir, "mano_rest.ply")
# rest_mano= trimesh.load(rest_mano_path, process = False, maintain_order = True)
# rest_mano_verts = rest_mano.vertices
# rest_mano_faces = rest_mano.faces

end = -1
trans_dict = {}
rot_dict = {}
frame_nums = []
# mano_verts = []
for i, file in enumerate(filenames[::args.frequency]):
    with open(file) as f:
        data = json.load(f)
        angles = np.asarray(data["angles"]).reshape(-1, 3)
        
    str_frame_num = os.path.basename(file).split(".")[0]
    frame_num = int(str_frame_num)
    frame_nums.append(frame_num)
    end = max(end, frame_num)
    hand.rotation_mode = "XYZ"
    hand.rotation_euler = Euler(angles[0], "XYZ")
    rot_dict[frame_num] = angles[0]
    trans_dict[frame_num] = data["translation"]
    hand.keyframe_insert("rotation_euler", frame=frame_num)
    hand.location = data["translation"]
    hand.keyframe_insert("location", frame=frame_num)
    for j in range(20):
        hand.pose.bones[f"bone_{j}"].rotation_mode = "XYZ"
        hand.pose.bones[f"bone_{j}"].rotation_euler = Euler(angles[j + 1], "XYZ")
        hand.pose.bones[f"bone_{j}"].keyframe_insert("rotation_euler", frame=frame_num)
        
    ## Load Mano file
    # mano_path = os.path.join(base_path, f"mano/mesh/{str_frame_num}.ply")
    # if os.path.exists(mano_path):
    #     mesh = trimesh.load(mano_path, process = False, maintain_order = True)
    #     verts = mesh.vertices
    # else:
    #     print("")
    #     print("########################     MANO not found !!! ###########################")
    #     print("")
    #     verts = None
    # mano_verts.append(verts)

global_rot = np.zeros((end+1, 3))
translations = np.zeros((end+1, 3))
for key, _ in trans_dict.items():
    global_rot[key] = rot_dict[key]
    translations[key] = trans_dict[key]
    
action = hand.animation_data.action
action.frame_range[0] = 0
action.frame_range[1] = end
bpy.data.scenes["Scene"].frame_start = 0
bpy.data.scenes["Scene"].frame_end = end
logging.info(f"Action Frame Range: {action.frame_range}")
hand_data = extract_all_data_grasp(20, hand, action)
## root translations and root_rotations are sampled every five frames
hand_data['root_translation'] = translations
hand_data['root_rotation'] = global_rot
hand_data['frame_nums'] = frame_nums
# hand_data['mano_verts'] = mano_verts
# hand_data['rest_mano_verts'] = rest_mano_verts
# hand_data['rest_mano_faces'] = rest_mano_faces

if args.frequency > 1:
    hand_metadata_path = os.path.join(base_path, f"meta_data_{args.frequency}.pkl")
else:
    hand_metadata_path = os.path.join(base_path, "meta_data.pkl")

with open(hand_metadata_path, "wb") as fi:
    pickle.dump(hand_data, fi)
    
blendfile_path = os.path.join(base_path, "blendfile.blend") 
bpy.ops.wm.save_as_mainfile(filepath=blendfile_path)
