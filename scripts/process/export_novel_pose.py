import bpy
import numpy as np
import pickle
import bmesh  # type: ignore
import logging

logging.basicConfig(level=logging.INFO)


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
        global_rot = np.empty((num_frames, 3))
        global_trans = np.empty((num_frames, 3))
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
            global_trans[input_id] = np.array([armature_obj.location[0], armature_obj.location[1], armature_obj.location[2]])
            global_rot[input_id] = np.array([armature_obj.rotation_euler[0], armature_obj.rotation_euler[1], armature_obj.rotation_euler[2]])

        # reset the frame
        bpy.context.scene.frame_set(init_frame_id)
        return matrixs, tails, heads, pose_param, pose_matrix_world, eulers, global_rot, global_trans


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


def extract_skinning_weights_data(armature_obj, mesh_obj, normalize=True):
    """Extract the skinning weights data."""
    n_verts = len(mesh_obj.data.vertices)
    n_bones = len(armature_obj.data.bones)

    vg_names = [vg.name for vg in mesh_obj.vertex_groups]
    bone_names = [bone.name for bone in armature_obj.data.bones]
    vg_to_bone_map = [
        bone_names.index(vg_name) if vg_name in bone_names else None
        for vg_name in vg_names
    ]

    weights = np.zeros((n_verts, n_bones), dtype=np.float32)
    for i in range(n_verts):
        for grp in mesh_obj.data.vertices[i].groups:
            bone_id = vg_to_bone_map[grp.group]
            if bone_id is not None:
                weights[i, bone_id] = grp.weight
    if normalize:
        weights = weights / weights.sum(axis=1, keepdims=True)
    return weights


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
        global_rot, 
        global_trans
    ) = extract_skeleton_data_grasp(num_bones, armature_obj, action, rest_space=False)

    # root_rotation = pose_param[:, 0, :3, :3]
    # root_translation = pose_param[:, 0, :3, 3]
    pose_param = pose_param[:, :, :3, :3]
    rest_matrix_world = np.stack([np.eye(4) for i in range(20)])

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
        "root_translation": global_trans, 
        "root_rotation": global_rot
    }


# End animation.py


BASE_PATH = bpy.path.abspath("//")

hand = bpy.data.objects['hand']
end = 250

action = hand.animation_data.action
action.frame_range[0] = 0
action.frame_range[1] = end
bpy.data.scenes["Scene"].frame_start = 0
bpy.data.scenes["Scene"].frame_end = end
logging.info(f"Action Frame Range: {action.frame_range}")
hand_data = extract_all_data_grasp(20, hand, action)

hand_metadata_path = "./meta_data0.pkl"
with open(hand_metadata_path, "wb") as fi:
    pickle.dump(hand_data, fi)
