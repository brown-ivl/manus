import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
import numpy as np
from src.utils.extra import *

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        dim=-1,
    )
    return quaternions


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        # pyre-ignore[16]
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,
    ].reshape(batch_dim + (4,))


def relative_pose(start_pose, target_pose):
    start_pose_inv = torch.linalg.inv(start_pose)
    rel_trans = torch.matmul(target_pose, start_pose_inv)
    return rel_trans


def relative_pose_euler(start_pose, target_pose):
    rel_pose = target_pose - start_pose
    return rel_pose


def get_abs_from_rel(start_pose, rel_pose):
    target_pose = np.matmul(rel_pose, start_pose)
    return target_pose


def to_homo(x):
    """
    converts tensor into homogeneous form for any transformation.
    """
    dim = len(x.shape)
    one = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True).to(x.device)
    if dim == 4:  # BxNxixj
        one = one.reshape(1, 1, 1, 4)
        one = torch.tile(one, (x.shape[0], x.shape[1], 1, 1))
    if dim == 3:  # Bxixj or #Nxixj
        one = one.reshape(1, 1, 4)
        one = torch.tile(one, (x.shape[0], 1, 1))
    ret = torch.cat((x, one), dim=-2)
    return ret


def get_pose_wrt_root(rest_pose, pose_param, global_pose, global_t, kintree):
    BS = pose_param.shape[0]
    rest_pose = torch.tile(rest_pose.unsqueeze(0), (BS, 1, 1, 1))
    global_t = global_t.unsqueeze(-1)

    global_trans = torch.concat([global_pose, global_t], axis=-1)
    #  matrix = torch.zeros_like(rest_pose)
    pose_param = F.pad(pose_param, (0, 1, 0, 0), "constant", 0)
    pose_param = to_homo(pose_param)
    global_trans = to_homo(global_trans)
    matrix = [torch.eye(4).unsqueeze(0) for _ in range(len(kintree))]
    for i in range(len(kintree)):
        if kintree[str(i)] == -1:
            matrix[i] = torch.einsum("Bij,Bjk->Bik", global_trans, rest_pose[:, i])
            matrix[i] = torch.einsum("Bij,Bjk->Bik", matrix[i], pose_param[:, i])

    for i in range(len(kintree)):
        parent = kintree[str(i)]
        if parent == -1:
            continue
        parent_local = rest_pose[:, parent]
        parent_local = torch.linalg.inv(parent_local)
        current_local = rest_pose[:, i]
        local = torch.einsum("Bij,Bjk->Bik", parent_local, current_local)
        pose = torch.einsum("Bij,Bjk->Bik", local, pose_param[:, i])
        pose = torch.einsum("Bij,Bjk->Bik", matrix[parent], pose)
        matrix[i] = pose
    matrix = torch.stack(matrix, dim=1)
    return matrix


def get_pose_wrt_root_dup(rest_pose, pose_param, global_pose, global_t, kintree):
    BS = pose_param.shape[0]
    rest_pose = torch.tile(rest_pose.unsqueeze(0), (BS, 1, 1, 1))
    global_t = global_t.unsqueeze(-1)
    global_trans = torch.concat([global_pose, global_t], axis=-1)
    pose_param = F.pad(pose_param, (0, 1, 0, 0), "constant", 0)
    pose_param = to_homo(pose_param)
    global_trans = to_homo(global_trans)
    matrix = [torch.eye(4).unsqueeze(0) for _ in range(len(kintree))]
    for i in range(len(kintree)):
        if kintree[str(i)] == -1:
            matrix[i] = torch.einsum("Bij,Bjk->Bik", global_trans, rest_pose[:, i])
            matrix[i] = torch.einsum("Bij,Bjk->Bik", matrix[i], pose_param[:, i])

    for i in range(len(kintree)):
        parent = kintree[str(i)]
        if parent == -1:
            continue

        parent_local = rest_pose[:, parent]
        parent_local = torch.linalg.inv(parent_local)
        current_local = rest_pose[:, i]
        local = torch.einsum("Bij,Bjk->Bik", parent_local, current_local)
        pose = torch.einsum("Bij,Bjk->Bik", local, pose_param[:, i])
        pose = torch.einsum("Bij,Bjk->Bik", matrix[parent], pose)
        matrix[i] = pose
    matrix = torch.stack(matrix, dim=1)
    return matrix


def get_keypoints(pose_matrix, rest_pose, rest_joints):
    BS = pose_matrix.shape[0]
    rest_joints = homo(rest_joints).unsqueeze(-1)
    rest_pose_inverse = torch.linalg.inv(rest_pose)
    keypoints = torch.einsum("Nij,Njk->Nik", rest_pose_inverse, rest_joints)
    keypoints = torch.tile(keypoints.unsqueeze(0), (BS, 1, 1, 1))
    keypoints = torch.einsum("BNij,BNjk->BNik", pose_matrix, keypoints)
    return keypoints[:, :, :3, 0]


def project_points(points, K, extrin):
    BS = points.shape[0]
    points = homo(points)
    P = torch.matmul(K, extrin)
    P = torch.tile(P.unsqueeze(0), (BS, points.shape[1], 1, 1))
    points = torch.einsum("BNij,BNj->BNi", P, points)
    points = points / points[..., 2:]
    return points[..., :2]

    # self.limits[6:19:4, 0, 0] = -torch.pi / 6
    # self.limits[6:19:4, 0, 1] = torch.pi / 6
    # self.limits[6:19:4, 2, 0] = -torch.pi / 2
    # self.limits[6:19:4, 2, 1] = torch.pi / 9

    # self.limits[7:20:4, 2, 0] = -torch.pi / 2
    # self.limits[7:20:4, 2, 1] = 0

    # self.limits[8:21:4, 2, 0] = -torch.pi / 2
    # self.limits[8:21:4, 2, 1] = 0


def apply_limits_to_poses(euler_c, bnames):
    limits_xz = ["bone_5", "bone_9", "bone_13", "bone_17"]
    limits_x = [
        "bone_6",
        "bone_7",
        "bone_10",
        "bone_11",
        "bone_14",
        "bone_15",
        "bone_18",
        "bone_19",
    ]
    bone_mapping = {
        "bone_5": [7, 8],
        "bone_6": [9],
        "bone_7": [10],
        "bone_9": [11, 12],
        "bone_10": [13],
        "bone_11": [14],
        "bone_13": [15, 16],
        "bone_14": [17],
        "bone_15": [18],
        "bone_17": [19, 20],
        "bone_18": [21],
        "bone_19": [22],
    }

    for i, bn in enumerate(bnames):
        if bn in limits_xz:
            bone_idx = bone_mapping[bn]
            euler_c[:, bone_idx[0]] = torch.clamp(
                euler_c[:, bone_idx[0]], -torch.pi / 6, torch.pi / 6
            )
            euler_c[:, bone_idx[1]] = torch.clamp(
                euler_c[:, bone_idx[1]], -torch.pi / 2, torch.pi / 9
            )
        elif bn in limits_x:
            bone_idx = bone_mapping[bn]
            euler_c[:, bone_idx[0]] = torch.clamp(
                euler_c[:, bone_idx[0]], -torch.pi / 2, 0
            )
        else:
            euler_c[:, i] = torch.clamp(euler_c[:, i], -torch.pi, torch.pi)
    return euler_c


def apply_constraints_to_poses(
    euler,
    bnames,
    dof_xz=["bone_0", "bone_1", "bone_2", "bone_5", "bone_9", "bone_13", "bone_17"],
    dof_xyz=[],
    dof_x=[
        "bone_3",
        "bone_6",
        "bone_7",
        "bone_10",
        "bone_11",
        "bone_14",
        "bone_15",
        "bone_18",
        "bone_19",
    ],
):
    # return euler.reshape(euler.shape[0], -1)
    """
    Apply bone constraints to the poses
    Wrist: XYZ
    Thumb:
        - CMC (Carpometacarpal): XZ
        - MCP (Meta carpphalangeal): X
        - IP (Interphalangeal): X

    Other Fingers:
        - MCP (Meta carpphalangeal): XZ
        - PIP (Proximal interphalangeal): X
        - DIP (Distal interphalangeal): X
    """
    # return euler.reshape(euler.shape[0], -1)
    num_bones = euler.shape[1]
    tc = len(dof_xz) * 2 + len(dof_xyz) * 3 + len(dof_x) * 1
    euler_c = np.zeros((euler.shape[0], tc), dtype=np.float32)
    count = 0
    for i, bn in enumerate(bnames):
        if bn in dof_xyz:
            euler_c[:, count : count + 3] = euler[:, i, :]
            count += 3
        elif bn in dof_xz:
            euler_c[:, count] = euler[:, i, 0]
            euler_c[:, count + 1] = euler[:, i, 2]
            count += 2
        elif bn in dof_x:
            euler_c[:, count] = euler[:, i, 2]
            count += 1
    euler_c = np.array(euler_c)
    return euler_c


def remove_constraints_to_poses(
    euler_c,
    bnames,
    dof_xz=["bone_0", "bone_1", "bone_2", "bone_5", "bone_9", "bone_13", "bone_17"],
    dof_xyz=[],
    dof_x=[
        "bone_3",
        "bone_6",
        "bone_7",
        "bone_10",
        "bone_11",
        "bone_14",
        "bone_15",
        "bone_18",
        "bone_19",
    ],
    repeated=["bone_4", "bone_8", "bone_12", "bone_16"],
    device=torch.device("cpu"),
):
    """
    Remove bone constraints to the poses
    Wrist: XYZ
    Thumb:
        - CMC (Carpometacarpal): XZ
        - MCP (Meta carpphalangeal): X
        - IP (Interphalangeal): X

    Other Fingers:
        - MCP (Meta carpphalangeal): XZ
        - PIP (Proximal interphalangeal): X
        - DIP (Distal interphalangeal): X
    """
    # return euler_c.reshape(euler_c.shape[0], -1, 3)
    num_bones = len(dof_xz) + len(dof_xyz) + len(dof_x) + len(repeated)

    euler = torch.zeros((euler_c.shape[0], num_bones, 3), dtype=torch.float32).to(
        device
    )

    count = 0
    ## Assuming batch_size is 1
    for i, bn in enumerate(bnames):
        if bn in dof_xyz:
            euler[:, i, :] = euler_c[:, count : count + 3]
            count += 3
        elif bn in dof_xz:
            euler[:, i, 0] = euler_c[:, count]
            euler[:, i, 2] = euler_c[:, count + 1]
            count += 2
        elif bn in dof_x:
            euler[:, i, 2] = euler_c[:, count]
            count += 1

    return euler


def euler_angles_to_quats(euler):
    """
    Convert rotations given as Euler angles to quaternions.
    Args:
        euler: Euler angles in radians as tensor of shape (..., 3).
    Returns:
        Quaternions with real part first, as tensor of shape (..., 4).
    """
    return matrix_to_quaternion(euler_angles_to_matrix(euler, "XYZ", intrinsic=True))


def euler_angles_to_matrix(euler_angles, convention, intrinsic=False):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    Proof for intrinsic and extrinsic rotation:
    https://math.stackexchange.com/questions/1137745/proof-of-the-extrinsic-to-intrinsic-rotation-transform

    This function may be buggy for the 4D
    """

    if intrinsic:
        convention = convention[::-1]
        if euler_angles.dim() == 3:
            euler_angles = euler_angles.flip(
                2,
            )
        elif euler_angles.dim() == 2:
            euler_angles = euler_angles.flip(
                1,
            )

    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def convert_armature_space_to_world_space(data):
    data = edict(data)
    ## Multiplying with Matrix World
    data.rest_matrixs = np.einsum(
        "Bij,Bjk->Bik", data.rest_matrix_world, data.rest_matrixs
    )
    one = np.ones((data.rest_tails.shape[0], 1))
    rest_tails = np.concatenate([data.rest_tails, one], axis=-1)[..., np.newaxis]
    rest_heads = np.concatenate([data.rest_heads, one], axis=-1)[..., np.newaxis]
    data.rest_tails = np.einsum("Bij,Bjk->Bik", data.rest_matrix_world, rest_tails)[
        :, :3, 0
    ]
    data.rest_heads = np.einsum("Bij,Bjk->Bik", data.rest_matrix_world, rest_heads)[
        :, :3, 0
    ]

    data.pose_matrixs = np.einsum(
        "BNij,BNjk->BNik", data.pose_matrix_world, data.pose_matrixs
    )
    one = np.ones((data.pose_tails.shape[0], data.pose_tails.shape[1], 1))
    pose_tails = np.concatenate([data.pose_tails, one], axis=-1)[..., np.newaxis]
    pose_heads = np.concatenate([data.pose_heads, one], axis=-1)[..., np.newaxis]
    data.pose_tails = np.einsum("BNij,BNjk->BNik", data.pose_matrix_world, pose_tails)[
        :, :, :3, 0
    ]
    data.pose_heads = np.einsum("BNij,BNjk->BNik", data.pose_matrix_world, pose_heads)[
        :, :, :3, 0
    ]

    return data


def euler_angles_to_armature_space(pose, kintree, rest_matrixs, global_T):
    # Convert the Euler angles to the matrix form
    pose_matrixs = euler_angles_to_matrix(pose, "XYZ", intrinsic=True)

    # Separate the global pose and pred pose from pred_pose
    global_R = pose_matrixs[:, 0]
    pose_matrixs = pose_matrixs[:, 1:]

    # Conversion from local pose to armature space using kintree
    pose_matrixs = get_pose_wrt_root(
        rest_matrixs, pose_matrixs, global_R, global_T, kintree
    )

    return pose_matrixs


def build_kintree(bnames, bnames_parent):
    if not isinstance(bnames, list):
        bnames = bnames.tolist()
    if not isinstance(bnames_parent, list):
        bnames_parent = bnames_parent.tolist()

    kintree = {}
    for idx, _ in enumerate(bnames):
        parent = bnames_parent[idx]
        if (parent is not None) and (parent != "None"):
            parent_idx = bnames.index(parent)
            kintree[str(idx)] = parent_idx
        else:
            kintree[str(idx)] = -1
    return kintree
