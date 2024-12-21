import torch
import torch.nn.functional as F

def homo(points: torch.Tensor) -> torch.Tensor:
    """Get the homogeneous coordinates."""
    return F.pad(points, (0, 1), value=1)

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


def project_points(joints, K, extrin, matrix_world=None):
    BS = joints.shape[0]
    joints = to_homo(joints)
    if matrix_world is not None:
        joints = torch.einsum("BNij,BNj->BNi", matrix_world, joints)

    P = torch.matmul(K, extrin)
    P = torch.tile(P.unsqueeze(0), (BS, joints.shape[1], 1, 1))
    joints = torch.einsum("BNij,BNj->BNi", P, joints)
    joints = joints / joints[..., 2:]
    return joints[..., :2]


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