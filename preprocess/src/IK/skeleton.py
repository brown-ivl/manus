import numpy as np
import torch
from tqdm import trange
from torch.optim import Adam, SGD, RAdam
from adabelief_pytorch import AdaBelief
from torch.nn.parameter import Parameter
from src.utils.transforms import (
    euler_angles_to_matrix,
    get_keypoints,
    get_pose_wrt_root,
)


class KinematicChain:
    def __init__(self, bones, device=torch.device("cpu")):
        self.device = device
        self.bones = bones
        self.parents = [-1 for _ in range(len(bones))]
        # Temporary, rename if stuff works
        self.rest_matrices = [np.eye(4) for _ in range(len(bones))]
        self.kintree = {}
        self.tails = torch.zeros(
            (len(self.bones), 3), dtype=torch.float32, device=device
        )
        self.heads = torch.zeros(
            (len(self.bones), 3), dtype=torch.float32, device=device
        )
        for _, bone in bones.items():
            bone_idx = bone["idx"]
            self.rest_matrices[bone_idx] = bone["rest_matrix"]
            self.parents[bone_idx] = bone["parent_idx"]
            self.kintree[str(bone_idx)] = bone["parent_idx"]
            self.tails[bone_idx] = torch.tensor(bone["tail"])
            self.heads[bone_idx] = torch.tensor(bone["head"])
        self.rest_matrices = torch.from_numpy(np.asarray(self.rest_matrices)).to(device).float()

        self.dof = torch.zeros(
            (len(self.bones) + 1, 3), dtype=torch.bool
        )  # num_joint, 3 (x, y, z)

        self.limits = torch.zeros(
            (len(self.bones) + 1, 3, 2), dtype=torch.float, device=device
        )  # num_joint, 3 (x, y, z), 2 (low, high)

        self.limits[:, :, 0] = -torch.pi
        self.limits[:, :, 1] = torch.pi

        xz = [True, False, True]  # useful for dof

        # DOF taken from http://www.iri.upc.edu/files/academic/pfc/74-PFC.pdf#page=31
        # RU
        self.dof[0, :] = True  # all

        # CMC*
        self.dof[1, xz] = True  # xz
        self.dof[2, xz] = True  # xz
        
        # TODO: Figure out limits on thumbs
        self.limits[1, 0, 0] = -torch.pi / 9
        self.limits[1, 0, 1] = torch.pi / 9
        # self.limits[1, 2, 0] = -torch.pi / 2
        # self.limits[1, 2, 1] = 0
        self.limits[2, 0, 0] = -torch.pi / 9
        self.limits[2, 0, 1] = torch.pi / 9

        # CMC
        # self.dof[5:18:4, 2] = True  # z
        # self.limits[5:18:4, 2, 0] = -torch.pi / 2
        # self.limits[5:18:4, 2, 1] = 0

        # MCP
        self.dof[3, xz] = True  # xz
        self.dof[6:19:4, xz] = True
        self.limits[6:19:4, 0, 0] = -torch.pi / 6
        self.limits[6:19:4, 0, 1] = torch.pi / 6
        self.limits[6:19:4, 2, 0] = -torch.pi / 2
        self.limits[6:19:4, 2, 1] = torch.pi / 9

        # PIP
        self.dof[4, 2] = True
        self.dof[7:20:4, 2] = True
        self.limits[7:20:4, 2, 0] = -torch.pi / 2
        self.limits[7:20:4, 2, 1] = torch.pi / 9

        # DIP
        self.dof[8:21:4, 2] = True
        self.limits[8:21:4, 2, 0] = -torch.pi / 2
        self.limits[8:21:4, 2, 1] = 0
       
    def allow_metacarpal_dof(self, constraint_on=False, limits_on=False, axes = [False, False, True]):
        
        self.dof[5, axes] = True if constraint_on else False  # xz
        self.dof[9, axes] = True if constraint_on else False  # xz
        self.dof[13, axes] = True if constraint_on else False  # xz
        self.dof[17, axes] = True if constraint_on else False  # xz
        
        self.limits[5, 0, 0] = -torch.pi / 8 if limits_on else 0.0
        self.limits[5, 0, 1] =  torch.pi / 8 if limits_on else 0.0
        
        self.limits[9, 0, 0] = -torch.pi / 8 if limits_on else 0.0
        self.limits[9, 0, 0] = torch.pi / 8 if limits_on else 0.0
        
        self.limits[13, 0, 0] = -torch.pi / 8 if limits_on else 0.0
        self.limits[13, 0, 1] =  torch.pi / 8 if limits_on else 0.0
        
        self.limits[17, 0, 0] = -torch.pi / 8 if limits_on else 0.0
        self.limits[17, 0, 1] =  torch.pi / 8 if limits_on else 0.0
            

    def plot_skeleton(self, trans, angles, target=None):
        """Debug function"""
        import polyscope as ps
        import trimesh

        ps.init()

        (
            _,
            heads,
            tails,
        ) = self.forward(trans, angles)
        heads = heads.detach().cpu().numpy()
        tails = tails.detach().cpu().numpy()
        pt_cloud = ps.register_point_cloud("heads", heads)
        ps.register_point_cloud("tails", tails)

        pt_cloud.add_vector_quantity(
            "bones",
            tails - heads,
            color=(0, 0, 0),
            enabled=True,
            vectortype="ambient",
            radius=0.004,
        )

        if not target is None:
            ps.register_point_cloud("target", target)

        pcd = trimesh.PointCloud(heads)
        _ = pcd.export("points.ply")

        ps.show()

    def loss(self, trans_params, angle_params, target, to_use, constraint, limit=False):
        predicted, _, __ = self.forward(trans_params, angle_params, constraint)
        keypoint_loss = predicted - target
        keypoint_loss = torch.square(torch.linalg.norm(keypoint_loss, axis=1))
        
        ## Give more weightage to the tip finger joints
        tip_idx = [4, 8, 12, 16, 20]
        keypoint_loss[tip_idx] = keypoint_loss[tip_idx] * 2
        keypoint_loss = keypoint_loss[to_use]
        keypoint_loss = keypoint_loss.mean()
        
        loss = {"keypoint_loss": keypoint_loss}
        if limit:
            if constraint:
                limits = self.limits.reshape(-1, 2)[self.dof.flatten()]
            else:
                limits = self.limits.reshape(-1, 2)
                
            low_limit = limits[:, 0] 
            hi_limit = limits[:, 1]
            l_hinge1 = torch.max(torch.zeros_like(low_limit), angle_params - hi_limit)**2
            l_hinge2 = torch.max(torch.zeros_like(low_limit), low_limit - angle_params)**2
            
            limit_loss = torch.sum(l_hinge1 + l_hinge2)
            
            # limit_loss_lo = limits[:, 0] - angle_params
            # limit_loss_hi = limits[:, 1] - angle_params
            
            # limit_loss_lo = torch.where(limit_loss_lo > 0.0, limit_loss_lo.long(), 0)
            # limit_loss = torch.sum(torch.square(limit_loss_lo))
            # limit_loss_hi = torch.where(limit_loss_hi < 0.0, limit_loss_hi.long(), 0)
            # limit_loss += torch.sum(torch.square(limit_loss_hi))
            loss["limit_loss"] = limit_loss
        return loss
    
    def update_rest_pose(self, angle_params, trans_params):
        
        ## Warning: Doesn't work
        
        angle_params = angle_params.reshape(-1, 3)
        ik_keyp, heads, tails = self.forward(trans_params, angle_params, constraint=False)
        
        posed_keyps = ik_keyp[5:18:4]
        posed_keyps = posed_keyps - ik_keyp[0]
        posed_keyps = np.array(posed_keyps.detach().cpu())
        posed_vec = posed_keyps / np.linalg.norm(posed_keyps, axis=1, keepdims=True)
        posed_vec_avg = posed_vec.mean(axis=0)
        posed_rel_angle = np.arccos(np.dot(posed_vec, posed_vec_avg))
        
        trans_params = torch.zeros(3, device=self.device)
        update_rest_params = torch.zeros((len(self.bones) + 1) * 3, device=self.device).reshape(-1, 3)
        rest, heads, tails = self.forward(trans_params, update_rest_params, constraint=False)
        
        rest_keyps = rest[5:18:4]
        rest_keyps = rest_keyps - rest[0]
        rest_keyps = np.array(rest_keyps.detach().cpu())
        norm = np.linalg.norm(rest_keyps, axis=1, keepdims=True)
        rest_vec = rest_keyps / norm
        rest_vec_avg = rest_vec.mean(axis=0)
        rest_vec_avg = rest_vec_avg / np.linalg.norm(rest_vec_avg)
        rest_rel_angle = np.arccos(np.dot(rest_vec, rest_vec_avg))
        
        delta_angles = [] 
        
        from scipy.spatial.transform import Rotation as R
        
        for i in range(4):
            cross = np.cross(rest_vec_avg, rest_vec[i])
            axis = cross / np.linalg.norm(cross)
            angle = posed_rel_angle[i]
            
            if i > 1:
                angle = -1 *  angle
                
            rotmat = R.from_rotvec(angle * axis).as_matrix()
            updated_vec =rotmat @ rest_vec_avg
            
            cross = np.cross(rest_vec[i], updated_vec)
            axis = cross / np.linalg.norm(cross)
            angle = np.arccos(np.dot(rest_vec[i], updated_vec))
            
            if i > 1:
                angle = -1 * angle 
                
            euler = R.from_rotvec(angle * axis).as_euler('xyz')
            delta_angles.append(euler)
        
        delta_angles = np.array(delta_angles)  
        
        ## TODO: (Doesn't work)
        ## Dump the updated rest pose
        ## Use Blender to update rest pose 
        ## Dump rest pose from the Blender
        ## Continue IK fitting with new rest pose. 
        
        update_rest_params = update_rest_params.detach().cpu().numpy()
        update_rest_params[5] = delta_angles[0]
        update_rest_params[9] = delta_angles[1]
        update_rest_params[13] = delta_angles[2]
        update_rest_params[17] = delta_angles[3]
        
        update_rest_params = update_rest_params.tolist()
        
        for idx, (bone_name, bone) in enumerate(self.bones.items()):
            self.bones[bone_name]["rest_pose"] = update_rest_params[idx+1]
            
        return self.bones

    def forward(self, trans_params, angle_params, constraint=False):
        angles = torch.zeros((len(self.bones) + 1) * 3, device=self.device)
        if constraint:
            angles[self.dof.flatten()] = angle_params
        else:
            angles = angle_params

        angles = angles.reshape(-1, 3)

        pose_matrices = euler_angles_to_matrix(angles, "XYZ", intrinsic=True)
        global_translation = trans_params.unsqueeze(0)
        matrix = get_pose_wrt_root(
            self.rest_matrices,
            pose_matrices[1:].unsqueeze(0),
            pose_matrices[:1],
            global_translation,
            self.kintree,
        )

        heads = get_keypoints(matrix, self.rest_matrices, self.heads).squeeze(0)
        tails = get_keypoints(matrix, self.rest_matrices, self.tails).squeeze(0)

        scaled_tails = torch.zeros_like(tails)
        scaled_heads = torch.zeros_like(heads)
        for i in range(len(self.kintree)):
            parent = self.kintree[str(i)]
            if parent == -1:
                scaled_heads[i] = heads[i]
            else:
                scaled_heads[i] = scaled_tails[parent]
            dir_vec = tails[i] - heads[i]
            dir_vec = dir_vec / torch.linalg.norm(dir_vec)
            scaled_tails[i] = scaled_heads[i] + dir_vec * self.bones[f"bone_{i}"]["len"]

        keypoints = torch.vstack([scaled_heads[:1], scaled_tails])
        return keypoints, scaled_heads, scaled_tails

    def update_bone_lengths(self, keypoints: np.ndarray):
        for bone_name, bone in self.bones.items():
            curr_id = bone["idx"] + 1
            parent_id = bone["parent_idx"] + 1
            bone_vecs = keypoints[:, curr_id] - keypoints[:, parent_id]
            to_use = ~torch.logical_or(
                torch.isclose(
                    keypoints[:, curr_id, 3], torch.tensor(0.0, device=self.device)
                ),
                torch.isclose(
                    keypoints[:, parent_id, 3], torch.tensor(0.0, device=self.device)
                ),
            )
            if not torch.count_nonzero(to_use):
                raise ValueError(f"No frame has length of bone {bone_name}")
            bone_lens = torch.linalg.norm(bone_vecs[:, :3], axis=1)[to_use]
            self.bones[bone_name]["len"] = bone_lens.mean().item()

    def IK(
        self,
        target,
        to_use,
        constraint,
        limit,
        lr=1e-1,
        trans_init=None,
        angles_init=None,
        max_iter=10000,
        threshold=1e-6
    ):
        if trans_init is None:
            trans_init = torch.zeros(3, device=self.device)
        trans_params = Parameter(trans_init)

        if angles_init is None:
            # 20 Joint angles + Global rotation
            angles_init = torch.zeros(
                (len(self.bones) + 1, 3), device=self.device
            ).flatten()

        if constraint:
            angle_params = Parameter(angles_init[self.dof.flatten()])
        else:
            angle_params = Parameter(angles_init)

        optimizer = AdaBelief([trans_params, angle_params], lr = lr,  eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
        # optimizer = Adam([trans_params, angle_params], lr=lr, betas=(0.9, 0.999))
            
        pbar = trange(max_iter)
        least_loss = 1e10
        least_iter = 0
        least_params = angle_params
        for i in pbar:
            optimizer.zero_grad()
            loss = self.loss(
                trans_params, angle_params, target, to_use, constraint, limit
            )
            
            limit_loss = loss['limit_loss'] if limit else 0
            total_loss = loss["keypoint_loss"] + limit_loss
            
            total_loss.backward()
            optimizer.step()
            
                
            pbar.set_description(
                f"loss: {total_loss:.6f}"
                + f", keypoint_loss: {loss['keypoint_loss']:.6f}"
                + (f", limit_loss: {loss['limit_loss']:.6f}"
                if limit
                else "")
            )

            # Early stopping
            if total_loss < least_loss:
                least_loss = total_loss
                least_iter = i
                least_params = angle_params.detach().clone()
            if i - least_iter > 20 :
                if abs(total_loss - least_loss) < threshold:
                    break

        if constraint:
            to_return = torch.zeros(
                (len(self.bones) + 1, 3), device=self.device
            ).flatten()
            to_return[self.dof.flatten()] = least_params
        else:
            to_return = least_params

        return trans_params, to_return