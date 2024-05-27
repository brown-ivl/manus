import cv2
import numpy as np
import torch
import json
from natsort import natsorted
import h5py
import joblib
import math
from src.utils import params as param_utils
from src.utils.extra import *
from src.utils.transforms import (
    project_points,
    apply_constraints_to_poses,
    build_kintree,
    euler_angles_to_quats,
    convert_armature_space_to_world_space,
    get_pose_wrt_root,
    euler_angles_to_matrix,
    get_keypoints,
)
from src.utils.cam_utils import get_opengl_camera_attributes, get_scene_extent
from src.utils.extra import create_skinning_grid, create_voxel_grid
from src.utils.train_utils import (
    init_mano_weights,
    sample_gaussians_on_bones_func,
    sample_gaussians_on_mano,
)


class Dataset(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    def __init__(self, opts, split="train"):
        self.resize_factor = opts.resize_factor
        self.split = split
        self.near = opts.near
        self.far = opts.far
        self.training = split == "train"
        self.bg_color = opts.bg_color
        self.opts = opts
        self.subject_id = opts.subject
        self.width = opts.width
        self.height = opts.height

        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            opts.root_dir,
        )

        self.dtype = torch.get_default_dtype()
        self.root_dir = root_fp
        self.actions, self.index_list, self.metadata_dict, to_choose = (
            self.dataset_index_list(
                self.root_dir,
                split,
                self.opts.num_time_steps,
                self.opts.split_ratio,
                self.opts.rand_views_per_timestep,
            )
        )

        # Compute the cameras once
        self.get_all_cameras(to_choose)
        # self.print_data_stats()

        super().__init__()

    def sample_gaussians_on_bones(
        self, sample_size, mano_weights=False, init_type=None
    ):
        action = self.index_list[0][0]
        fno = self.index_list[0][1]
        bones_rest = self.metadata_dict[action][fno]["bones_rest"]
        bones_rest = to_tensor(bones_rest)

        points, points_colors = sample_gaussians_on_bones_func(bones_rest, sample_size)
        # points, points_colors = sample_gaussians_on_mano(self.mano_data, sample_size)

        if mano_weights:
            if init_type == "mano_init_voxel":
                weights, mask = init_mano_weights(
                    points, self.mano_data, filter_grid=True
                )
            elif init_type == "mano_init_points":
                weights, mask = init_mano_weights(
                    points, self.mano_data, filter_grid=False
                )
            weights = to_tensor(weights)
            colors = visualize_skin_weights(weights)
        else:
            weights = None
            colors = None

        dump_points(points, "./init_gaussians.ply", colors)
        return points, points_colors, weights

    def build_voxel_grid(
        self, grid_boundary=[-1, 1], res=128, ratio=[1, 1, 1], offset=[0, 0, 0]
    ):
        action = self.index_list[0][0]
        fno = self.index_list[0][1]
        bones_rest = self.metadata_dict[action][fno]["bones_rest"]
        bones_rest = to_tensor(bones_rest)
        heads = bones_rest.heads
        tails = bones_rest.tails
        keypts = to_numpy(torch.cat([heads[:1], tails], dim=0))
        cano_min = np.min(keypts, axis=0)
        cano_max = np.max(keypts, axis=0)
        center = (cano_max + cano_min) / 2
        offset = np.array(offset)
        center += offset

        x_ratio, y_ratio, z_ratio = ratio
        res_scaled = res / np.array([x_ratio, y_ratio, z_ratio])
        res_scaled = res_scaled.astype(np.int32)
        d, h, w = (
            math.floor(res_scaled[2]),
            math.floor(res_scaled[1]),
            math.floor(res_scaled[0]),
        )
        grid_points = create_skinning_grid(d, h, w)
        scale = np.linalg.norm(cano_max - cano_min) / 2
        scale = np.array([[scale * z_ratio, scale * y_ratio, scale * x_ratio]]).astype(
            np.float32
        )
        grid_points = grid_points * scale + center
        weights, mask = init_mano_weights(
            to_tensor(grid_points), self.mano_data, neighbors=4
        )
        weights = to_tensor(weights)
        grid_points = to_tensor(grid_points)
        colors = visualize_skin_weights(weights)
        dump_points(grid_points, "./init_grid.ply", colors)
        grid_points = grid_points.reshape(
            res_scaled[2], res_scaled[1], res_scaled[0], 3
        )
        weights = weights.reshape(
            res_scaled[2], res_scaled[1], res_scaled[0], weights.shape[-1]
        )
        if mask is not None:
            mask = mask.reshape(res_scaled[2], res_scaled[1], res_scaled[0])
        return to_tensor(scale), to_tensor(center), grid_points, weights, mask

    def dataset_index_list(
        self, root_dir, split, num_time_steps, split_ratio, rand_views_per_timestep
    ):
        actions = natsorted([fp for fp in os.listdir(root_dir)])
        if self.opts.sequences != "all":
            chosen_actions = []
            for action in self.opts.sequences:
                action = f"{action}.hdf5"
                if action in actions:
                    chosen_actions.append(action)
            actions = chosen_actions

        if len(actions) == 1 and self.opts.split_by_action:
            split_ratio = -1

        if (split_ratio > 0) and self.opts.split_by_action:
            if split == "train":
                actions = actions[: int(split_ratio * len(actions))]
            else:
                actions = actions[int(split_ratio * len(actions)) :]

        index_list = []
        metadata_dict = {}
        for idx, action_path in enumerate(actions):
            action = action_path.split(".")[0]
            metadata_dict[action] = {}
            h5_path = os.path.join(root_dir, action_path)
            with h5py.File(
                h5_path,
                "r",
            ) as file:
                frame_nos = list(file["frames"].keys())
                Ks = file.get("K")
                cam_names = list(Ks.keys())

                # Make a metadata dict here only

                for fno in frame_nos:
                    metadata = file["frames"][fno]["metadata"]
                    metadata = self.fetch_metadata(metadata)
                    metadata["frame_id"] = fno
                    metadata["action"] = action
                    metadata_dict[action][fno] = metadata

            frame_nos = natsorted(frame_nos)

            if (num_time_steps < 0) or (num_time_steps > len(frame_nos)):
                to_choose = frame_nos
            else:
                to_choose = frame_nos[:: (len(frame_nos) // num_time_steps)]

            for fno in to_choose:
                if rand_views_per_timestep < 0:
                    for view in cam_names:
                        index_list.extend([(action, fno, view)])
                else:
                    index_list.extend([(action, fno, None)])

        if not self.opts.split_by_action:
            if split_ratio > 0:
                if split == "train":
                    index_list = index_list[: int(split_ratio * len(index_list))]
                else:
                    index_list = index_list[int(split_ratio * len(index_list)) :]

            with open(f"./{split}_split.json", "w") as f:
                json.dump(index_list, f)

        return actions, index_list, metadata_dict, to_choose

    def get_all_cameras(self, to_choose):
        ## Cameras are not changing for the actions
        action = self.index_list[0][0]

        h5_path = os.path.join(self.root_dir, f"{action}.hdf5")
        d_dict = defaultdict(list)

        mano_data = {}
        with h5py.File(
            h5_path,
            "r",
        ) as file:
            mano = file.get("mano_rest")
            for k, v in mano.items():
                mano_data[k] = v[:]

            frames = file.get("frames")
            Ks = file.get("K")
            self.cam_names = list(Ks.keys())
            self.cam2idx = {
                cam_name: idx for idx, cam_name in enumerate(self.cam_names)
            }
            extrs = file.get("extr")
            frames_list = list(frames.keys())
            self.all_cameras = []

            ## Loading all the choosen timesteps
            for _ in to_choose:
                for cam_name in self.cam_names:
                    K = Ks[cam_name][:]
                    extr = extrs[cam_name][:]
                    attr_dict = get_opengl_camera_attributes(
                        K,
                        extr,
                        self.width,
                        self.height,
                        resize_factor=self.resize_factor,
                    )
                    for k, v in attr_dict.items():
                        d_dict[k].append(v)
                    d_dict["cam_name"].append(cam_name)

        for k, v in d_dict.items():
            d_dict[k] = np.stack(v, axis=0)

        self.all_cameras = Cameras(**d_dict)
        self.extent = get_scene_extent(self.all_cameras.camera_center)
        self.mano_data = mano_data

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        data = self.fetch_data(idx)
        return data

    def fetch_data_by_frame(self, action, frame_id, cam_name):
        try:
            idx = self.index_list.index((action, frame_id, cam_name))
            data = self.fetch_data(idx)
        except:
            data = None
        return data

    def fetch_metadata(self, metadata):
        bnames = [n[0].decode("UTF-8") for n in metadata["bnames"][:].tolist()]
        bnames = np.array(bnames)
        bnames_parent = [
            n[0].decode("UTF-8") for n in metadata["bnames_parent"][:].tolist()
        ]
        bone_ids = np.arange(self.opts.n_bones).tolist()

        b_dict_rest = {
            "bnames": bnames,
            "heads": metadata["rest_heads"][bone_ids],
            "tails": metadata["rest_tails"][bone_ids],
            "transforms": metadata["rest_matrixs"][bone_ids],
        }
        bones_rest = Bones(**b_dict_rest)

        eulers = metadata["eulers"][:]
        eulers_c = apply_constraints_to_poses(eulers[None], bnames)
        kintree = build_kintree(bnames, bnames_parent)
        r_T = metadata["root_translation"][:]
        r_R = metadata["root_rotation"][:]

        b_dict_posed = {
            "bnames": bnames,
            "heads": metadata["pose_heads"][bone_ids],
            "tails": metadata["pose_tails"][bone_ids],
            "transforms": metadata["pose_matrixs"][bone_ids],
            "eulers": eulers,
            "eulers_c": eulers_c,
            "root_translation": r_T,
            "root_rotation": r_R,
            "kintree": kintree,
        }

        bones_posed = Bones(**b_dict_posed)

        quats = euler_angles_to_quats(
            to_tensor(np.concatenate([eulers, r_R[None]], axis=0))
        )
        pose_latent = quats.flatten()
        # pose_latent = torch.cat([to_tensor(r_T), quats.flatten()])

        metadata_dict = {
            "bones_rest": bones_rest,
            "bones_posed": bones_posed,
            "pose_latent": pose_latent,
        }
        return metadata_dict

    def get_roi_mask(self, img, roi):
        roi_mask = np.zeros_like(img[..., 3])
        roi_mask[roi[1] : roi[3], roi[0] : roi[2]] = 255
        return roi_mask

    def get_bg_color(self):
        if self.bg_color == "random":
            color_bkgd = np.random.rand(3).astype(np.float32)
        elif self.bg_color == "white":
            color_bkgd = np.ones(3).astype(np.float32)
        elif self.bg_color == "black":
            color_bkgd = np.zeros(3).astype(np.float32)
        return color_bkgd

    def fetch_images(self, data, cam_name):
        images = data["images"]
        bboxes = data["bbox"]
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        crop_img = images[cam_name][:]
        bbox = bboxes[cam_name][:]
        xmin, ymin, xmax, ymax = bbox
        try:
            img[ymin:ymax, xmin:xmax, :] = crop_img
        except:
            breakpoint()
        # roi_mask = self.get_roi_mask(img, self.rois[idx])
        # modify_mask = cv2.bitwise_and(roi_mask, img[..., 3])
        # img[..., 3] = modify_mask.astype(np.uint8)
        img = cv2.resize(
            img,
            (0, 0),
            fx=self.resize_factor,
            fy=self.resize_factor,
            interpolation=cv2.INTER_AREA,
        )
        img = img / 255.0

        color_bkgd = self.get_bg_color()

        if img.shape[-1] == 4:
            rgb = img[..., :3]
            alpha = img[..., 3:]
            img[..., :3] = rgb * alpha + color_bkgd * (1.0 - alpha)

        return img

    def get_data_from_h5(self, index):
        action, frame_id, cam_name = self.index_list[index]

        ## Randomly Choose cameras for each timestep
        if cam_name is None:
            cam_name = np.random.default_rng().choice(
                self.cam_names, size=self.opts.rand_views_per_timestep, replace=False
            )
        else:
            cam_name = [cam_name]

        h5_path = os.path.join(self.root_dir, f"{action}.hdf5")
        with h5py.File(
            h5_path,
            "r",
        ) as file:
            frames = file.get("frames")
            data = frames[str(frame_id)]
            metadata = self.metadata_dict[action][str(frame_id)]
            rgba_list = []
            camidx = []
            for cam in cam_name:
                rgba = self.fetch_images(data, cam)
                rgba_list.append(rgba)
                camidx.append(self.cam2idx[cam])
        rgba_list = np.array(rgba_list)
        cameras = self.all_cameras[camidx]
        info = [self.subject_id, action, frame_id, cam_name]
        return rgba_list, cameras, metadata, info

    def fetch_data(self, index):
        rgba, camera, metadata_dict, info = self.get_data_from_h5(index)
        rgba = np.stack(rgba, axis=0)
        image = rgba[..., :3]
        mask = rgba[..., 3:]
        color_bkgd = self.get_bg_color()

        data_dict = {
            "info": info,
            "rgb": to_tensor(image),
            "mask": to_tensor(mask),
            "camera": to_tensor(camera),
            "scaling_modifier": 1.0,
            "bg_color": to_tensor(color_bkgd),
            "bones_rest": to_tensor(metadata_dict["bones_rest"]),
            "bones_posed": to_tensor(metadata_dict["bones_posed"]),
            "pose_latent": to_tensor(metadata_dict["pose_latent"]),
        }
        return data_dict

    def get_all_images_per_frame(self, action, frame_id):
        images = []
        for cam in self.cam_names:
            idx = self.index_list.index((action, frame_id, cam))
            rgba, camera, metadata_dict, info = self.get_data_from_h5(idx)
            rgba = rgba / 255.0
            images.append(rgba)
        images = np.array(images)
        return images

    def get_prune_mask(self, points):
        points_mask = np.zeros(points.shape[0])

        action, frame_id = "grasp_mug", "390"
        images = self.get_all_images_per_frame(action, frame_id)

        for idx, cam in enumerate(self.all_cameras):
            l_mask = np.zeros_like(points_mask)
            cam = to_tensor(cam)
            K, R, T = cam.K, cam.R, cam.T
            extrin = torch.cat([R, T[..., None]], dim=1)
            p2d = project_points(points[None], K, extrin)[0]
            mask = images[idx][..., -1]

            x, y = p2d[..., 0], p2d[..., 1]
            mask_x = torch.logical_and(x >= 0, x < self.width)
            mask_y = torch.logical_and(y >= 0, y < self.height)
            mask_xy = torch.logical_and(mask_x, mask_y)
            masked_p2d = p2d[mask_xy]
            masked_p2d = to_numpy(masked_p2d)
            masked_p2d = masked_p2d.astype(np.int32)
            nmask = mask[masked_p2d[..., 1], masked_p2d[..., 0]] == 1
            indices = np.where(mask_xy)[0][nmask]

            l_mask[indices] = 1
            points_mask += l_mask

            ## Visualize 2D points
            # vis = np.zeros((self.height, self.width, 3))
            # vis[masked_p2d[..., 1], masked_p2d[..., 0]] = [1, 1, 1]
            # os.makedirs('./vis', exist_ok=True)
            # cv2.imwrite(f'./vis/{idx}.png', vis * 255)
            # cv2.imwrite(f'./vis/{idx}_mask.png', mask * 255)

        ## Prune all points which are not visible most of the cameras
        points_mask = points_mask < 35
        dump_points(points[points_mask == 0], "./prune_points.ply")
        # breakpoint()
        return points_mask

    @classmethod
    def encode_meta_id(cls, action, frame_id):
        return "%s___%05d" % (action, int(frame_id))

    @classmethod
    def decode_meta_id(cls, meta_id: str):
        action, frame_id = meta_id.split("___")
        return action, int(frame_id)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, opts, split="train"):
        super().__init__()
        self.resize_factor = opts.resize_factor
        self.bg_color = opts.color_bkgd_aug
        self.frame_sample_rate = opts.frame_sample_rate
        self.width = 1080
        self.height = 1080
        self.test_on_canonical_pose = opts.test_on_canonical_pose
        self.n_bones = 20
        self.subject_id = opts.subject
        self.mode = opts.contact_render_type

        self.cam_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", opts.cam_path
        )

        self.metadata_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", opts.metadata_path
        )

        self.cano_cam_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", opts.cano_cam_path
        )

        cam_names = None
        if self.mode == "acc_gt_eval":
            if ".txt" in self.cam_path:
                params = param_utils.read_params(self.cam_path)
                intrs = []
                extrs = []
                cam_names = []
                for param in params:
                    cam_names.append(param["cam_name"])
                    intr, dist = param_utils.get_intr(param)
                    new_intr, roi = param_utils.get_undistort_params(
                        intr, dist, (self.width, self.height)
                    )
                    fx, fy, cx, cy = (
                        new_intr[0, 0],
                        new_intr[1, 1],
                        new_intr[0, 2],
                        new_intr[1, 2],
                    )
                    intrs.append([fx, fy, cx, cy])
                    extrs.append(param_utils.get_extr(param))
                cam_data = {"intrs": intrs, "extrs": extrs, "cam_name": cam_names}
            else:
                cam_data = joblib.load(self.cam_path)
        else:
            cam_data = joblib.load(self.cam_path)
        meta_data = joblib.load(self.metadata_path)
        action = self.metadata_path.split("/")[-2]
        cano_cam_data = joblib.load(self.cano_cam_path)

        meta_data = convert_armature_space_to_world_space(meta_data)

        bone_ids = np.arange(self.n_bones).tolist()

        bnames = [n for n in meta_data["bnames"][:].tolist()]
        b_dict_rest = {
            "bnames": bnames,
            "heads": meta_data["rest_heads"][bone_ids],
            "tails": meta_data["rest_tails"][bone_ids],
            "transforms": meta_data["rest_matrixs"][bone_ids],
        }
        self.bones_rest = Bones(**b_dict_rest)
        self.bones_rest = to_tensor(self.bones_rest)

        Ks = cam_data["intrs"]
        extrins = cam_data["extrs"]

        self.all_cameras = []
        self.bones_posed_list = []
        self.pose_latent_list = []
        self.pose_latent_list = []
        d_dict = defaultdict(list)

        if self.mode == "gt_eval":
            selected_indices = meta_data["frame_nums"][-250:]
            frame_ids = selected_indices
            self.n_frames = len(selected_indices)
            selected = np.array(selected_indices)

        elif self.mode == "acc_gt_eval":
            self.n_frames = len(extrins)
            selected = np.arange(0, meta_data["pose_tails"].shape[0], 1)
            frame_ids = selected
        else:
            self.n_frames = len(extrins[:: self.frame_sample_rate])
            if meta_data["pose_tails"].shape[0] < self.n_frames:
                b_step = 1
            else:
                b_step = math.ceil(meta_data["pose_tails"].shape[0] / self.n_frames)
            selected = np.arange(0, meta_data["pose_tails"].shape[0], b_step)
            frame_ids = selected

        meta_data["pose_tails"] = meta_data["pose_tails"][selected]
        meta_data["pose_heads"] = meta_data["pose_heads"][selected]
        meta_data["pose_matrixs"] = meta_data["pose_matrixs"][selected]

        eulers = meta_data["eulers"]
        r_T = meta_data["root_translation"]
        r_R = meta_data["root_rotation"]

        value = to_tensor(np.concatenate([r_R[:, None, :], eulers], axis=1))
        quats = euler_angles_to_quats(value)
        pose_latent = quats.reshape(-1, quats.shape[1] * quats.shape[2])
        pose_latent = pose_latent[selected]

        self.infos = []
        for i in range(self.n_frames):
            cam_name = cam_names[i] if cam_names is not None else str(i)

            if i > meta_data["pose_tails"].shape[0] - 1:
                idx = meta_data["pose_tails"].shape[0] - 1
            else:
                idx = i

            frame_id = frame_ids[idx]
            info = [self.subject_id, action, str(frame_id), cam_name]
            self.infos.append(info)

            if self.test_on_canonical_pose:
                self.bones_posed_list.append(self.bones_rest)
                self.pose_latent_list.append(pose_latent[idx])
            else:
                b_dict_posed = {
                    "bnames": bnames,
                    "heads": meta_data["pose_heads"][idx, bone_ids],
                    "tails": meta_data["pose_tails"][idx, bone_ids],
                    "transforms": meta_data["pose_matrixs"][idx, bone_ids],
                }

                bones_posed = Bones(**b_dict_posed)
                self.pose_latent_list.append(pose_latent[idx])

                bones_posed = to_tensor(bones_posed)

                self.bones_posed_list.append(bones_posed)

            extr = extrins[i]
            fx, fy, cx, cy = Ks[i]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            if self.mode == "acc_gt_eval":
                attr_dict = get_opengl_camera_attributes(
                    K, extr, 1080, 1080, resize_factor=1.0
                )

            else:
                attr_dict = get_opengl_camera_attributes(
                    K, extr, self.width, self.height, resize_factor=1.0
                )

            for k, v in attr_dict.items():
                d_dict[k].append(v)
            d_dict["cam_name"].append(str(i))

        for k, v in d_dict.items():
            d_dict[k] = np.stack(v, axis=0)

        self.all_cameras = Cameras(**d_dict)
        self.all_cameras = to_tensor(self.all_cameras)

        cano_K = cano_cam_data["intrs"][0]
        cano_K = np.array(
            [[cano_K[0], 0, cano_K[2]], [0, cano_K[1], cano_K[3]], [0, 0, 1]]
        )
        cano_extr = cano_cam_data["extrs"][0]

        cano_d_dict = {}
        cano_attr = get_opengl_camera_attributes(
            cano_K, cano_extr, self.width, self.height
        )
        for k, v in cano_attr.items():
            cano_d_dict[k] = [v]
        cano_d_dict["cam_name"] = ["0"]

        for k, v in cano_d_dict.items():
            cano_d_dict[k] = np.stack(v, axis=0)

        self.cano_camera = to_tensor(Cameras(**cano_d_dict))

    def __getitem__(self, index):
        data = self.fetch_data(index)
        return data

    def fetch_data(self, index):
        camera = self.all_cameras[index]
        bones_rest = self.bones_rest
        bones_posed = self.bones_posed_list[index]
        pose_latent = self.pose_latent_list[index]
        data_dict = {
            "idx": index,
            "info": self.infos[index],
            "camera": camera,
            "cano_camera": self.cano_camera,
            "scaling_modifier": 1.0,
            "bones_rest": bones_rest,
            "bones_posed": bones_posed,
            "pose_latent": pose_latent,
            "bg_color": (
                torch.tensor([1.0, 1.0, 1.0])
                if self.bg_color == "white"
                else torch.tensor([0.0, 0.0, 0.0])
            ),
        }
        return data_dict

    def __len__(self):
        return self.n_frames
