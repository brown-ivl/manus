import cv2
import joblib
from src.utils.extra import *
from src.utils import params as param_utils
from src.utils.cam_utils import get_opengl_camera_attributes, get_scene_extent


class Dataset(torch.utils.data.Dataset):
    def __init__(self, opts, split='train'):
        self.width = opts.width
        self.height = opts.height
        self.resize_factor = opts.resize_factor
        self.instance_dir = opts.root_dir
        self.split = split
        self.bg_color = opts.bg_color

        if not os.path.exists(self.instance_dir):
            raise ValueError(f"Data directory {self.instance_dir} is empty")

        image_dir = os.path.join(self.instance_dir, "images", "refined_seg")

        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory {image_dir} does not exist")

        # self.cam_file = os.path.join(self.instance_dir, '../..', 'calib', 'optim_params.txt')
        self.cam_file = os.path.join(opts.params_dir, 'optim_params.txt')
        if not os.path.exists(self.cam_file):
            raise ValueError("Camera file does not exist")

        cameras = param_utils.read_params(self.cam_file)

        # Remove the lower hemisphere views
        skip_images = [
            "brics-sbc-003_cam0",
            "brics-sbc-003_cam1",
            "brics-sbc-004_cam1",
            "brics-sbc-008_cam0",
            "brics-sbc-008_cam1",
            "brics-sbc-009_cam0",
            "brics-sbc-013_cam0",
            "brics-sbc-013_cam1",
            "brics-sbc-014_cam0",
            "brics-sbc-018_cam0",
            "brics-sbc-018_cam1",
            "brics-sbc-019_cam0",
            # "brics-sbc-027_cam0",
            # "brics-sbc-027_cam1",
            # "brics-sbc-028_cam0",
            # "brics-sbc-029_cam0",
            # "brics-sbc-029_cam1",
            # "brics-sbc-030_cam0",
            # "brics-sbc-030_cam1",
        ]

        cameras = [cam for cam in cameras if cam["cam_name"] not in skip_images]
        # cameras = np.random.choice(cameras, 9)

        if opts.split_ratio < 0.0:
            pass
        else:
            if self.split == 'train':
                # cameras = cameras[:int(opts.split_ratio * len(cameras))]
                cameras = cameras[2:]
            else:
                cameras = cameras[:2]
                # cameras = cameras[int(opts.split_ratio * len(cameras)):]



        self.images = []
        self.masks = []
        d_dict = defaultdict(list)
        self.cam2idx = {}

        for idx, cam in enumerate(cameras):
            extr = param_utils.get_extr(cam)
            K, dist = param_utils.get_intr(cam)
            cam_name = cam["cam_name"]

            img_path = os.path.join(image_dir, cam_name, '*')
            img_path = glob.glob(img_path)[0]

            self.cam2idx[cam_name] = idx

            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            new_K, roi = param_utils.get_undistort_params(K, dist, (self.width, self.height))
            image = param_utils.undistort_image(K, new_K, dist, image)
            new_K = new_K.astype(np.float32)
            extr = extr.astype(np.float32)

            attr_dict = get_opengl_camera_attributes(new_K, extr, self.width, self.height,
                                                     resize_factor=self.resize_factor)
            for k, v in attr_dict.items():
                d_dict[k].append(v)
            d_dict['cam_name'].append(cam_name)

            if image.shape[-1] == 4:
                b, g, r, alpha = cv2.split(image)
            else:
                b, g, r = cv2.split(image)

            rgb = np.stack([r, g, b], axis=-1)
            alpha = alpha[..., np.newaxis] / 255.0
            mask = alpha

            rgb = rgb / 255.0
            color_bkgd = self.get_bg_color()
            rgb = rgb * alpha + color_bkgd * (1.0 - alpha)
            self.images.append(rgb)
            self.masks.append(mask)

        for k, v in d_dict.items():
            d_dict[k] = np.stack(v, axis=0)

        self.all_cameras = Cameras(**d_dict)
        self.extent = get_scene_extent(self.all_cameras.camera_center)
        dump_points(self.all_cameras.camera_center, f'./cam_centers_{self.split}.ply')
        self.n_images = len(self.images)
        self.all_cameras = to_tensor(self.all_cameras)

    def get_bg_color(self):
        if self.bg_color == "random":
            color_bkgd = np.random.rand(3).astype(np.float32)
        elif self.bg_color == "white":
            color_bkgd = np.ones(3).astype(np.float32)
        elif self.bg_color == "black":
            color_bkgd = np.zeros(3).astype(np.float32)
        return color_bkgd

    def sample_gaussians(self, sample_size, sample_mesh=False):
        mesh_dir = os.path.join(self.instance_dir, 'mesh', 'ngp_mesh')
        mesh_path = glob.glob(os.path.join(mesh_dir, '*.ply'))[0]
        mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
        points = mesh.sample(sample_size)

        if not sample_mesh:
            rand_ind = np.random.randint(sample_size, size=sample_size)
            grid_points = create_voxel_grid([-1, 1], 256)
            np.random.shuffle(grid_points)
            grid_points = grid_points[rand_ind]
            mean = points.mean(axis=0)
            scale = np.max(np.max(points, axis=0) - np.min(points, axis=0)) / 2
            points = grid_points * scale + mean
        else:
            noise = np.random.normal(0, 0.03, points.shape)
            points = points + noise
        points_colors = torch.rand(points.shape)

        dump_points(points, './init_gaussians.ply', points_colors)
        return points, points_colors

    def get_scene_extent(self, cam_centers):
        ## Used in densify_and_prune
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        radius = diagonal * 1.1
        return radius

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        sampling_idx = idx
        rgb = self.images[sampling_idx]
        mask = self.masks[sampling_idx]
        camera = self.all_cameras[sampling_idx]

        data = {
            "rgb": torch.from_numpy(rgb).float(),
            "mask": torch.from_numpy(mask).float(),
            "camera": camera,
            "scaling_modifier": 1.0,
            "bg_color": to_tensor(self.get_bg_color())
        }
        return data

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def get_scale_mat(self):
        return np.load(self.cam_file)["scale_mat_0"]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, opts, split="test"):
        super().__init__()
        self.resize_factor = opts.resize_factor
        self.bg_color = opts.color_bkgd_aug
        self.frame_sample_rate = opts.frame_sample_rate
        self.width = 1280
        self.height = 720

        self.cam_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            opts.cam_path
        )

        cam_data = joblib.load(self.cam_path)
        Ks = cam_data["intrs"]
        extrins = cam_data["extrs"]
        self.n_frames = len(extrins[::self.frame_sample_rate])

        d_dict = defaultdict(list)

        for i in range(self.n_frames):
            fx, fy, cx, cy = Ks[i]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            RT = extrins[i]
            RT = np.vstack((RT, np.array([0, 0, 0, 1])))

            attr_dict = get_opengl_camera_attributes(K, RT[:3, :4], self.width, self.height,
                                                     resize_factor=self.resize_factor)

            for k, v in attr_dict.items():
                d_dict[k].append(v)
            d_dict['cam_name'].append(str(i))

        for k, v in d_dict.items():
            d_dict[k] = np.stack(v, axis=0)

        self.all_cameras = Cameras(**d_dict)

    def __getitem__(self, index):
        data = self.fetch_data(index)
        return data

    def fetch_data(self, index):
        camera = self.all_cameras[index]
        camera = to_tensor(camera)
        data_dict = {
            "idx": index,
            "camera": camera,
            "scaling_modifier": 1.0,
            "bg_color": torch.tensor([1.0, 1.0, 1.0]) if self.bg_color == "white" else torch.tensor([0.0, 0.0, 0.0])
        }
        return data_dict

    def __len__(self):
        return self.n_frames

