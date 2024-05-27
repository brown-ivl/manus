import torch
from src.models.gaussian import GaussianModel
from src.utils.extra import *
from src.utils.gaussian_utils import skinning_weights_from_voxel_grid

class HandGaussianModel(GaussianModel):
    """
    
    """
    def __init__(
        self,
        opts,
        points=None,
        points_colors=None,
        mano_weights=None,
        grid_center=None,
        grid_scale=None,
        grid_points=None,
        grid_weights=None,
        grid_mask=None,
    ):
        super().__init__(opts, points, points_colors)

        if mano_weights is None:
            mano_weights = np.random.randn(opts.num_gaussians, 20)

        self.opts = opts
        self.skin_weights_init_type = self.opts.skin_weights_init_type
        self.grid_center = grid_center
        self.grid_scale = grid_scale
        self.grid_points = grid_points
        self.grid_weights = grid_weights
        self.grid_weights_init = grid_weights
        self.grid_mask = grid_mask
        self._skin_weights = torch.empty(0)

        ## Initialize skinning weights
        device = torch.device("cuda")
        if self.skin_weights_init_type == "mano_init_voxel":
            if points is not None:
                fused_point_cloud = attach(to_tensor(points), device)
                self._skin_weights = skinning_weights_from_voxel_grid(
                    fused_point_cloud,
                    self.grid_center,
                    self.grid_scale,
                    self.grid_weights,
                )
                self._skin_weights = attach(self._skin_weights, device)
            else:
                self._skin_weights = None
        elif self.skin_weights_init_type == "mano_init_points":
            if mano_weights is not None:
                mano_weights = to_tensor(mano_weights)
                mano_weights = attach(mano_weights, device)
                self._skin_weights = mano_weights
        else:
            raise ValueError("Unknown skin weights init type")

        ## Turn it off and on only when needed
        self.optimizing_skin_weights = False
        self.optimizing_offsets = False

        self.n_transforms = 21

    def get_skin_weights(self):
        skin_wts = None
        device = self.get_xyz.device
        if self.skin_weights_init_type == "mano_init_voxel":
            self._skin_weights = skinning_weights_from_voxel_grid(
                self.get_xyz, self.grid_center, self.grid_scale, self.grid_weights
            )
            skin_wts = attach(self._skin_weights, device)
        else:
            skin_wts = attach(self._skin_weights, device)

        return skin_wts
