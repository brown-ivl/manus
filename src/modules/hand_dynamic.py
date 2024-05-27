from easydict import EasyDict as edict
import lpips
import hydra
from src.modules.base import BaseTrainingModule
from src.utils.gaussian_utils import (
    render_gaussians,
    strip_symmetric,
    update_learning_rate,
    get_points_outside_mask,
)
from src.utils.train_utils import (
    remove_nans_from_checkpoint,
)
from src.utils.loss_utils import psnr
from src.utils.transforms import project_points
from src.utils.extra import *


class TrainingModule(BaseTrainingModule):
    """
    Training Module for the Hand Dynamic Model
    """
    def __init__(self, opts, mode):
        super().__init__(opts, mode)
        self.opts = opts
        self.mode = mode
        self.automatic_optimization = False
        self.do_density_update = True

        dataset = self.train_data if self.mode != "test" else self.test_data

        points, points_colors, mano_weights = None, None, None
        grid_scale, grid_center, grid_points, grid_weights, grid_mask = (
            None,
            None,
            None,
            None,
            None,
        )

        if self.mode != "test":
            dataset = self.train_data
            points, points_colors, mano_weights = dataset.sample_gaussians_on_bones(
                self.opts.model.opts.sample_size,
                mano_weights=True,
                init_type=self.opts.model.opts.skin_weights_init_type,
            )
            if self.opts.model.opts.skin_weights_init_type == "mano_init_voxel":
                grid_size = self.opts.model.opts.grid_size
                grid_res = self.opts.model.opts.grid_res
                grid_offset = self.opts.model.opts.grid_offset
                grid_scale, grid_center, grid_points, grid_weights, grid_mask = (
                    dataset.build_voxel_grid(
                        ratio=grid_size, res=grid_res, offset=grid_offset
                    )
                )

            with torch.no_grad():
                self.loss_fn_vgg = lpips.LPIPS(net="vgg")

        self.model = hydra.utils.instantiate(
            self.opts.model,
            points=points,
            points_colors=points_colors,
            mano_weights=mano_weights,
            grid_scale=grid_scale,
            grid_center=grid_center,
            grid_points=grid_points,
            grid_weights=grid_weights,
            grid_mask=grid_mask,
            _recursive_=False,
        )

        self.model.training_setup() if self.mode != "test" else None

    def configure_optimizers(self):
        return self.model.optimizer

    def get_offset_grid(self, pose_latent):
        grid_points = self.model.grid_points.reshape(-1, 3).to(self.device)
        pose_latent = pose_latent[None].repeat(grid_points.shape[0], 1)
        offset_grid = self.model.offset_mlp(grid_points, pose_latent)
        offset_grid = offset_grid.reshape(self.model.grid_points.shape)
        return offset_grid

    def forward(self, batch):
        cano_xyz = self.model.get_xyz
        device = cano_xyz.device
        bones_posed = batch["bones_posed"]
        bones_rest = batch["bones_rest"]
        posed_transforms = attach(bones_posed.transforms, device)
        rest_transforms = attach(bones_rest.transforms, device)
        transforms = torch.einsum(
            "nij,njk->nik", posed_transforms, torch.linalg.inv(rest_transforms)
        )
        skin_wts = self.model.get_skin_weights()

        if self.opts.model.opts.skin_weights_init_type == "mano_init_voxel":
            # Append a background transform as Identity
            transforms = torch.cat(
                [transforms, torch.eye(4)[None, :, :].to(transforms)], dim=0
            )

        assert skin_wts.shape[-1] == transforms.shape[0]

        tf = torch.einsum("nb, bij->nij", skin_wts, transforms)
        posed_xyz = torch.einsum("nij, nj->ni", tf, homo(cano_xyz))[..., :3]

        """
        ## Visualization for debugging
        keypts = torch.cat([bones_posed.heads[:1], bones_posed.tails], axis=0)
        keypts2d = project_points(keypts[None], camera.K, camera.extr[:, :3, :4])[0]
        keypts2d = to_numpy(keypts2d)
        img = plot_points_in_image(keypts2d, to_numpy(batch['rgb'][0] * 255))
        dump_image(img, './projected_points.png')
        breakpoint()
        colors = visualize_skin_weights(skin_wts)
        dump_points(posed_xyz, 'posed_xyz.ply', colors )
        dump_points(keypts, 'keypts.ply')
        breakpoint()
        """

        cov = self.model.get_covariance(full=True)
        cov = torch.einsum(
            "bij,bjk,bkl->bil", tf[..., :3, :3], cov, tf[..., :3, :3].transpose(1, 2)
        )
        cov = strip_symmetric(cov)
        pred = {
            "posed_xyz": posed_xyz,
            "posed_cov": cov,
            "cano_xyz": self.model.get_xyz,
            "cano_features": self.model.get_features,
            "cano_opacity": self.model.get_opacity,
            "tf": tf,
            "skin_wts": skin_wts,
        }
        return edict(pred)

    def render(self, batch):
        pred = self(batch)
        rendered = render_gaussians(
            pred.posed_xyz,
            pred.posed_cov,
            pred.cano_xyz,
            pred.cano_features,
            pred.cano_opacity,
            batch["camera"],
            batch["bg_color"],
            sh_degree=self.model.opts.sh_degree,
            tf=pred.tf,
        )

        if self.trainer.validating:
            # Visualize pose as well
            bones_posed = batch["bones_posed"]
            camera = batch["camera"]
            keypts = torch.cat([bones_posed.heads[:1], bones_posed.tails], axis=0)
            keypts2d = project_points(keypts[None], camera.K, camera.extr[:, :3, :4])[0]
            keypts2d = to_numpy(keypts2d)
            # render = plot_points_in_image(keypts2d, to_numpy(rendered['render'] * 255))
            # rgb = plot_points_in_image(keypts2d, to_numpy(batch['rgb'][0] * 255))[None]
            # batch['rgb'] = attach(to_tensor(rgb / 255), batch['rgb'].device)
            # rendered['render'] = attach(to_tensor(render / 255), rendered['render'].device)

        rendered.update(pred)
        return rendered

    def dump_gaussians(self, pred, batch_idx):
        colors = visualize_skin_weights(self.model.get_skin_weights())
        if self.mode == "train":
            results_dir = self.val_results_dir
        else:
            results_dir = self.test_results_dir
        # dump_points(pred['posed_xyz'], os.path.join(results_dir, f'{self.global_step}_posed.ply'), colors)
        dump_points(
            pred["posed_xyz"],
            os.path.join(results_dir, f"{batch_idx}_posed.ply"),
            colors,
        )

        # Dump cano weights
        dump_points(
            pred["cano_xyz"],
            os.path.join(results_dir, f"{self.global_step}_cano.ply"),
            colors,
        )

        # ## Dump the grid weights
        # colors = visualize_skin_weights(self.model.grid_weights.reshape(-1, self.model.grid_weights.shape[-1]))
        # grid_points = self.model.grid_points.reshape(-1, 3)
        # dump_points(grid_points, os.path.join(results_dir, f'{self.global_step}_grid_points.ply'), colors)

    def on_after_backward(self):
        if self.do_density_update:
            if self.global_step < self.model.opts.remove_seg_end:
                camera = self.rendered["camera"]
                points = self.rendered["posed_xyz"].detach()
                mask = self.rendered["mask"]
                bones_posed = self.rendered["bones_posed"]
                heads = bones_posed.heads
                tails = bones_posed.tails
                keypoints = torch.cat([heads[:1], tails], axis=0)

                pts_mask = get_points_outside_mask(
                    camera, points, mask, keypoints, dilate=True
                )[..., 0]
                self.pts_mask += pts_mask
            else:
                prune_points_with_bbox = True
                if self.global_step % 100 == 0:
                    posed_xyz = self.rendered["posed_xyz"]
                    bones_posed = self.rendered["bones_posed"]
                    heads = bones_posed.heads
                    tails = bones_posed.tails
                    keypoints = torch.cat([heads[:1], tails], axis=0)
                    mask = torch.cdist(posed_xyz, keypoints).mean(1) > 0.2
                    self.pts_mask += mask

            prune_mask = self.pts_mask.clone() if self.pts_mask.sum() > 0 else None

            res = self.density_update(
                self.rendered, self.trainer.optimizers[0], prune_mask
            )
            if res:
                self.pts_mask = torch.zeros(
                    self.model.get_xyz.shape[0], dtype=torch.bool, device=self.device
                )

    def on_before_optimizer_step(self, opts):
        if len(opts.param_groups) > 1:
            opts = update_learning_rate(opts, self.model, self.global_step)

    def on_train_epoch_start(self):
        self.pts_mask = torch.zeros(
            self.model.get_xyz.shape[0], dtype=torch.bool, device=self.device
        )

    def training_step(self, batch, batch_idx):
        rendered = self.render(batch)
        final_loss = self.loss_func(
            batch,
            rendered,
            self.opts.losses,
            self.opts.loss_weight,
            self.opts.trainer.log_losses,
        )

        final_loss = final_loss / self.opts.trainer.accum_iter

        if self.training:
            rendered["camera"] = batch["camera"]
            rendered["mask"] = batch["mask"]
            rendered["bones_posed"] = batch["bones_posed"]

        self.rendered = rendered

        opts = self.optimizers()

        if ((batch_idx + 1) % self.opts.trainer.accum_iter == 0) or (
            batch_idx + 1 == len(self.train_data)
        ):
            if is_list(opts):
                for opt in opts:
                    opt.zero_grad()
            else:
                opts.zero_grad()

        self.manual_backward(final_loss)

        if ((batch_idx + 1) % self.opts.trainer.accum_iter == 0) or (
            batch_idx + 1 == len(self.train_data)
        ):
            if is_list(opts):
                for opt in opts:
                    opt.step()
            else:
                opts.step()

        self.log("loss", final_loss, sync_dist=True, batch_size=self.batch_size)
        psnr_val = psnr(rendered["render"], batch["rgb"][0])
        self.log("train/psnr", psnr_val, sync_dist=True, batch_size=self.batch_size)
        return {"loss": final_loss}

    def on_load_checkpoint(self, checkpoint):
        if self.opts.model.opts.skin_weights_init_type == "mano_init_voxel":
            self.model.grid_scale = checkpoint["extra_params"]["grid_scale"]
            self.model.grid_center = checkpoint["extra_params"]["grid_center"]
            self.model.grid_points = checkpoint["extra_params"]["grid_points"]
            self.model.grid_weights = checkpoint["extra_params"]["grid_weights"]
        elif self.opts.model.opts.skin_weights_init_type == "mano_init_points":
            self.model._skin_weights = checkpoint["extra_params"]["mano_weights"]

        checkpoint = remove_nans_from_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        if "extra_params" not in checkpoint:
            checkpoint["extra_params"] = {}

        my_dic_keys = list(checkpoint["state_dict"].keys())

        for key in my_dic_keys:
            if "loss_fn_vgg" in key:
                del checkpoint["state_dict"][key]
            if "lpips_evalfn" in key:
                del checkpoint["state_dict"][key]

        checkpoint["extra_params"]["num_gaussians"] = self.model.get_xyz.shape[0]

        if self.opts.model.opts.skin_weights_init_type == "mano_init_voxel":
            checkpoint["extra_params"]["grid_scale"] = self.model.grid_scale
            checkpoint["extra_params"]["grid_center"] = self.model.grid_center
            checkpoint["extra_params"]["grid_points"] = self.model.grid_points
            checkpoint["extra_params"]["grid_weights"] = self.model.grid_weights
        else:
            checkpoint["extra_params"]["mano_weights"] = self.model.get_skin_weights()
