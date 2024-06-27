import cv2
import json
from tqdm import tqdm
from pytorch_lightning import LightningModule
from src.utils.transforms import project_points
from src.utils.extra import *
from src.utils.gaussian_utils import density_update
from src.utils.train_utils import setup_dataloaders, sample_on_bones
from easydict import EasyDict as edict
from src.utils.loss_utils import l1_loss, ssim, psnr, l2_loss, lpips_loss, write_csv
import time


class BaseTrainingModule(LightningModule):
    def __init__(self, opts, mode):
        super(BaseTrainingModule, self).__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.batch_size = opts.trainer.batch_size
        self.num_workers = opts.trainer.num_workers
        self.opts = opts

        self.test_on_train_dataset = self.opts.test_dataset.opts.test_on_train_dataset
        self.worst_cases = self.opts.test_dataset.opts.worst_cases

        if self.worst_cases:
            self.test_on_train_dataset = True

        if self.test_on_train_dataset:
            self.opts.test_dataset = self.opts.train_dataset
            self.opts.test_dataset.opts.split_ratio = 0

        if mode != "test":
            self.train_data, self.train_loader = setup_dataloaders(
                self.opts.train_dataset,
                self.batch_size,
                self.num_workers,
                "train",
                False,
                True,
            )
            self.val_data, self.val_loader = setup_dataloaders(
                self.opts.train_dataset,
                self.batch_size,
                self.num_workers,
                "val",
                False,
                True,
            )
        else:
            self.test_data, self.test_loader = setup_dataloaders(
                self.opts.test_dataset,
                self.batch_size,
                self.num_workers,
                "test",
                False,
                True,
            )

        self.exp_name = opts.trainer.exp_name
        self.result_dir = "results/"
        os.makedirs(self.result_dir, exist_ok=True)
        self.automatic_optimization = False

    def update_learning_rate(self, optimizer):
        """Learning rate scheduling per step"""
        for param_group in optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.model.xyz_scheduler_args(self.global_step)
                param_group["lr"] = lr
                return lr

    def on_save_checkpoint(self, checkpoint):
        if "extra_params" not in checkpoint:
            checkpoint["extra_params"] = {}
        checkpoint["extra_params"]["num_gaussians"] = self.model.get_xyz.shape[0]

    def on_train_epoch_start(self):
        self.train_outputs = []

    def forward(self, batch):
        pass

    def render(self, batch):
        pass

    def density_update(self, pred, optimizer, mask_to_prune=None):
        res = density_update(
            self.model,
            pred,
            self.train_data.extent,
            self.global_step,
            self.train_data.bg_color,
            mask_to_prune,
        )
        if res:
            self.trainer.optimizers[0] = optimizer
        return res

    def training_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_start(self):
        self.val_results_dir = os.path.join(self.result_dir, "val_results")
        os.makedirs(self.val_results_dir, exist_ok=True)
        self.val_images = []
        self.psnr_vals = []
        self.ssim_vals = []
        self.lpips_vals = []
        self.render_time = []

    def validation_step(self, batch, batch_idx):
        start = time.time()
        pred = self.render(batch)
        end = time.time()
        img = to_numpy(torch.clamp(pred["render"], 0, 1))
        img = (img * 255).astype(np.uint8)
        if len(batch["rgb"].shape) == 4:
            gt_img = batch["rgb"][0]
        else:
            gt_img = batch["rgb"]

        gt_img = dump_image(gt_img, return_img=True)
        diff = gt_img / 255.0 - img / 255.0
        diff = diff * 255.0
        final_img = concat_img_array(img, gt_img)
        final_img = concat_img_array(final_img, diff)
        self.val_images.append(final_img)

        # if self.model._skin_weights is not None:
        #     colors = visualize_skin_weights(self.model.get_skin_weights)
        # else:
        colors = None

        if batch_idx == 0:
            self.dump_gaussians(pred, batch_idx)

        gt = batch["rgb"]
        mask = batch["mask"]
        if len(gt.shape) == 4:
            gt = gt[0]
            mask = mask[0]

        render = pred["render"] * mask
        gt = gt * mask
        psnr_val = psnr(render, gt)
        ssim_val = ssim(render, gt)

        lpips_val = torch.mean(lpips_loss(render, gt.unsqueeze(0)))

        self.psnr_vals.append(to_numpy(psnr_val))
        self.ssim_vals.append(to_numpy(ssim_val))
        self.lpips_vals.append(to_numpy(lpips_val))
        self.render_time.append(end - start)

    def on_validation_epoch_end(self):
        psnr_val = np.mean(self.psnr_vals)
        ssim_val = np.mean(self.ssim_vals)
        lpips_val = np.mean(self.lpips_vals)
        render_time = np.mean(self.render_time)

        self.log("val/psnr", psnr_val, sync_dist=True, batch_size=self.batch_size)
        self.log("val/ssim", ssim_val, sync_dist=True, batch_size=self.batch_size)
        self.log("val/lpips", lpips_val, sync_dist=True, batch_size=self.batch_size)

        row = [
            self.exp_name,
            self.global_step,
            psnr_val,
            ssim_val,
            lpips_val,
            render_time,
        ]
        csv_path = os.path.join(self.val_results_dir, "val_results.csv")

        if os.path.exists(csv_path):
            cprint(f"Saving results to {csv_path}")
            write_csv(csv_path, row, include_header=False)
        else:
            write_csv(csv_path, row, include_header=True)

        val_image_dir = os.path.join(self.val_results_dir, "images")
        os.makedirs(val_image_dir, exist_ok=True)
        for idx, val_image in enumerate(self.val_images):
            name = f"{self.global_step}_{idx}.png"
            path = os.path.join(val_image_dir, name)
            img = Image.fromarray(val_image)
            img.save(path)

    def on_test_epoch_start(self):
        name = "eval_results"
        self.test_results_dir = os.path.join(self.result_dir, name)
        os.makedirs(self.test_results_dir, exist_ok=True)
        self.test_images = []

    def test_step(self, batch, batch_idx):
        evaluate = False
        if evaluate:
            with open(f"./val_split.json", "r") as f:
                infos = json.load(f)
            self.psnr_vals = []
            self.ssim_vals = []
            self.lpips_vals = []
            self.render_time = []

            for info in tqdm(infos):
                action, frame_id, cam_name = info
                batch = self.test_data.fetch_data_by_frame(action, frame_id, cam_name)
                batch = to_tensor(batch, device=self.device)
                start = time.time()
                pred = self.render(batch)
                end = time.time()
                gt = batch["rgb"]
                if len(gt.shape) == 4:
                    gt = gt[0]

                render = pred["render"] * batch["mask"][0]
                gt = gt * batch["mask"][0]

                psnr_val = psnr(render, gt)
                ssim_val = ssim(render, gt)
                lpips_val = torch.mean(
                    lpips_loss(render, gt.unsqueeze(0), lpips_evalfn)
                )
                self.psnr_vals.append(to_numpy(psnr_val))
                self.ssim_vals.append(to_numpy(ssim_val))
                self.lpips_vals.append(to_numpy(lpips_val))
                self.render_time.append(end - start)

                # print(f"psnr: {psnr_val}, ssim: {ssim_val}, lpips: {lpips_val}, render_time: {end - start}")

            metric_dict = {}
            metric_dict["psnr"] = np.asarray(self.psnr_vals).tolist()
            metric_dict["ssim"] = np.asarray(self.ssim_vals).tolist()
            metric_dict["lpips"] = np.asarray(self.lpips_vals).tolist()
            metric_dict["render_time"] = np.asarray(self.render_time).tolist()

            with open("./metrics.json", "w") as f:
                json.dump(metric_dict, f)

            print("Evaluation Completed !!")
            exit(0)

        pred = self.render(batch)
        img = to_numpy(torch.clamp(pred["render"], 0, 1))
        img = (img * 255).astype(np.uint8)
        if self.test_on_train_dataset:
            gt_img = dump_image(batch["rgb"][0], return_img=True)
            diff = (gt_img / 255.0 - img / 255.0) ** 2
            diff = diff * 255.0
            img = concat_img_array(img, gt_img)
            img = concat_img_array(img, diff)

        self.test_images.append(img)

        try:
            if self.model.get_skin_weights() is not None:
                colors = visualize_skin_weights(self.model.get_skin_weights())
                if colors.shape[0] != pred["posed_xyz"].shape[0]:
                    final_colors = np.zeros((pred["posed_xyz"].shape[0], 4))
                    final_colors[..., -1] = 1
                    final_colors[: colors.shape[0]] = colors
                else:
                    final_colors = colors
                self.dump_gaussians(pred, batch_idx, final_colors)
        except:
            self.dump_gaussians(pred, batch_idx) if batch_idx == 0 else None

        self.dump_gaussians(pred, batch_idx)

    def dump_gaussians(self, pred, batch_idx, results_dir=None, colors=None):
        if self.mode == "train":
            if results_dir is None:
                results_dir = self.val_results_dir
            else:
                results_dir = results_dir
        else:
            if results_dir is None:
                results_dir = self.test_results_dir
            else:
                results_dir = results_dir

        idx = batch_idx
        results_dir = os.path.join(results_dir, "gaussians")
        os.makedirs(results_dir, exist_ok=True)
        dump_points(
            pred["posed_xyz"],
            os.path.join(results_dir, f"{self.global_step}_{idx}_posed.ply"),
            colors,
        )

    def on_test_epoch_end(self):
        if "wandb" in self.opts.trainer.loggers:
            self.logger.log_image(key=f"test/result", images=self.test_images)
        else:
            if self.test_on_train_dataset:
                dir_name = "test_train"
            elif self.opts.test_dataset.opts.test_on_canonical_pose:
                dir_name = "test_cano"
            else:
                dir_name = "test_novel"

            # test_image_dir = os.path.join(self.test_results_dir, dir_name)
            # os.makedirs(test_image_dir, exist_ok=True)
            dump_video(
                self.test_images, os.path.join(self.test_results_dir, f"{dir_name}.mp4")
            )
            # for idx, test_image in tqdm(enumerate(self.test_images)):
            #     dump_image(test_image, os.path.join(test_image_dir, f'{idx}.png'))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        return self.model.optimizer

    def loss_func(self, batch, pred, losses_dict, loss_weight, log_losses=True):
        mask = batch["mask"][0]
        gt_image = batch["rgb"][..., :3]
        pred_image = pred["render"]

        losses = edict({})
        if "rgb_loss" in losses_dict:
            loss = l1_loss(pred_image, gt_image, mean=False)
            losses.rgb_loss = torch.mean(loss)

        if "lpips_loss" in losses_dict:
            if self.global_step < self.model.opts.start_lpips_iter:
                losses.lpips_loss = 0.0
            else:
                pred = pred_image[None].permute(0, 3, 1, 2)
                ref = gt_image.permute(0, 3, 1, 2)
                self.loss_fn_vgg = self.loss_fn_vgg.to(pred.device)
                dist = self.loss_fn_vgg.forward(pred, ref)
                losses.lpips_loss = dist.mean()

        if "l2_loss" in losses_dict:
            losses.l2_loss = l2_loss(pred_image, gt_image, mean=True)

        if "ssim_loss" in losses_dict:
            losses.ssim_loss = 1.0 - ssim(pred_image, gt_image)

        if "isotropic_reg" in losses_dict:
            max_scale = torch.max(self.model.get_scaling, dim=1)[0]
            min_scale = torch.min(self.model.get_scaling, dim=1)[0]
            isotropic_loss = torch.mean(
                (min_scale / (max_scale + 1e-8) - self.model.opts.condition_number) ** 2
            )

            losses.isotropic_reg = isotropic_loss

        final_loss = 0
        for name, loss in losses.items():
            if log_losses:
                self.log(name, loss, sync_dist=True, prog_bar=True)
            loss_index = losses_dict.index(name)
            weight = loss_weight[loss_index]
            final_loss += weight * loss
        return final_loss
