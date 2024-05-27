import os
from natsort import natsorted
import sys
import glob
import sklearn
from PIL import Image
import scipy
import cv2
import numpy as np
from sklearn.metrics import jaccard_score

import csv

sys.path.append(os.getcwd())
import argparse

import taichi as ti

ti.init(arch=ti.cuda, device_memory_fraction=0.5)
# import warnings
# warnings.filterwarnings('ignore')

colors = np.asarray([
    [255, 255, 255],
    [43, 159, 43],
    [31, 119, 178],
    [173, 198, 231],
    [254, 186, 119],
    [151, 222, 137],
    [213, 38, 39],
    [254, 151, 149],
    [196, 175, 212],
    [139, 85, 74],
    [195, 155, 147],
    [246, 181, 209],
    [126, 126, 126],
    [198, 199, 198],
    [218, 218, 140],
    [25, 190, 206],
    [156, 217, 228]
]).astype(np.float32)


def get_contact_dist(pt1, pt2):
    ti_pt1 = ti.ndarray(shape=(pt1.shape[0], 2), dtype=ti.f32)
    ti_pt2 = ti.ndarray(shape=(pt2.shape[0], 2), dtype=ti.f32)
    ti_contact_map = ti.ndarray(shape=pt1.shape[0], dtype=ti.f32)
    ti_contact_indices = ti.ndarray(shape=pt1.shape[0], dtype=ti.f32)

    ti_pt1.from_numpy(pt1)
    ti_pt2.from_numpy(pt2)

    @ti.kernel
    def calculate_distances(ti_pt1: ti.types.ndarray(ndim=2),
                            ti_pt2: ti.types.ndarray(ndim=2),
                            ti_contact_map: ti.types.ndarray(ndim=1),
                            ti_contact_indices: ti.types.ndarray(ndim=1)):
        for i in range(pt1.shape[0]):
            min_dist = 1e9
            for j in range(pt2.shape[0]):
                dist = 0.0
                for k in range(2):
                    dist += (ti_pt1[i, k] - ti_pt2[j, k]) ** 2
                dist = ti.sqrt(dist)
                if dist < min_dist:
                    min_dist = dist
                    ti_contact_indices[i] = j
            ti_contact_map[i] = min_dist

    calculate_distances(ti_pt1, ti_pt2, ti_contact_map, ti_contact_indices)
    return ti_contact_map.to_numpy(), ti_contact_indices.to_numpy()


def get_skin_mask(img, gt_mask):
    # colors = np.asarray([
    #     [76, 151, 61],
    #     [58, 124, 185],
    #     [185, 204, 231],
    #     [246, 199, 136],
    #     [168, 214, 152],
    #     [213, 78, 54],
    #     [243, 175, 160],
    #     [201, 189, 217],
    #     [146, 103, 86],
    #     [200, 170, 158],
    #     [243, 199, 213],
    #     [137, 139, 136],
    #     [203, 205, 203],
    #     [224, 220, 155],
    #     [82, 180, 210],
    #     [172, 216, 227]
    # ]).astype(np.float32)

    colors = np.asarray([
        [43, 159, 43],
        [31, 119, 178],
        [173, 198, 231],
        [254, 186, 119],
        [151, 222, 137],
        [213, 38, 39],
        [254, 151, 149],
        [196, 175, 212],
        [139, 85, 74],
        [195, 155, 147],
        [246, 181, 209],
        [126, 126, 126],
        [198, 199, 198],
        [218, 218, 140],
        [25, 190, 206],
        [156, 217, 228]
    ]).astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    all_masks = []
    bkgd = np.zeros_like(gt_mask)
    all_masks.append(bkgd)
    for i in range(colors.shape[0]):
        offset = np.array([10, 10, 10])
        lower = colors[i] - offset
        upper = colors[i] + offset
        lower = (lower[0], lower[1], lower[2])
        upper = (upper[0], upper[1], upper[2])
        mask = cv2.inRange(img, lower, upper)
        erode_mask = cv2.erode(mask, kernel, iterations=1)
        final_mask = cv2.dilate(erode_mask, kernel, iterations=1)
        all_masks.append(final_mask)

    all_masks = np.stack(all_masks, axis=-1)
    all_masks = np.argmax(all_masks, axis=-1)

    # mask = ((all_masks > 0)*255).astype(np.uint8)
    # from PIL import Image
    # img = Image.fromarray(mask)
    # img.save('./test.png')
    all_masks = all_masks * gt_mask
    residual = np.logical_xor(gt_mask > 0, all_masks > 0)
    res_coord = np.argwhere(residual)
    skin_coord = np.argwhere(all_masks > 0)

    dist, indices = get_contact_dist(res_coord, skin_coord)
    indices = indices.astype(np.int32)
    idx_skin_coord = skin_coord[indices]
    final_mask = all_masks.copy()
    final_mask[res_coord[:, 0], res_coord[:, 1]] = all_masks[idx_skin_coord[:, 0], idx_skin_coord[:, 1]]

    ## Get color of the skin
    # img = Image.fromarray(((final_mask > 0) * 255).astype(np.uint8))
    # img.save('./test.png')
    # breakpoint()

    return final_mask


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--object_exp_name', type=str, required=True)
    parser.add_argument('--grasp_path', type=str, required=True)
    return parser


def cal_iou(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection + 1e-6)
    return iou


def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    dice = np.mean(2 * intersect / total_sum)
    return round(dice, 5)  # round up to 3 decimal places


def precision_score(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)
    precision = np.mean(intersect / total_pixel_pred)
    return round(precision, 5)


def recall_score(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    recall = np.mean(intersect / total_pixel_truth)
    return round(recall, 5)


def calculate_per_bone_iou(skin_mask, gt_mask, pred_mask):
    pred_mask_list = []
    iou_list = []
    gt_mask_list = []
    f1_list = []

    for i in range(16):
        local_mask = (skin_mask == i)
        pred_mask_list.append(pred_mask * local_mask)
        gt_mask_list.append(gt_mask * local_mask)
        iou_score = cal_iou(gt_mask * local_mask, pred_mask * local_mask)

        true = (gt_mask * local_mask) * 255
        pred = (pred_mask * local_mask) * 255
        f1_score = sklearn.metrics.f1_score(true.ravel(), pred.ravel(), zero_division=np.nan)

        f1_list.append(f1_score)
        iou_list.append(iou_score)

    pred_mask_list = np.stack(pred_mask_list, axis=-1)
    pred_mask_list = np.argmax(pred_mask_list, axis=-1)
    pred_mask = colors[pred_mask_list].astype(np.uint8)

    gt_mask_list = np.stack(gt_mask_list, axis=-1)
    gt_mask_list = np.argmax(gt_mask_list, axis=-1)
    gt_mask = colors[gt_mask_list].astype(np.uint8)

    return iou_list, f1_list, pred_mask, gt_mask


def evaluate_metric(skin_mask, gt_mask, pred_mask):
    ## Acc calculation
    iou_score_acc = cal_iou(gt_mask, pred_mask)
    true = gt_mask * 255
    pred = pred_mask * 255
    f1_score_acc = sklearn.metrics.f1_score(true.ravel(), pred.ravel(), zero_division=np.nan)

    ## Per Bone Calculation
    iou_list, f1_list, _, _ = calculate_per_bone_iou(skin_mask, gt_mask, pred_mask)
    return np.asarray(iou_list), np.asarray(f1_list), iou_score_acc, f1_score_acc

    # gt_mask = gt_mask[..., None].repeat(3, axis=-1)
    # pred_mask = pred_mask[..., None].repeat(3, axis=-1)
    # ## Blend images
    # green = np.array([0, 1, 0])
    # white = np.array([1, 1, 1])
    # alpha = (gt_rgb[..., -1:] > 128) * 1
    # rgb = gt_rgb[..., :3] / 255.0
    # gt_mask = gt_mask / 255.0
    # our_mask = our_mask / 255.0
    # mano_mask = mano_mask / 255.0
    #
    # final_rgb = (rgb * alpha) + white * (1 - alpha)
    # final_rgb = (final_rgb * 255).astype(np.uint8)
    #
    # gt_mask = gt_mask * white * 0.7 + 0.3 * final_rgb
    # our_mask = our_mask * white * 0.7 + 0.3 * final_rgb
    # mano_mask = mano_mask * white * 0.7 + 0.3 * final_rgb
    #
    # pred_mask = cv2.bitwise_and(pred_mask, gt_mask)
    #
    # # gt_mask = (gt_mask * 255).astype(np.uint8)
    # # our_mask = (our_mask * 255).astype(np.uint8)
    # # mano_mask = (mano_mask * 255).astype(np.uint8)
    # # cv2.putText(our_mask, f"{our_iou_score:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
    # # cv2.putText(mano_mask, f"{mano_iou_score:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
    #
    # collage = np.concatenate([final_rgb, gt_mask, our_mask, mano_mask], axis=1)
    # row1 = [*our_iou_list, our_iou_score_acc]
    # row2 = [*mano_iou_list, mano_iou_score_acc]
    # row3 = [*our_f1_list, our_f1_score_acc]
    # row4 = [*mano_f1_list, mano_f1_score_acc]
    # return row1, row2, row3, row4, collage
    #


def blend_masks(rgb, alpha, mask, weight=0.5):
    # color = np.asarray([33, 171, 205]) / 255
    color = np.asarray([0, 128, 0]) / 255
    mask = mask[..., None].repeat(3, axis=-1)
    mask = mask * color
    final = rgb * weight + (1 - weight) * mask
    final = final * alpha + (1 - alpha) * np.array([1, 1, 1])
    return final


def combine_images(rgb, gt_mask, our_mask, mano_mask, harp_mask):
    alpha = rgb[..., -1:] > 128
    rgb = rgb[..., :3] / 255
    our_mask = our_mask / 255
    mano_mask = mano_mask / 255
    harp_mask = harp_mask / 255
    gt_mask = gt_mask / 255

    final_o = blend_masks(rgb, alpha, our_mask)
    final_m = blend_masks(rgb, alpha, mano_mask)
    final_h = blend_masks(rgb, alpha, harp_mask)
    final_gt = blend_masks(rgb, alpha, gt_mask)
    rgb = rgb * alpha + (1 - alpha) * np.array([1, 1, 1])

    row = np.concatenate([rgb, final_gt, final_m, final_h, final_o], axis=1)
    row = row * 255
    return row


def main():
    args = get_parser().parse_args()
    root_dir = '/'.join(args.grasp_path.split('/')[:-3])
    gt_contact_dir = os.path.join(root_dir, "evals", f'{args.object_exp_name}_action', "gt_contacts_seg")

    gt_mask_path = natsorted(glob.glob(os.path.join(gt_contact_dir, '*.png')))
    gt_img_path = natsorted(glob.glob(os.path.join(gt_contact_dir.replace("gt_contacts_seg", "gt_contacts"), '*.png')))
    our_mask_path = natsorted(glob.glob(os.path.join(args.exp_dir, 'results/eval_results/ours/acc_gt_eval/', '*.png')))
    mano_mask_path = natsorted(
        glob.glob(os.path.join(args.exp_dir, 'results/eval_results/mano/acc_eval_rendered/', '*.png')))

    harp_mask_path = natsorted(
        glob.glob(os.path.join(args.exp_dir, 'results/eval_results/harp/acc_eval_rendered/', '*.png')))

    collage = []
    row_o = []
    row_m = []
    row_h = []
    row_of = []
    row_mf = []
    row_hf = []

    for i in range(len(gt_mask_path)):
        print("-----------------------------")
        print("gt_mask_path: ", gt_mask_path[i])
        print("our_mask_path: ", our_mask_path[i])
        print("mano_mask_path: ", mano_mask_path[i])
        print("harp_mask_path: ", harp_mask_path[i])
        print("-----------------------------")

        gt_rgb = cv2.imread(gt_img_path[i], cv2.IMREAD_UNCHANGED)
        gt_mask = cv2.imread(gt_mask_path[i])
        gt_mask = cv2.inRange(gt_mask, (128, 128, 128), (255, 255, 255))

        our_mask = cv2.imread(our_mask_path[i])
        skin_img = our_mask[:, :1080, :][..., ::-1]
        our_mask = our_mask[:, 1080:, :]
        our_mask = cv2.inRange(our_mask, (128, 128, 128), (255, 255, 255))

        mano_mask = cv2.imread(mano_mask_path[i])
        mano_mask = cv2.inRange(mano_mask, (128, 128, 128), (255, 255, 255))

        harp_mask = cv2.imread(harp_mask_path[i])
        harp_mask = cv2.inRange(harp_mask, (128, 128, 128), (255, 255, 255))

        skin_mask = get_skin_mask(skin_img, gt_rgb[..., -1] > 128)

        iou_o, f1_o, iou_acc_o, f1_acc_o = evaluate_metric(skin_mask, gt_mask, our_mask)
        iou_m, f1_m, iou_acc_m, f1_acc_m = evaluate_metric(skin_mask, gt_mask, mano_mask)
        iou_h, f1_h, iou_acc_h, f1_acc_h = evaluate_metric(skin_mask, gt_mask, harp_mask)
        img_row = combine_images(gt_rgb, gt_mask, our_mask, mano_mask, harp_mask)
        collage.append(img_row)
        row_m.append([*iou_m, iou_acc_m])
        row_o.append([*iou_o, iou_acc_o])
        row_h.append([*iou_h, iou_acc_h])
        row_mf.append([*f1_m, f1_acc_m])
        row_of.append([*f1_o, f1_acc_o])
        row_hf.append([*f1_h, f1_acc_h])

    collages = np.vstack(collage).astype(np.uint8)
    out_path = os.path.join(args.exp_dir, 'results/eval_results/eval_collage.png')
    cv2.imwrite(out_path, collages)

    row_o = np.around(np.vstack(row_o).mean(axis=0), decimals=3)
    row_m = np.around(np.vstack(row_m).mean(axis=0), decimals=3)
    row_h = np.around(np.vstack(row_h).mean(axis=0), decimals=3)
    row_of = np.around(np.vstack(row_of).mean(axis=0), decimals=3)
    row_mf = np.around(np.vstack(row_mf).mean(axis=0), decimals=3)
    row_hf = np.around(np.vstack(row_hf).mean(axis=0), decimals=3)

    with open(os.path.join(args.exp_dir, 'results/eval_results/eval_metric.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["", "bone1", "bone2", "bone3", "bone4", "bone5", "bone6", "bone7", "bone8", "bone9", "bone10", "bone11",
             "bone12", "bone13", "bone14", "bone15", "bone16", "combined"])

        writer.writerow(["ours"] + row_o.tolist())
        writer.writerow(["mano"] + row_m.tolist())
        writer.writerow(["harp"] + row_h.tolist())
        writer.writerow(["ours_f1"] + row_of.tolist())
        writer.writerow(["mano_f1"] + row_mf.tolist())
        writer.writerow(["harp_f1"] + row_hf.tolist())


if __name__ == '__main__':
    main()
