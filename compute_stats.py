"""compute_stats.py. Calculates the statistical measurements for the CoNIC Challenge.
This code supports binary panoptic quality for binary segmentation, multiclass panoptic quality for
simultaneous segmentation and classification and multiclass coefficient of determination (R2) for
multiclass regression. Binary panoptic quality is calculated per image and the results are averaged.
For multiclass panoptic quality, stats are calculated over the entire dataset for each class before taking
the average over the classes.
Usage:
    compute_stats.py [--mode=<str>] [--pred=<path>] [--true=<path>] [--out=<path>] [--stats=<int>]
    compute_stats.py (-h | --help)
    compute_stats.py --version
Options:
    -h --help                   Show this string.
    --version                   Show version.
    --mode=<str>                Choose either `regression` or `seg_class`.
    --pred=<path>               Path to the results directory.
    --true=<path>               Path to the ground truth directory.
    --out=<path>                Path to write the tile-level multipq metrics.
    --stats=<int>               Perform statistical significance tests.
"""

from docopt import docopt
import numpy as np
import os
import pandas as pd
import json
from tqdm.auto import tqdm

from metrics.stats_utils import get_pq, get_multi_pq_info, get_multi_r2


if __name__ == "__main__":
    args = docopt(__doc__, version="CoNIC-stats-v1.0")

    mode = args["--mode"]
    pred_path =  args["--pred"]
    true_path =  args["--true"]
    out = args["--out"]
    stats = args["--stats"]

    seg_metrics_names = ["pq", "multi_pq+"]
    reg_metrics_names = ["r2"]

    # do initial checks
    if mode not in ["regression", "seg_class"]:
        raise ValueError("`mode` must be either `regression` or `seg_class`")

    all_metrics = {}
    if mode == "seg_class":
        # check to make sure input is a single numpy array
        pred_format = pred_path.split(".")[-1]
        true_format = true_path.split(".")[-1]
        if pred_format != "npy" or true_format != "npy":
            raise ValueError("pred and true must be in npy format.")

        # initialise empty placeholder lists
        pq_list = []
        pq_det_list = []
        pq_seg_list = []
        mpq_info_list = []
        # load the prediction and ground truth arrays
        pred_array = np.load(pred_path)
        true_array = np.load(true_path)

        nr_patches = pred_array.shape[0]

        for patch_idx in tqdm(range(nr_patches)):
            # get a single patch
            pred = pred_array[patch_idx]
            true = true_array[patch_idx]

            # instance segmentation map
            pred_inst = pred[..., 0]
            true_inst = true[..., 0]
            # classification map
            pred_class = pred[..., 1]
            true_class = true[..., 1]

            # ===============================================================

            for idx, metric in enumerate(seg_metrics_names):
                if metric == "pq":
                    # get binary panoptic quality
                    pq = get_pq(true_inst, pred_inst)
                    pq_list.append(pq[0][2])
                    pq_det_list.append(pq[0][0])
                    pq_seg_list.append(pq[0][1])
                elif metric == "multi_pq+":
                    # get the multiclass pq stats info from single image
                    mpq_info_single = get_multi_pq_info(true, pred)
                    mpq_info = []
                    # aggregate the stat info per class
                    for single_class_pq in mpq_info_single:
                        tp = single_class_pq[0]
                        fp = single_class_pq[1]
                        fn = single_class_pq[2]
                        sum_iou = single_class_pq[3]
                        mpq_info.append([tp, fp, fn, sum_iou])
                    mpq_info_list.append(mpq_info)
                else:
                    raise ValueError("%s is not supported!" % metric)

        pq_metrics = np.array(pq_list)
        np.save(os.path.join(os.path.dirname(out),
                             "pq_metrics.npy"
                             ), pq_metrics)
        pq_det_metrics = np.array(pq_det_list)
        np.save(os.path.join(os.path.dirname(out),
                             "pq_det_metrics.npy"
                             ), pq_det_metrics)
        pq_seg_metrics = np.array(pq_seg_list)
        np.save(os.path.join(os.path.dirname(out),
                             "pq_seg_metrics.npy"
                             ), pq_seg_metrics)
        pq_metrics_avg = np.mean(pq_metrics, axis=-1)  # average over all images
        pq_det_metrics_avg = np.mean(pq_det_metrics, axis=-1)
        pq_seg_metrics_avg = np.mean(pq_seg_metrics, axis=-1)
        if "multi_pq+" in seg_metrics_names:

            mpq_info_metrics = np.array(mpq_info_list, dtype="float")
            # save sample-specific multi_pq+ info
            # np.save(os.path.join(os.path.dirname(pred_path), "multi_pq_info.npy"), mpq_info_metrics)
            # sum over all the images
            total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)

        # patch-level multipq for statistical significance tests

        tiles_mpq_list = []
        # for each class, get the multiclass PQ
        for cat_idx in range(mpq_info_metrics.shape[1]):
            tp = mpq_info_metrics[:, cat_idx][:, 0]
            fp = mpq_info_metrics[:, cat_idx][:, 1]
            fn = mpq_info_metrics[:, cat_idx][:, 2]
            sum_iou = mpq_info_metrics[:, cat_idx][:, 3]

            # get the F1-score i.e DQ
            dq = tp / (
                (tp + 0.5 * fp + 0.5 * fn) + 1.0e-6
            )
            # get the SQ, when not paired, it has 0 IoU so does not impact
            sq = sum_iou / (tp + 1.0e-6)
            tiles_mpq_list.append(dq * sq)
        tiles_mpq_metrics = np.stack(tiles_mpq_list, axis=-1)
        np.save(os.path.join(os.path.dirname(out),
                             "tiles_mpq_metrics.npy"
                             ), tiles_mpq_metrics)

        for idx, metric in enumerate(seg_metrics_names):
            if metric == "multi_pq+":
                mpq_list = []
                # for each class, get the multiclass PQ
                for cat_idx in range(total_mpq_info_metrics.shape[0]):
                    total_tp = total_mpq_info_metrics[cat_idx][0]
                    total_fp = total_mpq_info_metrics[cat_idx][1]
                    total_fn = total_mpq_info_metrics[cat_idx][2]
                    total_sum_iou = total_mpq_info_metrics[cat_idx][3]

                    # get the F1-score i.e DQ
                    dq = total_tp / (
                        (total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6
                    )
                    # get the SQ, when not paired, it has 0 IoU so does not impact
                    sq = total_sum_iou / (total_tp + 1.0e-6)
                    mpq_list.append(dq * sq)
                mpq_metrics = np.array(mpq_list)
                np.save(os.path.join(os.path.dirname(out),
                             "mpq_metrics.npy"
                             ), mpq_metrics)
                nuclei_labels = ["neutrophil", "epithelial", "lymphocite", "plasma", "eosinophil", "connective"]
                mpq_metrics_per_class = {nuclei_labels[i]: np.around(mpq_metrics[i], 4) for i in range(len(mpq_metrics))}

                all_metrics[metric] = [np.mean(mpq_metrics)]
            else:
                all_metrics[metric] = [pq_metrics_avg]
                all_metrics['pq_det'] = [pq_det_metrics_avg]
                all_metrics['pq_seg'] = [pq_seg_metrics_avg]

    else:
        # first check to make sure ground truth and prediction is in csv format
        if not os.path.isfile(true_path) or not os.path.isfile(pred_path):
            raise ValueError("pred and true must be in csv format.")

        pred_format = pred_path.split(".")[-1]
        true_format = true_path.split(".")[-1]
        if pred_format != "csv" or true_format != "csv":
            raise ValueError("pred and true must be in csv format.")

        pred_csv = pd.read_csv(pred_path)
        true_csv = pd.read_csv(true_path)

        for idx, metric in enumerate(reg_metrics_names):
            if metric == "r2":
                # calculate multiclass coefficient of determination
                r2 = get_multi_r2(true_csv, pred_csv)
                all_metrics["multi_r2"] = [r2]
            else:
                raise ValueError("%s is not supported!" % metric)

    df = pd.DataFrame(all_metrics)
    df = df.to_string(index=False)
    np.save(out, tiles_mpq_metrics)
    print(mpq_metrics_per_class)
    print(df)
