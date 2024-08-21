from datasets import LizardMaskRCNN
from losses import fastrcnn_loss
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection import roi_heads
from tqdm import tqdm
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

import transforms as T
import numpy as np
import pandas as pd
import argparse
import torch
import os
import time
import torchvision

import json


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=96)

    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument(
        "--store_predictions", dest="store_predictions", action="store_true"
    )
    parser.add_argument(
        "--coco_evaluation_mode",
        dest="coco_evaluation_mode", action="store_true"
    )
    parser.add_argument("--gpu", dest="gpu", action="store_true")

    parser.add_argument(
        "--out", type=str, default="mask-rcnn-resnet50-base-fpn-hist-match-glas"
    )

    parser.add_argument("--source", type=str, default="glas")

    parser.set_defaults(
        coco_evaluation_mode=True,
        store_predictions=True,
        gpu=True,
    )

    args = parser.parse_args()
    return args


def _get_iou_types(model):

    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(
        model_without_ddp,
        torchvision.models.detection.KeypointRCNN
    ):
        iou_types.append("keypoints")
    return iou_types


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # backbone  = resnet_fpn_backbone("resnet101", pretrained=True)
    # model.backbone = backbone

    # replace the anchor generator with one adapted for smaller cell sizes:

    # create an anchor_generator for the FPN which by default has 5 outputs
    conic_anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))

    model.rpn.anchor_generator = conic_anchor_generator

    # 256 because that's the number of features that FPN returns
    model.rpn.head = RPNHead(
        256, conic_anchor_generator.num_anchors_per_location()[0])

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # this is dense object detection problem so,
    # increase the number of detection per image

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    model.roi_heads.detections_per_img = 450
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def main(
        num_classes,
        store_predictions,
        coco_evaluation_mode,
        out,
        gpu,
        epoch,
        source
):
    roi_heads.fastrcnn_loss = fastrcnn_loss

    model = get_model_instance_segmentation(num_classes)

    device = "cuda" if torch.cuda.is_available() and gpu else "cpu"

    model.load_state_dict(
        torch.load(
            os.path.join(
                os.getcwd(),
                "results",
                out,
                "model_checkpoints",
                out + "-epoch" + str(epoch) + ".pt",
            ),
            map_location=torch.device(device),
        )
    )

    model.eval()
    model.to(device)

    file_names = pd.read_csv(os.path.join("./data", "patch_info.csv"))[
        "patch_info"
    ].to_list()

    valid_ids = [f for f in file_names if source in f]

    # with open("data-splits/crag/test/crag-test-fold8.json", "r") as f:
    #    valid_ids = json.load(f)

    test_dataset = LizardMaskRCNN(
        valid_ids,
        root_dir="./data",
        transforms=T.Compose(
            [
                T.ToTensor(),
            ]
        ),
    )

    data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    pred = []

    if coco_evaluation_mode:
        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        coco_evaluator.coco_eval["bbox"].params.maxDets = [1, 10, 450]
        coco_evaluator.coco_eval["segm"].params.maxDets = [1, 10, 450]

    for step, (images, targets) in enumerate(tqdm(data_loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        if coco_evaluation_mode:

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = {
                target["image_id"].item(): output
                for target, output in zip(targets, outputs)
            }
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time

        outputs = [{k: v.to(torch.device(
            "cpu")) for k, v in t.items()} for t in outputs]

        if store_predictions:

            for j, output in enumerate(outputs):
                pred_bboxes = output["boxes"]
                pred_labels = output["labels"]
                pred_scores = output["scores"]
                pred_masks = output["masks"]
                boxes, labels, scores, masks = [], [], [], []
                for index in range(len(pred_scores)):
                    if pred_scores[index] >= 0.45:
                        boxes.append(pred_bboxes[index])
                        labels.append(pred_labels[index])
                        scores.append(pred_scores[index])
                        masks.append(pred_masks[index])
                if len(boxes) != 0:
                    predictions = {
                        "boxes": torch.stack(boxes),
                        "labels": torch.stack(labels),
                        "scores": torch.stack(scores),
                        "masks": torch.stack(masks),
                    }
                else:
                    predictions = {
                        "boxes": torch.stack([torch.zeros(4)]),
                        "labels": torch.stack([torch.zeros(1)]),
                        "scores": torch.stack([torch.zeros(1)]),
                        "masks": torch.stack([torch.zeros((256, 256))]),
                    }

                classification = np.zeros((256, 256))
                segmentation = np.zeros((256, 256))
                for i, mask in enumerate(predictions["masks"]):
                    bin = torch.where(mask > 0.5, 1, 0)
                    # no overlap
                    if (
                        np.sum((
                            bin * predictions["labels"][i]).squeeze(
                        ).detach().numpy() * classification) == 0
                    ):

                        classification += ((
                            bin * predictions["labels"][i]
                        ).squeeze().detach().numpy()
                        )

                        segmentation += (bin * i).squeeze().detach().numpy()
                pred.append([segmentation, classification])

    if coco_evaluation_mode:
        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    if store_predictions:

        if not os.path.exists(
            os.path.join(
                os.getcwd(),
                "results",
                out,
                "predictions")):

            os.mkdir(os.path.join(os.getcwd(), "results", out, "predictions"))

        pred = np.moveaxis(np.array(pred), 1, -1)
        np.save(
            os.path.join(
                os.getcwd(),
                "results",
                out,
                "predictions",
                "pred-epoch" + str(epoch) + ".npy",
            ),
            pred,
        )

    pq_outputs = (
        os.popen(
            "python3 compute_stats.py --mode seg_class --pred " + os.path.join(
                os.getcwd(), "results", out, "predictions",
                "pred-epoch" + str(epoch) + ".npy",) + " --true "
            "/home/jdnunes/conic-survey/targets/"
            "hold-out-val/true-" + source + ".npy"
            " --out " + os.path.join(
                os.getcwd(),
                "results",
                out,
                "metrics",
                "tile-level-mpq.npy",)
        )
        .read()
        .split("\n")
    )

    class_wise_pq = pq_outputs[0].replace("'", '"')
    class_wise_pq = json.loads(class_wise_pq)
    metrics = pq_outputs[1].strip().split(" ")
    values = pq_outputs[2].strip().split(" ")
    while "" in values:
        values.remove("")
    while "" in metrics:
        metrics.remove("")
    values = [round(float(v), 4) for v in values]
    metrics = dict(zip(metrics, values))
    metrics = {**class_wise_pq, **metrics}

    map50_bbox = round(coco_evaluator.coco_eval["bbox"].stats[1], 4)
    map50_segm = round(coco_evaluator.coco_eval["segm"].stats[1], 4)
    map75_bbox = round(coco_evaluator.coco_eval["bbox"].stats[2], 4)
    map75_segm = round(coco_evaluator.coco_eval["segm"].stats[2], 4)

    map_dict = {
        "mAP50_bbox": map50_bbox,
        "mAP50_segm": map50_segm,
        "mAP75_bbox": map75_bbox,
        "mAP75_segm": map75_segm,
    }

    metrics = {
        "model": out,
        "backbone": "resnet-50-fpn",
        "test set centre": source,
        **map_dict,
        **metrics,
        **class_wise_pq,
    }

    # save results to registry file
    if os.path.exists("conic-dataset-results.xlsx"):
        df = pd.DataFrame(
            pd.read_excel(
                "conic-dataset-results.xlsx",
                sheet_name="Results",
                engine="openpyxl"
            )
        )

    df.loc[len(df)] = list(metrics.values())
    df.to_excel(
        r"conic-dataset-results.xlsx", sheet_name="Results", index=False)


if __name__ == "__main__":

    args = parse_args()
    num_classes = args.num_classes
    store_predictions = args.store_predictions
    coco_evaluation_mode = args.coco_evaluation_mode
    out = args.out
    gpu = args.gpu
    epoch = args.epoch
    source = args.source

    main(
        num_classes,
        store_predictions,
        coco_evaluation_mode,
        out,
        gpu,
        epoch,
        source
    )
