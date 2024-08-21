import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models.detection import roi_heads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from tqdm import tqdm

import transforms as T
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from datasets import LizardMaskRCNN
from losses import fastrcnn_loss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out", type=str,
        default="mask-rcnn-resnet50-base-fpn-hist-match-crag"
    )

    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--warmup_factor", type=float, default=1.0 / 1000)
    parser.add_argument("--gpu", dest="gpu", action="store_true")
    parser.add_argument("--set_alpha", dest="set_alpha", action="store_true")
    parser.add_argument("--source", type=str, default="crag")
    parser.set_defaults(gpu=True, set_alpha=True)

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
        model_without_ddp, torchvision.models.detection.KeypointRCNN
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
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]),
    )

    model.rpn.anchor_generator = conic_anchor_generator

    # 256 because that's the number of features that FPN returns
    model.rpn.head = RPNHead(
        256,
        conic_anchor_generator.num_anchors_per_location()[0])

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
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    model.roi_heads.detections_per_img = 450
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def main(
    out,
    source,
    num_classes,
    n_epochs,
    lr,
    momentum,
    weight_decay,
    warmup_factor,
    gpu,
):
    file_names = pd.read_csv(
        os.path.join(os.getcwd(), "data", "patch_info.csv"))[
        "patch_info"
    ].to_list()

    train_ids = [f for f in file_names if source not in f]

    train_ids, valid_ids = train_test_split(
        train_ids,
        test_size=0.1,
        random_state=42
    )

    train_dataset = LizardMaskRCNN(
        train_ids,
        root_dir=os.path.join(os.getcwd(), "data"),
        transforms=T.Compose(
            [
                T.ToTensor(),
                T.RandomHorizontalFlip(0.5),
            ]
        ),
    )

    valid_dataset = LizardMaskRCNN(
        valid_ids,
        root_dir=os.path.join(os.getcwd(), "data"),
        transforms=T.Compose(
            [
                T.ToTensor(),
            ]
        ),
    )

    data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        drop_last=True,
        num_workers=16,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=8,
        drop_last=True,
        num_workers=16,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    scaler = None

    plt.ion()
    loss_list = []
    loss_over_epochs = []
    iteration = 1

    roi_heads.fastrcnn_loss = fastrcnn_loss
    model = get_model_instance_segmentation(num_classes)

    device = "cuda" if torch.cuda.is_available() and gpu else "cpu"
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    if not os.path.exists(os.path.join(os.getcwd(), "results", out)):
        os.mkdir(os.path.join(os.getcwd(), "results", out))

    if not os.path.exists(
        os.path.join(
            os.getcwd(),
            "results",
            out.split(".")[0],
            "model_checkpoints")
    ):
        os.mkdir(
            os.path.join(os.getcwd(),
                         "results", out, "model_checkpoints"))

    if not os.path.exists(
        os.path.join(
            os.getcwd(), "results", out, "metrics")):
        os.mkdir(
            os.path.join(os.getcwd(), "results", out, "metrics"))

    with open(
        os.path.join(os.getcwd(), "results", out, "metrics", "train_eval.txt"
                     ), "w"
    ) as f:

        f.write("EPOCH AP50-bbox AP50-segm loss \n")

    scaler = None

    for epoch in range(n_epochs):
        loss_epoch = []
        print("EPOCH: ", str(epoch))
        if epoch == 0:

            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )
        if epoch == 1:
            del lr_scheduler
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=10,
                threshold=0.05,
                verbose=True,
            )

        for step, (images, targets) in enumerate(tqdm(data_loader)):

            model.train()
            images = list(image.to(device) for image in images)
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]

            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()

            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()
            if epoch == 0:
                lr_scheduler.step()
            if gpu:
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(
                    loss for loss in loss_dict_reduced.values())
                loss_value = losses_reduced.item()
            else:
                loss_value = losses.item()

            loss_list.append(loss_value)
            loss_epoch.append(loss_value)
            iteration += 1

        loss_epoch_mean = np.mean(loss_epoch)
        loss_over_epochs.append(loss_epoch_mean)

        print("Average loss for epoch = {:.4f} ".format(loss_epoch_mean))
        if epoch > 0:
            lr_scheduler.step(loss_epoch_mean)
        model.eval()

        coco = get_coco_api_from_dataset(valid_data_loader.dataset)
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        coco_evaluator.coco_eval["bbox"].params.maxDets = [1, 10, 450]
        coco_evaluator.coco_eval["segm"].params.maxDets = [1, 10, 450]
        for step, (images, targets) in enumerate(tqdm(valid_data_loader)):
            images = list(img.to(device) for img in images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outputs = model(images)

            outputs = [
                {k: v.to(
                    torch.device(
                        device)) for k, v in t.items()} for t in outputs
            ]
            res = {
                target["image_id"].item(): output
                for target, output in zip(targets, outputs)
            }
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time

        # evaluate
        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        map50_bbox = coco_evaluator.coco_eval["bbox"].stats[1]
        map50_segm = coco_evaluator.coco_eval["segm"].stats[1]

        if epoch % 4 == 0:
            # Save model
            torch.save(
                model.state_dict(),
                os.path.join(
                    os.getcwd(),
                    "results",
                    out,
                    "model_checkpoints",
                    out + "-" + "epoch" + str(epoch) + ".pt",
                ),
            )

        with open(
            os.path.join(
                os.getcwd(),
                "results",
                out,
                "metrics",
                "train_eval.txt"), "a+"
        ) as f:

            f.write(
                f"\n{epoch} {map50_bbox:.3f} {map50_segm:.3f} "
                f"{loss_epoch_mean:.3f}"
            )


if __name__ == "__main__":

    torch.manual_seed(42)

    args = parse_args()
    out = args.out
    num_classes = args.num_classes
    n_epochs = args.n_epochs
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    warmup_factor = args.warmup_factor
    gpu = args.gpu
    source = args.source

    main(
        out,
        source,
        num_classes,
        n_epochs,
        lr,
        momentum,
        weight_decay,
        warmup_factor,
        gpu,
    )
