from custom_roi_heads import RoIHeads
from relation_networks import RelationModule

from datasets import LizardMaskRCNN

from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.rpn import RPNHead
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

import pandas as pd
import numpy as np
import transforms as T
import torch
import os
import time
import argparse
import utils
import torchvision
import torch.nn.functional as F
import json
import torch.nn as nn

parser = argparse.ArgumentParser()

parser.add_argument(
    "--epoch",
    type=int,
    default=96)

parser.add_argument(
    "--out",
    type=str,
    default="mask-rcnn-resnet50-fpn-semantic-rm-crag")

parser.add_argument("--num_classes", type=int, default=7)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--set_alpha", dest="set_alpha", action="store_true")

parser.add_argument("--store_predictions", dest="store_predictions",
                    action="store_true")
parser.add_argument("--coco_evaluation_mode", dest="coco_evaluation_mode",
                    action="store_true")
parser.add_argument("--test_set_source",  type=str, default="crag")

parser.set_defaults(
    coco_evaluation_mode=True,
    store_predictions=True,
    gpu=True,)

args = parser.parse_args()

out = args.out
num_classes = args.num_classes
gpu = args.gpu
store_predictions = args.store_predictions
coco_evaluation_mode = args.coco_evaluation_mode
epoch = args.epoch
test_set_source = args.test_set_source

torch.manual_seed(42)

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


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(
        self,
        in_channels,
        representation_size,
        ):
        
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=representation_size,
            dim_feedforward=representation_size,
            activation=F.gelu,
            nhead=8,
            batch_first=True,
            dropout=0.5
            )
        self.semantic_rm =  nn.TransformerEncoder(encoder_layer, num_layers=1)
           
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc6(x))
        x = self.semantic_rm(x.unsqueeze(0))
        return x.squeeze(0)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # backbone  = resnet_fpn_backbone("resnet101", pretrained=True)
    #model.backbone = backbone

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
    in_channels = model.roi_heads.box_head.fc6.in_features

    model.roi_heads.box_head = TwoMLPHead(in_channels, 512)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(512, num_classes)
    # this is dense object detection problem so, increase the number of detection per image
    
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

# with open('data-splits/crag/train/crag-train-fold8.json', 'r') as f:
#    train_ids = json.load(f)

# with open('data-splits/crag/val/crag-val-fold8.json', 'r') as f:
#    valid_ids = json.load(f)

file_names = pd.read_csv(
    os.path.join("./data", "patch_info.csv"))["patch_info"].to_list()

valid_ids = [f for f in file_names if test_set_source in f]

test_dataset = LizardMaskRCNN(
    valid_ids,
    root_dir="./data",
    transforms=T.Compose([T.ToTensor(), ])
    )

data_loader = DataLoader(
    test_dataset,
    batch_size=2,
    num_workers=2,
    shuffle=False,
    collate_fn=collate_fn)

device = "cuda" if torch.cuda.is_available() and gpu else "cpu"

model = get_model_instance_segmentation(num_classes)
model.load_state_dict(
    torch.load(os.path.join(
        os.getcwd(),
        "results",
        out,
        "model_checkpoints",
        out+'-epoch'+str(epoch)+'.pt'),
        map_location=torch.device(device)))

model.eval()
model = model.to(device)

pred = []

if coco_evaluation_mode:
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    coco_evaluator.coco_eval['bbox'].params.maxDets = [1, 10, 450]
    coco_evaluator.coco_eval['segm'].params.maxDets = [1, 10, 450]


for step, (images, targets) in enumerate(tqdm(data_loader)):
    model_time = time.time()
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    outputs = model(images)
    
    if coco_evaluation_mode:
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = {target["image_id"].item(): output
                   for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        
    outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
    
    if store_predictions:
        
        for j, output in enumerate(outputs):
            pred_bboxes = output['boxes']
            pred_labels = output["labels"]
            pred_scores = output['scores']
            pred_masks = output['masks']
            boxes, labels, scores, masks = [], [], [], []
            for index in range(len(pred_scores)):
                if pred_scores[index] > 0.5:
                    boxes.append(pred_bboxes[index])
                    labels.append(pred_labels[index])
                    scores.append(pred_scores[index])
                    masks.append(pred_masks[index])
            if len(boxes) != 0:
                predictions = {
                    'boxes': torch.stack(boxes),
                    'labels': torch.stack(labels),
                    'scores': torch.stack(scores), 
                    'masks': torch.stack(masks)}
            else:
                    predictions = {
                    'boxes': torch.stack([torch.zeros(4)]),
                    'labels': torch.stack([torch.zeros(1)]),
                    'scores': torch.stack([torch.zeros(1)]), 
                    'masks': torch.stack([torch.zeros((256, 256))])
                    }
                
            classification = np.zeros((256, 256))
            segmentation = np.zeros((256, 256))
            for i, mask in enumerate(predictions["masks"]):
                bin = torch.where(mask > 0.5, 1, 0)
                # no overlap
                if np.sum(
                    (bin*predictions["labels"][i]).squeeze().detach().numpy()
                    * classification
                     ) == 0:

                    classification += (
                        bin * predictions["labels"][i]).squeeze().detach().numpy()

                    segmentation += (bin*i).squeeze().detach().numpy()
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
        "predictions")
    ):
        os.mkdir(os.path.join(
            os.getcwd(),
            "results",
            out,
            "predictions")
            )
    n = len(pred)
    pred = np.moveaxis(np.array(pred), 1, -1)
    np.save(os.path.join(
        os.getcwd(),
        "results",
        out,
        "predictions",
        'pred-epoch'+str(epoch)+'.npy'), pred)

    pq_outputs = (
        os.popen(
            "python3 compute_stats.py --mode seg_class --pred " + os.path.join(
                os.getcwd(), "results", out, "predictions",
                "pred-epoch" + str(epoch) + ".npy",) + " --true "
            "/home/jdnunes/conic-survey/targets/"
            "hold-out-val/true-" + test_set_source + ".npy"
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

class_wise_pq = pq_outputs[0].replace("'",'"')
class_wise_pq = json.loads(class_wise_pq)
metrics = pq_outputs[1].strip().split(" ")
values = pq_outputs[2].strip().split(" ")
while '' in values:
    values.remove('')
while '' in metrics:
    metrics.remove('')
values = [round(float(v), 4) for v in values]
metrics = dict(zip(metrics, values))
metrics = {**class_wise_pq, **metrics}

map50_bbox = round(coco_evaluator.coco_eval["bbox"].stats[1], 4)
map50_segm = round(coco_evaluator.coco_eval["segm"].stats[1], 4)
map75_bbox = round(coco_evaluator.coco_eval["bbox"].stats[2], 4)
map75_segm = round(coco_evaluator.coco_eval["segm"].stats[2], 4)

map_dict = {"mAP50_bbox": map50_bbox,
            "mAP50_segm": map50_segm,
            "mAP75_bbox": map75_bbox,
            "mAP75_segm": map75_segm
            }

metrics = {"model":out,
           "backbone":
            "resnet-50-fpn",
            "test set centre":test_set_source,
            **map_dict,
            **metrics,
            **class_wise_pq}

# save results to registry file
if os.path.exists("conic-dataset-results.xlsx"):
            df = pd.DataFrame(
                pd.read_excel(
                    "conic-dataset-results.xlsx",
                    sheet_name="Results",
                    engine='openpyxl'))

df.loc[len(df)] = list(metrics.values())
df.to_excel(r"conic-dataset-results.xlsx",
            sheet_name='Results',
            index=False)
