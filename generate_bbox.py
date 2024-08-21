from torchvision.ops import masks_to_boxes
import numpy as np
import torchvision.transforms.functional as F
import torch
import matplotlib.pyplot as plt


def get_bbox(mask, crop=False):

    obj_ids = torch.unique(mask)
    # if background in mask -> ignore
    if obj_ids[0] == 0:
        obj_ids = obj_ids[1:]
    invalid_bbox = []

    masks = mask == obj_ids[:, None, None]
    boxes = masks_to_boxes(masks)
    valid_boxes = []
    ids2remove = []
    for i, box in enumerate(boxes):
        # one corner at least must be inside the 224x224 region
        # if crop:
        #    if (box[0] > 224) or (box[2] < (256-224)) or (box[3] <
        #       (256-224)) or (box[1] > 224) or ((box[2]-box[0]) <= 0) or (
        #           (box[3]-box[1]) <= 0) or (torch.numel(box)) <= 1:

        #        masks = torch.cat([masks[:i, :, :], masks[i+1:, :, :]])
        #        continue

        # all boxes should have positive height and width
        if (box[2] - box[0]) <= 0 or ((box[3] - box[1]) <= 0) or torch.numel(box) <= 1: # noqa
            ids2remove.append(i)
            invalid_bbox.append(obj_ids[i])
            continue
        valid_boxes.append(np.array(box))
    keep = [i for i in range(len(boxes)) if i not in ids2remove]
    keep = tuple(keep)
    masks = masks[keep, ...]
    if valid_boxes == []:
        valid_boxes = [[0, 0, 1, 1]]
    num_objs = len(valid_boxes)

    return masks.type(torch.uint8), torch.as_tensor(
        np.array(valid_boxes)).to(torch.float32), num_objs, invalid_bbox


def draw_bboxes(imgs):

    plt.rcParams["savefig.bbox"] = "tight"
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return axs
