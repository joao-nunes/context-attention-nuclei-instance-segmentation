from types import FunctionType
from typing import Any
from torchvision.utils import draw_bounding_boxes
from skimage import morphology
import torch.distributed as dist
import torch
import numpy as np
import pathlib
import os
import shutil
import scipy.ndimage as ndi


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all
    processes have the averaged results. Returns a dict with the same
    fields as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred (ndarray): the 2d array contain instances where each
        instances is marked by non-zero integer.
        by_size (bool): renaming such that larger nuclei have
        a smaller id (on-top).

    Returns:
        new_pred (ndarray): Array with continguous ordering of instances.

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def cropping_center(x, crop_shape, batch=False):
    """Crop an array at the centre with specified dimensions."""
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return


def recur_find_ext(root_dir, ext_list):
    """Recursively find all files in directories end with
     the `ext` such as `ext='.png'`.

    Args:
        root_dir (str): Root directory to grab filepaths from.
        ext_list (list): File extensions to consider.

    Returns:
        file_path_list (list): sorted list of filepaths.
    """
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext_list:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def get_bounding_box(img):
    """Get the bounding box coordinates of a binary input- assumes a single object.

    Args:
        img: input binary image.

    Returns:
        bounding box coordinates

    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def process_bounding_boxes(
    network, images, device="cuda", num_classes=7, dilation=None
):

    predictions = network(images)
    box_splits = []
    colors = []

    # for each image in the batch
    for j in range(len(images)):

        splitted_boxes = [[] for i in range(num_classes)]
        labels = [[] for i in range(num_classes)]

        # split the predicted boxes into a multichan img where each chan maps a label
        for i in range(predictions[j]["boxes"].shape[0]):

            k = int(predictions[j]["labels"][i])
            splitted_boxes[k].append(predictions[j]["boxes"][i, :])
            labels[k].append((k, 0, 0))

        # remove the "containers" for non existing classes
        splitted_boxes = [splitted for splitted in splitted_boxes if splitted != []]
        labels = [color for color in labels if color != []]

        splitted_boxes = [torch.stack(splitted, 0) for splitted in splitted_boxes]

        box_splits.append(splitted_boxes)
        colors.append(labels)

    # draw the boxes (filled)
    drawn_boxes = [
        [
            draw_bounding_boxes(
                torch.zeros_like(images[j][:3].cpu()).to(torch.uint8),
                box_splits[j][i].cpu().detach().to(torch.float),
                colors=colors[j][i],
                fill=True,
            )
            for i in range(len(box_splits[j]))
        ]
        for j in range(len(images))
    ]

    del predictions
    drawn_boxes = [
        [
            colors[i][j][0][0] * (drawn_boxes[i][j] != 0).to(torch.uint8)
            for j in range(len(drawn_boxes[i]))
        ]
        for i in range(len(images))
    ]
    boxes = [[] for i in range(len(images))]

    # Append the drawn boxes in to the right position in a multichannel image
    # If a box does not exist to a given label, append zeros_like(input image)

    for j in range(len(images)):
        unique = [
            drawn_boxes[j][k].unique()[-1].numpy() for k in range(len(drawn_boxes[j]))
        ]
        for i in range(1, num_classes):
            if i in unique:
                for k in range(len(drawn_boxes[j])):
                    if np.unique(drawn_boxes[j][k])[-1] == i:
                        boxes[j].append(drawn_boxes[j][k][0].to(device))
            else:
                boxes[j].append(
                    torch.zeros_like(images[j][0:3][-1]).to(device).to(torch.float32)
                )
    del drawn_boxes
    # Normalize the filled boxes (network expects tensors in range [0, 1])
    # If dilation not None, performe binary dilation
    if dilation is None:
        boxes = [
            [
                torch.div(boxes[j][i], boxes[j][i].max())
                if boxes[j][i].max() != 0
                else boxes[j][i]
                for i in range(len(boxes[j]))
            ]
            for j in range(len(images))
        ]
    else:
        selem = np.ones((dilation, dilation))
        boxes = [
            [
                torch.tensor(
                    morphology.binary_dilation(
                        torch.div(boxes[j][i], boxes[j][i].max()).cpu(), selem=selem
                    )
                )
                .to(torch.float32)
                .to(device)
                if boxes[j][i].max() != 0
                else boxes[j][i]
                for i in range(len(boxes[j]))
            ]
            for j in range(len(images))
        ]
    dist = [
        [
            ndi.morphology.distance_transform_edt(boxes[j][i].cpu(), sampling=2)
            for i in range(len(boxes[j]))
        ]
        for j in range(len(boxes))
    ]
    # dist = [[torch.tensor(1 - dist[j][i]/(dist[j][i].max()+0.001), dtype=torch.float32).to(device) for i in range(len(boxes[j]))] for j in range(len(boxes))]
    dist = [
        [
            torch.tensor(
                dist[j][i] / (dist[j][i].max() + 0.001), dtype=torch.float32
            ).to(device)
            for i in range(len(boxes[j]))
        ]
        for j in range(len(boxes))
    ]
    boxes = [torch.stack(dist[i]) for i, _ in enumerate(dist)]

    return boxes


def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive,
    unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method
    <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same
    API call within a process.
    It does not collect any data from open-source users since
    it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;
    Args:
        obj (class instance or method): an object to extract info from.
    """
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")
