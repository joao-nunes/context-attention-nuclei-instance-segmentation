from typing import Optional
from one_hot import one_hot
import torch.nn.functional as F
import warnings
import torch


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 0.9,
    reduction: str = 'mean',
    eps: Optional[float] = None,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: logits tensor with shape :math:`(N, C, *)`
        where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where
        each value is :math:`0 ≤ targets[i] ≤ C -1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stabiliy.
        This is no longer used.
    Return:
        the computed loss.
    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(
            input,
            target,
            alpha=0.5,
            gamma=2.0,
            reduction='mean')

        >>> output.backward()
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*."
                         f"Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)})'
                         f'to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size},'
                         f'got {target.size()}')

    if not input.device == target.device:
        raise ValueError("input and target must be in the same device. "
                         f"Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1],
                                           device=input.device,
                                           dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)
    focal = -  weight * alpha.to(input.device) * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


# modify fastrcnn_loss to use focal loss in
# classification instead of cross_entropy
# focal loss helps in dealing with imbalanced data

def fastrcnn_loss(
     class_logits,
     box_regression,
     labels,
     regression_targets,
     set_alpha=True):

    """
    Computes the loss for Faster R-CNN.
    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    _, counts = labels.unique(return_counts=True)
    if set_alpha:
        beta = 0.9
        alpha = ((1 - torch.tensor(beta)) / (
            1 - torch.tensor(beta)**counts + 1e-6)).to(torch.float)
        # normalize so that sum(alpha_i)=N :(num_classes)
        alpha *= torch.numel(alpha) / (alpha.sum() + 1e-6)

        # missing labels effective number of samples = 0
        alpha = torch.tensor(
            [(1 - beta)
             if i not in labels.unique()
             else alpha[labels.unique() == i]
             for i in range(class_logits.shape[1])])
    else:
        alpha = torch.tensor(1)
    classification_loss = focal_loss(class_logits, labels, alpha)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()
    return classification_loss, box_loss
