from __future__ import annotations
from typing import Optional
import torch.nn.functional as F
from kornia.losses import focal_loss
import torch


def binary_cross_entropy_with_logits(logits: torch.Tensor,
                                     labels: torch.Tensor,
                                     multi_target: bool = False,
                                     weight: Optional[torch.Tensor] = None,
                                     reduction: str = 'mean') -> torch.Tensor:
    """Wrapper function for adding support for multi_target training.

    Args:
        logits (torch.Tensor): Tensor with shape (B, T, D) where B is batch
            size, T is timesteps and D is embedding dimension.
        labels (torch.Tensor): Tensor with shape (B, T) where B is batch size,
            T is timesteps.
        multi_target (bool, optional): Whether the labels are multi target or
            one target for the entire sequence. Defaults to False.
        weight (Optional[torch.Tensor], optional): a manual rescaling weight
            if provided it's repeated to match input tensor shape.
            Defaults to None.
        reduction (str, optional): Reduction type that will be applied on the
            loss function, supported: 'mean', 'sum' or 'none'.
            Defaults to 'mean'.

    Returns:
        torch.Tensor: Loss tensor.
    """
    if multi_target:
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
    else:
        labels = labels.view(-1)
    loss = F.binary_cross_entropy_with_logits(logits,
                                              labels,
                                              weight=weight,
                                              reduction=reduction)
    return loss


class BinaryCrossEntropyLoss(torch.nn.Module):
    
    def __init__(self, multi_target=False, weight=None, reduction='mean'):
        super().__init__()
        """Calculate binary cross-entropy loss with support for multi target training.

        Args:
            multi_target (bool, optional): Whether the labels are multi target or
                one target for the entire sequence. Defaults to False.
            weight (Optional[torch.Tensor], optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape.
                Defaults to None.
            reduction (str, optional): Reduction type that will be applied on the
                loss function, supported: 'mean', 'sum' or 'none'.
                Defaults to 'mean'.

        Returns:
            torch.Tensor: Loss tensor.
        Examples:
            loss = BinaryCrossEntropyLoss()(logits, targets)
        """
        self.multi_target = multi_target
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, logits, target):
        
        loss = binary_cross_entropy_with_logits(
            logits, 
            target,
            multi_target=self.multi_target,
            weight=self.weight,
            reduction=self.reduction,
        )
        
        return loss


def cross_entropy(logits: torch.Tensor,
                  labels: torch.Tensor,
                  multi_target: bool = False,
                  weight: Optional[torch.Tensor] = None,
                  ignore_index: int = -100,
                  reduction: str = 'mean',
                  label_smoothing: float = 0.0):
    """Wrapper function for adding support for multi_target training.

    Args:
        logits (torch.Tensor): Tensor with shape (B, T, D) where B is batch
            size, T is timesteps and D is embedding dimension.
        labels (torch.Tensor): Tensor with shape (B, T) where B is batch size,
            T is timesteps.
        multi_target (bool, optional): Whether the labels are multi target or
            one target for the entire sequence. Defaults to False.
        weight (Optional[torch.Tensor], optional): a manual rescaling weight
            if provided it's repeated to match input tensor shape.
            Defaults to None.
        ignore_index (int, optional): Index value that will be ignored during
            loss calculation. Defaults to -100.
        reduction (str, optional): Reduction type that will be applied on the
            loss function, supported: 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies
            the amount of smoothing when computing the loss, where 0.0 means
            no smoothing. Defaults to 0.0.

    Returns:
        torch.Tensor: Loss tensor.
    """
    if multi_target:
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
    else:
        labels = labels.view(-1)
    loss = F.cross_entropy(logits,
                           labels,
                           weight=weight,
                           reduction=reduction,
                           ignore_index=ignore_index,
                           label_smoothing=label_smoothing)
    return loss


class CrossEntropyLoss(torch.nn.Module):
    
    def __init__(self, multi_target=False, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
        super().__init__()
        """Calculate cross-entropy loss while ignoring specified target labels.

        Args:
            multi_target (bool, optional): Whether the labels are multi target or
                one target for the entire sequence. Defaults to False.
            weight (Optional[torch.Tensor], optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape.
                Defaults to None.
            ignore_index (int, optional): Index value that will be ignored during
                loss calculation. Defaults to -100.
            reduction (str, optional): Reduction type that will be applied on the
                loss function, supported: 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies
                the amount of smoothing when computing the loss, where 0.0 means
                no smoothing. Defaults to 0.0.

        Returns:
            torch.Tensor: Loss tensor.
        Examples:
            loss = CrossEntropyLoss()(logits, targets)
        """
        self.multi_target = multi_target
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, target):
            
        loss = cross_entropy(
            logits, 
            target,
            multi_target=self.multi_target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )
        
        return loss


def focal_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str | None = "mean",
) -> torch.Tensor:
    """Calculate focal loss while ignoring specified target labels.
    
    Args:
        logits (torch.Tensor): Model predictions.
        target (torch.Tensor): True labels.
        ignore_index (int): Label to ignore from loss calculation.
        alpha (float): Focal loss alpha parameter.
        gamma (float): Focal loss gamma parameter.
        reduction (str | None): Method to reduce loss.
    
    Returns:
        torch.Tensor: Loss tensor.
        
    This function calculates the focal loss between logits and targets, 
    while ignoring any examples where the target is equal to ignore_index.

    Examples:
    
        loss = focal_loss_with_mask(logits, targets, ignore_index=-100)
    """
    if not isinstance(ignore_index, int):
        raise ValueError('Expected `ignore_index` to be of type `int`. '
                         f'Received: {type(ignore_index)}')

    mask = target == ignore_index

    # To make focal_loss function work because
    # it cannot work with -ve numbers (e.g. -100).
    if ignore_index != 0:
        target_without_ignore_index = target.masked_fill(mask, 0)

    loss = focal_loss(
        pred=logits,
        target=target_without_ignore_index,
        alpha=alpha,
        gamma=gamma,
        reduction="none",
    )

    loss = loss.masked_fill(mask.view(-1, 1), torch.inf)

    if reduction == "mean":
        return loss[loss != torch.inf].mean()
    elif reduction == "sum":
        return loss[loss != torch.inf].sum()
    elif reduction is None:
        return loss
    else:
        raise ValueError(
            'Expected reduction to be "sum", "mean" or `None`. '
            f"Received: {reduction}."
        )

class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """Calculate focal loss while ignoring specified target labels.
        Args:
            logits (torch.Tensor): Model predictions.
            target (torch.Tensor): True labels.
            ignore_index (int): Label to ignore from loss calculation.
        alpha: Weighting factor that ranges between [0, 1]`.
            gamma: Focusing parameter gamma >= 0`.
            reduction (str | None): Reduction type for loss reduction.
                Supported: 'mean', 'sum' or 'none'. Defaults to 'mean'

        Returns:
            torch.Tensor: Loss tensor.
        Examples:
            loss = FocalLoss()(logits, targets)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        return focal_loss_with_mask(
            logits=logits,
            target=target,
            ignore_index=self.ignore_index,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
