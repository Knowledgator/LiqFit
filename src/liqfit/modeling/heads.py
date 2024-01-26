import abc
from typing import Optional

import torch
from torch import nn
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

from ..losses import binary_cross_entropy_with_logits, cross_entropy

class LiqFitHead(nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs) -> None:
        """LiqFitHead base class."""
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def compute_loss(self, logits, labels) -> torch.Tensor:
        raise NotImplementedError("Should be implemented in a subclass.")

    @staticmethod
    def init_weight(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 1e-2)

    @abc.abstractmethod
    def forward(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None
    ):
        pass

@dataclass
class HeadOutput(ModelOutput):
    embeddings: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class LabelClassificationHead(LiqFitHead):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        multi_target: bool,
        bias: bool = True,
        temperature: int = 1.0,
        eps: float = 1e-5,
    ):
        """Label Classification Head class for Binary or Multi-label tasks.

        Args:
            in_features (_type_): Number of input features.
            out_features (_type_): Number of output features.
            multi_target (_type_): Whether this class is for multi-target
                task or not.
            bias (bool, optional): Whether to add bias to the `Linear`
                layer or not. Defaults to True.
            temperature (int, optional): Temperature that will be used
                to calibrate the head to the task. Defaults to 1.0.
            eps (float, optional): Epsilon value for numirical stability.
                Defaults to 1e-5.
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.multi_target = multi_target
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        LiqFitHead.init_weight(self.linear)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        loss = binary_cross_entropy_with_logits(
            logits, labels, self.multi_target
        )
        return loss

    def forward(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.linear(embeddings)
        logits /= self.temperature + self.eps
        if labels is not None:
            loss = self.compute_loss(logits, labels)
        else:
            loss = None
        return HeadOutput(embeddings=embeddings, logits=logits, loss=loss)


class ClassClassificationHead(LiqFitHead):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        multi_target: bool,
        bias: bool = True,
        temperature: int = 1.0,
        eps: float = 1e-5,
        ignore_index: int = -100,
    ):
        """Class Classification Head class for Sequence/Token classification
            tasks.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            multi_target (bool): Whether this class is for multi-target task
                or not.
            bias (bool, optional): Whether to add bias to the `Linear`
                layer or not. Defaults to True.
            temperature (int, optional): Temperature that will be used
                to calibrate the head to the task. Defaults to 1.0.
            eps (float, optional): Epsilon value for numirical stability.
                Defaults to 1e-5.
            ignore_index (int, optional): Index that will be ignore in
                case of token classification tasks. Defaults to -100.
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.multi_target = multi_target
        self.ignore_index = ignore_index
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        LiqFitHead.init_weight(self.linear)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        return cross_entropy(
            logits, labels, self.multi_target, ignore_index=self.ignore_index
        )

    def forward(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.linear(embeddings) / (self.temperature + self.eps)
        if labels is not None:
            loss = self.compute_loss(logits, labels)
        else:
            loss = None
        return HeadOutput(embeddings=embeddings, logits=logits, loss=loss)


class ClassificationHead(LiqFitHead):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        pooler: nn.Module, 
        loss_func: nn.Module,
        bias: bool = True,
        temperature: int = 1.0,
        eps: float = 1e-5,
    ):
        """Class Classification Head class for Sequence/Token classification
            tasks.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            pooler (torch.nn.Module): Module that applier various pooling opperation on the outputs of a model .
            loss_func (torch.nn.Module): loss function object.
            out_features (int): Number of output features.
            bias (bool, optional): Whether to add bias to the `Linear`
                layer or not. Defaults to True.
            temperature (int, optional): Temperature that will be used
                to calibrate the head to the task. Defaults to 1.0.
            eps (float, optional): Epsilon value for numirical stability.
                Defaults to 1e-5.
            ignore_index (int, optional): Index that will be ignore in
                case of token classification tasks. Defaults to -100.
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.pooler = pooler
        self.loss_func = loss_func
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        LiqFitHead.init_weight(self.linear)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        return self.loss_func(
                    logits, labels
        )

    def forward(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pooled_input = self.pooler(embeddings)
        logits = self.linear(pooled_input) / (self.temperature + self.eps)
        if labels is not None:
            loss = self.compute_loss(logits, labels)
        else:
            loss = None
        return HeadOutput(embeddings=pooled_input, logits=logits, loss=loss)
