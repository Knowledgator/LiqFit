from __future__ import annotations

from typing import Optional

import inspect
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from transformers import PreTrainedModel, PretrainedConfig

from .backbone import LiqFitBackbone
from .heads import LiqFitHead, HeadOutput
from ..utils.standardization import convert_to_numpy

class LiqFitModel(PreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        backbone: LiqFitBackbone | nn.Module | PreTrainedModel,
        head: Optional[LiqFitHead | LogisticRegression] = None,
        loss_func: Optional[nn.Module] = None,
        normalize_backbone_embeddings: bool = False,
        labels_name: str = "labels",
        push_backbone_only: bool = False,
    ):
        """Model container that groups the backbone and head together
            and applies forward on both of them.

        Args:
            backbone (LiqFitBackbone): Backbone model.
            head (Optional[LiqFitHead  |  LogisticRegression], optional):
                Head that is defined for the task. Could be set to `None`
                if the head is already attached to the backbone.
                Defaults to None.
            loss_func (Optional[nn.Module]): class for calculation of loss functions.
            normalize_backbone_embeddings (bool, optional): Whether to
                normalize the backbone embeddings or not (Requires the
                backbone output to be a `torch.Tensor` not a Huggingface
                object). Defaults to False.
            labels_name (str, optional): Labels name that will be sent in the
                **kwargs for loss calculation. Defaults to "labels".

        Example 1:
            # make sure that the output from this model
            # is a torch.Tensor otherwise wrap it using LiqFitBackbone.
            my_backbone = AutoModel.from_pretrained(....)
            head = LiqFit.modeling.LabelClassificationHead(...)
            model = LiqFitModel(my_backbone.config, my_backbone, head)

        Example 2:
            class MyBackbone(LiqFitBackbone):
                def __init__(self):
                    my_backbone = AutoModel.from_pretrained(....)
                    super().__init__(my_backbone.config, backbone=backbone)
                def encode(self, input_ids, attention_mask=None) -> torch.Tensor:
                    output = self.backbone(input_ids, attention_mask=attention_mask)
                    return output

            my_backbone = MyBackbone()
            head = LiqFit.modeling.LabelClassificationHead(...)
            model = LiqFitModel(my_backbone.config, my_backbone, head)
        """

        super().__init__(config=config)
        self._is_sklearn_head = None
        self.backbone = backbone
        self._determine_and_validate_head_type(head)
        self.head = head
        self.loss_func = loss_func
        self.normalize_backbone_embeddings = normalize_backbone_embeddings
        self.labels_name = labels_name
        self.push_backbone_only = push_backbone_only
        self.expecting_labels = 'labels' in inspect.getfullargspec(self.backbone.forward).args

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: bool | None = None,
        commit_message: str | None = None,
        private: bool | None = None,
        token: bool | str | None = None,
        max_shard_size: int | str | None = "5GB",
        create_pr: bool = False,
        safe_serialization: bool = True,
        revision: str = None,
        commit_description: str = None,
        **deprecated_kwargs,
    ) -> str:
        if self.push_backbone_only:
            if isinstance(self.backbone, (LiqFitBackbone, PreTrainedModel)):
                return self.backbone.push_to_hub(
                    repo_id,
                    use_temp_dir,
                    commit_message,
                    private,
                    token,
                    max_shard_size,
                    create_pr,
                    safe_serialization,
                    revision,
                    commit_description,
                    **deprecated_kwargs,
                )
        else:
            output = super().push_to_hub(
                repo_id=repo_id,
                use_temp_dir=use_temp_dir,
                commit_message=commit_message,
                private=private,
                token=token,
                max_shard_size=max_shard_size,
                create_pr=create_pr,
                safe_serialization=safe_serialization,
                revision=revision,
                commit_description=commit_description,
                **deprecated_kwargs,
            )
        return output

    def freeze_weights(self):
        self.requires_grad_(False)

    def unfreeze_weights(self):
        self.requires_grad_(True)

    def _determine_and_validate_head_type(self, head):
        if head is None:
            return

        self._is_sklearn_head = isinstance(head, LogisticRegression)
        if not self._is_sklearn_head and not isinstance(head, LiqFitHead):
            raise TypeError(
                "Expected `head` to be of type "
                "`LogisticRegression` or `LiqFitHead`. "
                f"Received: {type(head)}."
            )

    def _backbone_forward(self, **kwargs):
        if isinstance(self.backbone, LiqFitBackbone):
            output = self.backbone.encode(**kwargs)
            if not isinstance(output, torch.Tensor):
                raise ValueError(
                    "Expected output from backbone model to be of type "
                    f"`torch.Tensor`. Received: {type(output)}."
                )
        else:
            output = self.backbone(**kwargs)
        return output

    def _torch_head_forward(self, embeddings, labels=None):
        output = self.head(embeddings, labels)
        return output

    def _sklearn_head_forward(self, embeddings):
        embeddings = convert_to_numpy(embeddings)
        output = self.head.predict(embeddings)
        return output

    def _head_forward(self, inputs, labels=None):
        if self._is_sklearn_head:
            return self._sklearn_head_forward(inputs)
        else:
            return self._torch_head_forward(inputs, labels)

    def forward(self, **kwargs):
        labels = kwargs.pop('labels', None)

        output = self._backbone_forward(**kwargs)
        
        if not isinstance(output, torch.Tensor):
            if isinstance(output, tuple):
                output = output[0]
            elif 'logits' in output:
                output = output['logits']
            elif 'last_hidden_state' in output:
                output = output['last_hidden_state']
            else:
                raise NotImplementedError('A model output should contains logits or last_hidden_state.')
            
        if self.normalize_backbone_embeddings:
            if isinstance(output, torch.Tensor):
                output = F.normalize(output, p=2.0, dim=-1)
            else:
                raise TypeError(
                    "Normalizing the embedding requires type of "
                    f"`torch.Tensor`. Received: {type(output)}."
                )
        if self.head is not None:
            output = self._head_forward(output, labels)
        elif self.loss_func is not None and labels is not None:
            loss = self.loss_func(output, labels)
            output = HeadOutput(logits=output, loss=loss)
        return output
