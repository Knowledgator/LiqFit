from __future__ import annotations
import abc

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class LiqFitBackbone(PreTrainedModel, abc.ABC):
    def __init__(
        self, config: PretrainedConfig, backbone: nn.Module, push_backbone_only: bool = False
    ) -> None:
        """Backbone model wrapper."""
        super().__init__(config=config)
        self.push_backbone_only = push_backbone_only
        self.backbone = backbone

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
            output = self.backbone.push_to_hub(
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

    @abc.abstractmethod
    def encode(self, input_ids, attention_mask=None) -> torch.Tensor:
        raise NotImplementedError("Should be implemented in a subclass.")

