from typing import Optional

import torch
from torch import nn


class GlobalMaxPooling1D(nn.Module):
    """Applies Global Max Pooling on the timesteps dimension."""

    def forward(self, x: torch.Tensor):
        return x.amax(-1)


class FirstTokenPooling1D(nn.Module):
    """Takes the first token's embedding."""

    def forward(self, x: torch.Tensor):
        return x[:, 0]


class LastTokenPooling1D(nn.Module):
    """Takes the last token's embedding."""
    
    def forward(self, x: torch.Tensor):
        return x[:, -1]
    

class GlobalAvgPooling1D(nn.Module):
    """Applies Global Average Pooling on the timesteps dimension."""

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        if attention_mask is not None:
            attention_mask = attention_mask.repeat((1, 1, x.shape[-1])).to(
                dtype=x.dtype
            )
            x = x * attention_mask
            return x.sum(1) / attention_mask.sum(1)
        else:
            return x.mean(dim=-1)


class GlobalSumPooling1D(nn.Module):
    """Applies Global Sum Pooling on the timesteps dimension."""

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        
        if attention_mask is not None:
            x = x * attention_mask
        
        return x.sum(-1)


class GlobalRMSPooling1D(nn.Module):
    """Applies Global RMS Pooling on the timesteps dimension."""

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        
        if attention_mask is not None:
            attention_mask = attention_mask.repeat((1, 1, x.shape[-1])).to(
                dtype=x.dtype
            )
            x = x * attention_mask
        
        return x.pow(2).mean(-1).sqrt()
        

class GlobalAbsMaxPooling1D(nn.Module):
    """Applies Global Max Pooling of absolute values on the timesteps dimension."""

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        
        if attention_mask is not None:
            attention_mask = attention_mask.repeat((1, 1, x.shape[-1])).to(
                dtype=x.dtype
            )
            x = x * attention_mask
        
        return x.abs().amax(-1)


class GlobalAbsAvgPooling1D(nn.Module):
    """Applies Global Average Pooling of absolute values on the timesteps dimension."""

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        
        if attention_mask is not None:
            attention_mask = attention_mask.repeat((1, 1, x.shape[-1])).to(
                dtype=x.dtype
            )
            x = x * attention_mask
        
        return x.abs().mean(-1)