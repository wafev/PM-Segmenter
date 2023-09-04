"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path

import torch.nn.functional as F

from timm.models.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None
        self.seq_ranks = None
        self.compute_attn = False

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if self.compute_attn:
            self.attn = attn
            attn.register_hook(self.compute_rank_attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

    def compute_rank_attn(self, grad):
        values = torch.sum(grad * self.attn, dim=0, keepdim=True) \
                      .sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)[0, 0, 0, 1:].data
        values = values / (self.attn.size(0) * self.attn.size(1) * self.attn.size(2))
        if self.seq_ranks is None:
            self.seq_ranks = torch.zeros(grad.size(3) - 1).cuda()

        self.seq_ranks += torch.abs(values)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, num_patches=577, channel=577, merge=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.merge = merge
        self.channel = channel
        if self.merge:
            self.merge_matrix = nn.Parameter(torch.eye(self.channel, num_patches), requires_grad=False)
            self.recover_matrix = nn.Parameter(torch.eye(num_patches, self.channel), requires_grad=False)
            self.token_mask = nn.Parameter(torch.ones(num_patches), requires_grad=False)

    def forward(self, x, mask=None, return_attention=False):
        if self.merge:
            # get the redundant for shortcut
            x_res = torch.tensordot(torch.diag(1 - self.token_mask), x, dims=([1], [1])).permute(1, 0, 2)
            # merge token
            x = torch.tensordot(self.merge_matrix, x, dims=([1], [1])).permute(1, 0, 2)
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.merge:
            # recover the token
            x = torch.tensordot(self.recover_matrix, x, dims=([1], [1])).permute(1, 0, 2)
            # add the shortcut feature
            x = x + x_res

        return x
