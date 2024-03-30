# Copyright 2022 Yifan Peng (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""E-Branchformer Encoder Layer definition.

Reference:
    Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji Watanabe,
    “Branchformer: Parallel MLP-Attention Architectures to Capture
    Local and Global Context for Speech Recognition and Understanding,”
    in Proceedings of ICML, 2022.

"""

import logging
from typing import List, Optional, Tuple, Union

import copy
import numpy
import torch
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.layers.cgmlp import ConvolutionalGatingMLP
from espnet2.asr.layers.fastformer import FastSelfAttention
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (  # noqa: H301
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)

class MyBranchformerEncoderLayer(torch.nn.Module):
    """My Branchformer Encoder Layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention, optional
        cgmlp: ConvolutionalGatingMLP, optional
        feed_forward_macaron: positionwise feed forward, optional
        feed_forward: positionwise feed forward macaron, optional
        dropout_rate (float): dropout probability
        merge_method (str): concat, learned_ave, fixed_ave
        cgmlp_weight (float): weight of the cgmlp branch, between 0 and 1,
            used if merge_method is fixed_ave
        attn_branch_drop_rate (float): probability of dropping the attn branch,
            used if merge_method is learned_ave
        stochastic_depth_rate (float): stochastic depth probability
    """

    def __init__(
        self,
        size: int,
        attn: Optional[torch.nn.Module],
        cgmlp: Optional[torch.nn.Module],
        feed_forward_macaron: Optional[torch.nn.Module],
        feed_forward: Optional[torch.nn.Module],
        dropout_rate: float,
        merge_method: str,
        cgmlp_weight: float = 0.5,
        attn_branch_drop_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        assert (attn is not None) or (
            cgmlp is not None
        ), "At least one branch should be valid"

        self.size = size

        self.attn = attn
        self.cgmlp = cgmlp
        self.feed_forward_macaron = feed_forward_macaron
        self.feed_forward = feed_forward

        self.ff_scale = 1.0

        self.merge_method = merge_method
        self.cgmlp_weight = cgmlp_weight
        self.attn_branch_drop_rate = attn_branch_drop_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_two_branches = (attn is not None) and (cgmlp is not None)

        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = LayerNorm(size) # for the Macaron FFN module
        if attn is not None:
            self.norm_mha = LayerNorm(size)  # for the MHA module
        if cgmlp is not None:
            self.norm_mlp = LayerNorm(size)  # for the MLP module
        if self.feed_forward is not None:
            self.norm_ff = LayerNorm(size) # for the FFN module
        self.norm_final = LayerNorm(size)  # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)

        if self.use_two_branches:
            if merge_method == "concat":
                self.merge_proj = torch.nn.Linear(size + size, size)

            elif merge_method == "learned_ave":
                # attention-based pooling for two branches
                self.pooling_proj1 = torch.nn.Linear(size, 1)
                self.pooling_proj2 = torch.nn.Linear(size, 1)

                # linear projections for calculating merging weights
                self.weight_proj1 = torch.nn.Linear(size, 1)
                self.weight_proj2 = torch.nn.Linear(size, 1)

                # linear projection after weighted average
                self.merge_proj = torch.nn.Linear(size, size)

            elif merge_method == "fixed_ave":
                assert (
                    0.0 <= cgmlp_weight <= 1.0
                ), "cgmlp weight should be between 0.0 and 1.0"

                # remove the other branch if only one branch is used
                if cgmlp_weight == 0.0:
                    self.use_two_branches = False
                    self.cgmlp = None
                    self.norm_mlp = None
                elif cgmlp_weight == 1.0:
                    self.use_two_branches = False
                    self.attn = None
                    self.norm_mha = None

                # linear projection after weighted average
                self.merge_proj = torch.nn.Linear(size, size)

            else:
                raise ValueError(f"unknown merge method: {merge_method}")

        else:
            self.merge_proj = torch.nn.Identity()

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if cache is not None:
            raise NotImplementedError("cache is not None, which is not tested")

        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        # Macaron FFN: positionwise feed forward module
        residual = x
        x = self.norm_ff_macaron(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        if self.attn is not None:
            x1 = self.norm_mha(x1)

            if isinstance(self.attn, FastSelfAttention):
                x_att = self.attn(x1, mask)
            else:
                if pos_emb is not None:
                    x_att = self.attn(x1, x1, x1, pos_emb, mask)
                else:
                    x_att = self.attn(x1, x1, x1, mask)

            x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        if self.cgmlp is not None:
            x2 = self.norm_mlp(x2)

            if pos_emb is not None:
                x2 = (x2, pos_emb)
            x2 = self.cgmlp(x2, mask)
            if isinstance(x2, tuple):
                x2 = x2[0]

            x2 = self.dropout(x2)

        # Merge two branches
        if self.use_two_branches:
            if self.merge_method == "concat":
                x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj(torch.cat([x1, x2], dim=-1))
                )
            elif self.merge_method == "learned_ave":
                if (
                    self.training
                    and self.attn_branch_drop_rate > 0
                    and torch.rand(1).item() < self.attn_branch_drop_rate
                ):
                    # Drop the attn branch
                    w1, w2 = 0.0, 1.0
                else:
                    # branch1
                    score1 = (
                        self.pooling_proj1(x1).transpose(1, 2) / self.size**0.5
                    )  # (batch, 1, time)
                    if mask is not None:
                        min_value = float(
                            numpy.finfo(
                                torch.tensor(0, dtype=score1.dtype).numpy().dtype
                            ).min
                        )
                        score1 = score1.masked_fill(mask.eq(0), min_value)
                        score1 = torch.softmax(score1, dim=-1).masked_fill(
                            mask.eq(0), 0.0
                        )
                    else:
                        score1 = torch.softmax(score1, dim=-1)
                    pooled1 = torch.matmul(score1, x1).squeeze(1)  # (batch, size)
                    weight1 = self.weight_proj1(pooled1)  # (batch, 1)

                    # branch2
                    score2 = (
                        self.pooling_proj2(x2).transpose(1, 2) / self.size**0.5
                    )  # (batch, 1, time)
                    if mask is not None:
                        min_value = float(
                            numpy.finfo(
                                torch.tensor(0, dtype=score2.dtype).numpy().dtype
                            ).min
                        )
                        score2 = score2.masked_fill(mask.eq(0), min_value)
                        score2 = torch.softmax(score2, dim=-1).masked_fill(
                            mask.eq(0), 0.0
                        )
                    else:
                        score2 = torch.softmax(score2, dim=-1)
                    pooled2 = torch.matmul(score2, x2).squeeze(1)  # (batch, size)
                    weight2 = self.weight_proj2(pooled2)  # (batch, 1)

                    # normalize weights of two branches
                    merge_weights = torch.softmax(
                        torch.cat([weight1, weight2], dim=-1), dim=-1
                    )  # (batch, 2)
                    merge_weights = merge_weights.unsqueeze(-1).unsqueeze(
                        -1
                    )  # (batch, 2, 1, 1)
                    w1, w2 = merge_weights[:, 0], merge_weights[:, 1]  # (batch, 1, 1)

                self.weight_global = w1
                self.weight_local = w2

                x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj(w1 * x1 + w2 * x2)
                )
            elif self.merge_method == "fixed_ave":
                x = x + stoch_layer_coeff * self.dropout(
                    self.merge_proj(
                        (1.0 - self.cgmlp_weight) * x1 + self.cgmlp_weight * x2
                    )
                )
            else:
                raise RuntimeError(f"unknown merge method: {self.merge_method}")
        else:
            if self.attn is None:
                x = x + stoch_layer_coeff * self.dropout(self.merge_proj(x2))
            elif self.cgmlp is None:
                x = x + stoch_layer_coeff * self.dropout(self.merge_proj(x1))
            else:
                # This should not happen
                raise RuntimeError("Both branches are not None, which is unexpected.")

        # FFN: positionwise feed forward module
        residual = x
        x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask
