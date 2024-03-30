#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright 2023 David Gimeno-Gómez (Pattern Recognition and Human Language Technology, UPV)
#  This code is mainly based on the implementation of Shigeki Karita (https://github.com/espnet/espnet/blob/3bb4cb5154d8ebb65f5eaf8e8d7e19714cd548aa/espnet/nets/pytorch_backend/transformer/decoder_layer.py)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder self-attention layer definition."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DecoderLayerSimT(nn.Module):
    """Single decoder layer module based on Sim-T:
        https://arxiv.org/pdf/2304.04991.pdf

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)


    """

    def __init__(
        self,
        size,
        mha_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayerSimT, self).__init__()
        self.size = size
        self.mha_attn = mha_attn
        self.feed_forward = feed_forward

        self.norm_mha = LayerNorm(size)
        self.norm_ffn = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, s1, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            score1: Attention score computed by the first decoder layer.
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        # -- Pre-MHA Module
        residual = tgt
        if self.normalize_before:
            tgt = self.norm_mha(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.mha_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear(tgt_concat)
        else:
            x = residual + self.dropout(self.mha_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm_mha(x)

        # -- -- getting the score matrix provided by the first decoder layer w.r.t. the labels
        if s1 is None:
            if cache is None:
                s1 = self.mha_attn.attn # (# batch, head, time1, time2)
            else:
                s1 = self.mha_attn.attn[:, :, :, -1:] # (#batch, head, time1, 1)

        # -- MHA Module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.mha_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.mha_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm_mha(x)

        # -- FFN Module
        residual = x
        if self.normalize_before:
            x = self.norm_ffn(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ffn(x)

        # -- Post-MHA Module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        x = residual + self.post_mha(x, s1)

        if not self.normalize_before:
            x = self.norm_mha(x)

        # -- FFN Module
        residual = x
        if self.normalize_before:
            x = self.norm_ffn(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ffn(x)


        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask, s1

    def post_mha(self, values, scores):
        """Apply the Post-MHA module defined in https://arxiv.org/pdf/2304.04991.pdf

        Args:
            values (torch.Tensor): Value tensor (#batch, time2, size).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """

        # -- transform value
        n_batch = values.size(0)
        value = self.mha_attn.linear_v(values).view(n_batch, -1, self.mha_attn.h, self.mha_attn.d_k)
        value = value.transpose(1, 2)  # (batch, head, time2, d_k)

        # -- applying attention context vector computed by the first layer of the decoder
        p_attn = self.mha_attn.dropout(scores)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.mha_attn.h * self.mha_attn.d_k)
        )  # (batch, time1, d_model)

        return x  # (batch, time1, d_model)
