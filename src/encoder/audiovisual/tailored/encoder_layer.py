# Copyright 2023 David Gimeno-Gómez (PRHLT, UPV)
# Code mainly based on the Yifan Peng's implementation (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unified Tailored AudioVisual E-Branchformer Encoder Layer definition.

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

class TailoredEncoderLayer(torch.nn.Module):
    """My Unified and Tailored Branchformer Encoder Layer module.

    Args:
        size (int): model dimension
        acoustic_attn: standard self-attention or efficient attention for the acoustic branch, optional
        acoustic_cgmlp: ConvolutionalGatingMLP for the acoustic branch, optional
        visual_attn: standard self-attention or efficient attention for the visual branch, optional
        visual_cgmlp: ConvolutionalGatingMLP for the visual branch, optional
        feed_forward_macaron: positionwise feed forward, shared between branches, optional
        feed_forward: positionwise feed forward macaron, shared between branches, optional
        dropout_rate (float): dropout probability
        acoustic_branch_drop_rate (float): probability of dropping the acoustic-based branch,
        stochastic_depth_rate (float): stochastic depth probability
    """

    def __init__(
        self,
        size: int,
        feed_forward_macaron: Optional[torch.nn.Module],
        acoustic_attn: Optional[torch.nn.Module],
        acoustic_cgmlp: Optional[torch.nn.Module],
        visual_attn: Optional[torch.nn.Module],
        visual_cgmlp: Optional[torch.nn.Module],
        feed_forward: Optional[torch.nn.Module],
        dropout_rate: float,
        acoustic_branch_drop_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()

        self.size = size
        self.ff_scale = 0.5

        # -- macaron feed forward module (sharing parameters between branches)
        self.feed_forward_macaron = feed_forward_macaron
        self.norm_ff_macaron = LayerNorm(size)

        # -- acoustic branch
        self.acoustic_attn = acoustic_attn
        if self.acoustic_attn is not None:
            self.acoustic_norm_mha = LayerNorm(size) # for the MHA module

        self.acoustic_cgmlp = acoustic_cgmlp
        if self.acoustic_cgmlp is not None:
            self.acoustic_norm_cgmlp = LayerNorm(size) # for the cgMLP module

        # -- visual branch
        self.visual_attn = visual_attn
        if self.visual_attn is not None:
            self.visual_norm_mha = LayerNorm(size) # for the MHA module

        self.visual_cgmlp = visual_cgmlp
        if self.visual_cgmlp is not None:
            self.visual_norm_cgmlp = LayerNorm(size)  # for the cgMLP module

        # -- feed forward module (sharing parameters between branches)
        self.feed_forward = feed_forward
        self.norm_ff = LayerNorm(size)

        # -- final output block normalisation
        self.norm_final = LayerNorm(size)

        # -- dropout layer and rates
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.acoustic_branch_drop_rate = acoustic_branch_drop_rate
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, audio_input, audio_masks, video_input, video_masks, cache=None):
        """Compute encoded features.

        Args:
            audio_input (Union[Tuple, torch.Tensor]): Input audio tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            audio_masks (torch.Tensor): Input video mask tensor for both inputs (#batch, time).
            video_input (Union[Tuple, torch.Tensor]): Input video tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            video_masks (torch.Tensor): Input video mask tensor for both inputs (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output audio tensor (#batch, time, size).
            torch.Tensor: Output audio mask tensor (#batch, time).
            torch.Tensor: Output video tensor (#batch, time, size).
            torch.Tensor: Output video mask tensor (#batch, time).
        """
        # -- input implementation details
        if cache is not None:
            raise NotImplementedError("cache is not None, which is not tested")

        if isinstance(audio_input, tuple):
            audio, audio_pos_emb = audio_input[0], audio_input[1]
        else:
            audio, audio_pos_emb = audio_input, None

        if isinstance(video_input, tuple):
            video, video_pos_emb = video_input[0], video_input[1]
        else:
            video, video_pos_emb = video_input, None

        # -- applying stochastic depth
        skip_layer = False
        stoch_layer_coeff = 1.0
        # -- -- with stochastic depth, residual connection `x + f(x)` becomes
        # -- -- `x <- x + 1 / (1 - p) * f(x)` at training time.
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if (audio_pos_emb is not None) and (video_pos_emb is not None):
                return (audio, audio_pos_emb), audio_masks, (video, video_pos_emb), video_masks
            if audio_pos_emb is not None:
                return (audio, audio_pos_emb), audio_masks, video, video_masks
            if video_pos_emb is not None:
                return audio, audio_masks, (video, video_pos_emb), video_masks

            return audio, audio_masks, video, video_masks

        # -- acoustic branch
        # -- -- macaron feed forward module
        residual = audio
        audio = self.norm_ff_macaron(audio)
        audio = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(audio))

        # -- -- tailored module
        residual = audio
        if (self.acoustic_attn is not None) and (self.acoustic_cgmlp is not None):
            raise RuntimeError(
                f"Only one of the possible acoustic tailored modules should be not None: {self.acoustic_attn}, {self.acoustic_cgmlp}."
            )

        # -- -- -- MHA module
        if self.acoustic_attn is not None:
            audio = self.acoustic_norm_mha(audio)

            if isinstance(self.acoustic_attn, FastSelfAttention):
                audio_att = self.acoustic_attn(audio, audio_masks)
            else:
                if audio_pos_emb is not None:
                    audio_att = self.acoustic_attn(audio, audio, audio, audio_pos_emb, audio_masks)
                else:
                    audio_att = self.acoustic_attn(audio, audio, audio, audio_masks)

            audio = residual + stoch_layer_coeff * self.dropout(audio_att)

        # -- -- -- cgMLP module
        if self.acoustic_cgmlp is not None:
            audio = self.acoustic_norm_cgmlp(audio)

            if audio_pos_emb is not None:
                audio = (audio, audio_pos_emb)
            audio = self.acoustic_cgmlp(audio, audio_masks)
            if isinstance(audio, tuple):
                audio = audio[0]

            audio = residual + stoch_layer_coeff * self.dropout(audio)

        # -- -- feed forward module
        residual = audio
        audio = self.norm_ff(audio)
        audio = residual + self.ff_scale * self.dropout(self.feed_forward(audio))

        # -- -- final output block normalisation
        audio = self.norm_final(audio)

        # -- visual branch
        # -- -- macaron feed forward module
        residual = video
        video = self.norm_ff_macaron(video)
        video = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(video))

        # -- -- tailored module
        residual = video
        if (self.visual_attn is not None) and (self.visual_cgmlp is not None):
            raise RuntimeError(
                f"Only one of the possible visual tailored modules should be not None: {self.visual_attn}, {self.visual_cgmlp}."
            )

        # -- -- -- MHA module
        if self.visual_attn is not None:
            video = self.visual_norm_mha(video)

            if isinstance(self.visual_attn, FastSelfAttention):
                video_att = self.visual_attn(video, video_masks)
            else:
                if video_pos_emb is not None:
                    video_att = self.visual_attn(video, video, video, video_pos_emb, video_masks)
                else:
                    video_att = self.visual_attn(video, video, video, video_masks)

            video = residual + stoch_layer_coeff * self.dropout(video_att)

        # -- -- -- cgMLP module
        residual = video
        if self.visual_cgmlp is not None:
            video = self.visual_norm_cgmlp(video)

            if video_pos_emb is not None:
                video = (video, video_pos_emb)
            video = self.visual_cgmlp(video, video_masks)
            if isinstance(video, tuple):
                video = video[0]

            video = residual + stoch_layer_coeff * self.dropout(video)

        # -- -- feed forward module
        residual = video
        video = self.norm_ff(video)
        video = residual + self.ff_scale * self.dropout(self.feed_forward(video))

        # -- -- final output block normalisation
        video = self.norm_final(video)

        # -- output implementation details
        if (audio_pos_emb is not None) and (video_pos_emb is not None):
            return (audio, audio_pos_emb), audio_masks, (video, video_pos_emb), video_masks
        if audio_pos_emb is not None:
            return (audio, audio_pos_emb), audio_masks, video, video_masks
        if video_pos_emb is not None:
            return audio, audio_masks, (video, video_pos_emb), video_masks

        return audio, audio_masks, video, video_masks
