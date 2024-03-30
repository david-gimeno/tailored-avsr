# Copyright 2023 David Gimeno-Gómez (PRHLT, UPV)
# Code mainly based on the Yifan Peng's implementation (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Early Audiovisual Upsampling Fusion with learnable weights.

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
import torch.nn.functional as F
from typeguard import check_argument_types
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask

from src.audiovisual_fusion.audiovisual_fusion_abs_module import AudioVisualFusionAbsModule
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

class AdaptiveAudioVisualFusion(AudioVisualFusionAbsModule):
    """Adaptive AudioVisual Fusion, whose modality merge method is mainly based on the
    proposed for the Branchformer architecture.

    Args:
        size (int): model dimension
        audiovisual_layer_type: layer defined to process
            the audiovisual speech representation
        merge_method (str): concat, learned_ave, fixed_ave
        activation_type (str): relu, swish
        acoustic_weight (float): weight of the acoustic branch, between 0 and 1,
            used if merge_method is fixed_ave
        dropout_rate (float): dropout probability
        acoustic_branch_drop_rate (float): probability of dropping the acoustic-based branch,
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        hidden_units: int = 2048,
        audiovisual_layer_type: str = "upsampling_positionwise",
        merge_method: str = "learned_ave",
        activation_type: str = "swish",
        acoustic_weight: float = 0.5,
        dropout_rate: float = 0.1,
        acoustic_branch_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.input_size = input_size
        self._output_size = output_size
        self.acoustic_weight = acoustic_weight
        self.acoustic_branch_drop_rate = acoustic_branch_drop_rate

        activation = get_activation(activation_type)
        if audiovisual_layer_type == "upsampling_positionwise":
            audiovisual_layer_class = PositionwiseFeedForward
        else:
            raise ValueError("Support only upsampling positionwise feed forward fusion.")

        # -- audio-visual merge method
        self.merge_method = merge_method
        if merge_method == "concat":
            self.audiovisual_layer = audiovisual_layer_class(
                idim=input_size+input_size,
                hidden_units=hidden_units,
                dropout_rate=dropout_rate,
                activation=activation,
            )
        elif merge_method == "learned_ave":
            # -- -- attention-based pooling for both modalities
            self.acoustic_pooling_proj = torch.nn.Linear(input_size, 1)
            self.visual_pooling_proj = torch.nn.Linear(input_size, 1)

            # -- -- linear projections for calculating merging weights
            self.acoustic_weight_proj = torch.nn.Linear(input_size, 1)
            self.visual_weight_proj = torch.nn.Linear(input_size, 1)

            # -- -- audiovisual merge projection after weighted average
            self.audiovisual_layer = audiovisual_layer_class(
                idim=input_size,
                hidden_units=hidden_units,
                dropout_rate=dropout_rate,
                activation=activation,
            )
        elif merge_method == "fixed_ave":
            assert (
                0.0 <= acoustic_weight <= 1.0
            ), "cgmlp weight should be between 0.0 and 1.0"
            self.audiovisual_layer = audiovisual_layer_class(
                idim=input_size,
                hidden_units=hidden_units,
                dropout_rate=dropout_rate,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknow merge method: {merge_method}")

        self.norm_final = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, audio_pad, audio_masks, video_pad, video_masks, cache=None):
        """Compute audiovisual encoded features.

        Args:
            audio_pad (torch.Tensor): Input audio tensor (#batch, time, size).
            audio_masks (torch.Tensor): Input audio mask (#batch, time).
            video_pad (torch.Tensor): Input video tensor (#batch, time, size).
            video_masks (torch.Tensor): Input video mask (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output audiovisual tensor (#batch, time, size).
            torch.Tensor: Output audiovisual length (#batch,).
        """
        # -- input implementation details
        if cache is not None:
            raise NotImplementedError("cache is not None, which is not tested")

        # -- merge module
        if self.merge_method == "concat":
            audiovisual = self.audiovisual_layer(
                torch.cat([audio_pad, video_pad], dim=-1)
            )

        elif self.merge_method == "learned_ave":
            if (
                self.training
                and self.acoustic_branch_drop_rate > 0
                and torch.rand(1).item() < self.acoustic_branch_drop_rate
            ):
                # -- dropping the acoustic branch
                self.acoustic_weight, self.visual_weight = 0.0, 1.0
            else:
                # -- acoustic branch
                acoustic_score = (
                    self.acoustic_pooling_proj(audio_pad).transpose(1, 2) / self.input_size**0.5
                ) # (batch, 1, time)
                if audio_masks is not None:
                    min_value = float(
                        numpy.finfo(
                            torch.tensor(0, dtype=acoustic_score.dtype).numpy().dtype
                        ).min
                    )
                    acoustic_score = acoustic_score.masked_fill(audio_masks.eq(0), min_value)
                    acoustic_score = torch.softmax(acoustic_score, dim=-1).masked_fill(
                        audio_masks.eq(0), 0.0
                    )
                else:
                    acoustic_score = torch.softmax(acoustic_score, dim=-1)
                acoustic_pooled = torch.matmul(acoustic_score, audio_pad).squeeze(1) # (batch, size)
                acoustic_weight = self.acoustic_weight_proj(acoustic_pooled) # (batch, 1)

                # -- visual branch
                visual_score = (
                    self.visual_pooling_proj(video_pad).transpose(1, 2) / self.input_size**0.5
                ) # (batch, 1, time)
                if video_masks is not None:
                    min_value = float(
                        numpy.finfo(
                            torch.tensor(0, dtype=visual_score.dtype).numpy().dtype
                        ).min
                    )
                    visual_score = visual_score.masked_fill(video_masks.eq(0), min_value)
                    visual_score = torch.softmax(visual_score, dim=-1).masked_fill(
                        video_masks.eq(0), 0.0
                    )
                else:
                    visual_score = torch.softmax(visual_score, dim=-1)
                visual_pooled = torch.matmul(visual_score, video_pad).squeeze(1) # (batch, size)
                visual_weight = self.visual_weight_proj(visual_pooled) # (batch, 1)

                # -- normalize weights of two branches
                merge_weights = torch.softmax(
                    torch.cat([acoustic_weight, visual_weight], dim=-1), dim=-1
                )
                merge_weights = merge_weights.unsqueeze(-1).unsqueeze(
                    -1
                ) # (batch, 2, 1, 1)
                self.acoustic_weight, self.visual_weight = merge_weights[:, 0], merge_weights[:, 1] # (batch, 1, 1)

            # -- audiovisual learnt weighted fusion
            audiovisual = self.audiovisual_layer(
                self.acoustic_weight * audio_pad + self.visual_weight * video_pad
            )
        elif self.merge_method == "fixed_ave":
            audiovisual = self.audiovisual_layer(
                self.acoustic_weight * audio_pad + (1.0 - self.acoustic_weight) * video_pad
            )
        else:
            raise RuntimeError(f"unknown merge method: {self.merge_method}")

        # -- final output block normalisation
        audiovisual = self.norm_final(audiovisual)

        # -- output implementation details
        audiovisual_masks = torch.logical_or(audio_masks, video_masks)
        audiovisual_olens = audiovisual_masks.squeeze(1).sum(1)

        return audiovisual, audiovisual_olens
