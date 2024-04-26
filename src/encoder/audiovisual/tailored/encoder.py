#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 David Gimeno-Gómez, PRHLT, UPV
# Code mainly based on the Shigeki Karita's implementation
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging
from typing import List, Optional, Tuple, Union

import numpy
import torch
from typeguard import check_argument_types

from src.encoder.audiovisual.audiovisual_abs_encoder import AudioVisualAbsEncoder

from espnet2.asr.layers.cgmlp import ConvolutionalGatingMLP
from espnet2.asr.layers.fastformer import FastSelfAttention
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet2.asr.ctc import CTC
from src.encoder.audiovisual.tailored.encoder_layer import TailoredEncoderLayer
from src.audiovisual_fusion.audiovisual_fusion_abs_module import AudioVisualFusionAbsModule

class TailoredEncoder(AudioVisualAbsEncoder):
    """Unified and Tailored AudioVisual Branchformer-based Encoder module.

    """
    def __init__(
        self,
        embed_pos_enc_layer_type,
        embed_rel_pos_type,
        output_size=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=12,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        acoustic_branch_drop_rate=0.0,
        attention_layer_type="rel_selfattn",
        positionwise_layer_type="linear",
        ffn_activation_type="swish",
        cgmlp_linear_units=2048,
        cgmlp_conv_kernel=31,
        gate_activation="identity",
        use_linear_after_conv=False,
        acoustic_use_attn: List[bool] = [True]*12,
        visual_use_attn: List[bool] = [False]*12,
        macaron=True,
        zero_triu=False,
        normalize_before=True,
        ignore_id=-1,
        interctc_use_conditioning: bool = False,
        audiovisual_interctc_conditioning: bool = False,
        interctc_layer_idx: List[int] = [],
        stochastic_depth_rate=0.0,
        max_pos_emb_len: int = 5000,
    ):
        """Construct an Encoder object."""
        assert check_argument_types()
        super().__init__()
        self.ignore_id = ignore_id
        self._output_size = output_size

        # -- sanity checks
        if embed_rel_pos_type == "legacy":
            if attention_layer_type == "rel_selfattn":
                attention_layer_type = "legacy_rel_selfattn"
        elif embed_rel_pos_type == "latest":
             assert attention_layer_type != "legacy_rel_selfattn"
             assert embed_pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown embed_rel_pos_type: " + embed_rel_pos_type)

        if embed_pos_enc_layer_type == "rel_pos":
            assert attention_layer_type == "rel_selfattn"
        elif embed_pos_enc_layer_type == "legacy_rel_pos":
            assert attention_layer_type == "legacy_rel_selfattn"
        elif embed_pos_enc_layer_type == "legacy_rel_pos":
            assert attention_layer_type == "legacy_rel_selfattn"
        elif embed_pos_enc_layer_type in ["abs_pos", "scaled_abs_pos"]:
            assert attention_layer_type == "fast_selfattn"
        else:
            raise ValueError("unknown pos_enc_layer: " + embed_pos_enc_layer_type)

        self.normalize_before = normalize_before
        activation = get_activation(ffn_activation_type)

        # -- modality embeddings
        self.modality_encoding = torch.nn.Embedding(2, output_size)
        self.modality_to_id = {"audio": 0, "video": 1}

        # -- positionwise feed forward network module
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type is None:
            logging.warning("no macaron ffn")
        else:
            raise ValueError("Support only linear.")

        # -- multiheaded self attention module
        if attention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif attention_layer_type == "legacy_rel_selfattn":
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif attention_layer_type == "rel_selfattn":
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        elif attention_layer_type == "fast_selfattn":
            encoder_selfattn_layer = FastSelfAttention
            encoder_selfattn_layer_args = (
                output_size,
                attention_heads,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown attention_layer_typer: " + attention_layer_type)

        ## -- convolutional gating MLP module
        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
        )

        # -- stochastic depth rates
        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks
        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        # -- acoustic branch drop rates
        if isinstance(acoustic_branch_drop_rate, float):
            acoustic_branch_drop_rate = [acoustic_branch_drop_rate] * num_blocks
        if len(acoustic_branch_drop_rate) != num_blocks:
            raise ValueError(
                f"Length of acoustic_branch_drop_rate ({len(acoustic_branch_drop_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        # -- acoustic and visual tailored blocks
        assert len(acoustic_use_attn) == num_blocks, f"Lenght of acoustic_use_attn ({len(acoustic_use_attn)}) should be equal to num_blocks ({num_blocks})"
        assert len(visual_use_attn) == num_blocks, f"Lenght of visual_use_attn ({len(visual_use_attn)}) should be equal to num_blocks ({num_blocks})"

        # -- encoder blocks
        self.encoders = repeat(
            num_blocks,
            lambda lnum: TailoredEncoderLayer(
                output_size,
                positionwise_layer(*positionwise_layer_args) if macaron else None,
                encoder_selfattn_layer(*encoder_selfattn_layer_args) if acoustic_use_attn[lnum] else None,
                cgmlp_layer(*cgmlp_layer_args) if not acoustic_use_attn[lnum] else None,
                encoder_selfattn_layer(*encoder_selfattn_layer_args) if visual_use_attn[lnum] else None,
                cgmlp_layer(*cgmlp_layer_args) if not visual_use_attn[lnum] else None,
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                acoustic_branch_drop_rate[lnum],
                stochastic_depth_rate[lnum],
            ),
        )

        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.audiovisual_interctc_conditioning = audiovisual_interctc_conditioning
        assert (
            not(self.interctc_use_conditioning == False and self.audiovisual_interctc_conditioning == True)
        ), "Audio-Visual InterCTC conditioning only can be applied if interctc_use_conditioning is set to True."
        self.conditioning_layer = None

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        audio_pad: torch.Tensor,
        audio_masks: torch.Tensor,
        video_pad: torch.Tensor,
        video_masks: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
        audiovisual_fusion: AudioVisualFusionAbsModule = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            audio_pad (Union[Tuple, torch.Tensor]): Input audio tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            audio_masks (torch.Tensor): Input audio mask (#batch, time).
            video_pad (Union[Tuple, torch.Tensor]): Input video tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            video_masks (torch.Tensor): Input video mask (#batch, time).
            prev_states (torch.Tensor): Not to be used now.
            ctc (CTC): Intermediate CTC module.
            audiovisual_fusion (AudioVisualFusionAbsModule) : Intermediate Audio-Visual fusion module.
        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output mask tensor (#batch, time).
            torch.Tensor: Not to be used now.
        """
        # -- modality encoding
        if isinstance(audio_pad, tuple):
            x, pos_emb = audio_pad
            x = x + self.modality_encoding(torch.tensor(self.modality_to_id["audio"]).to(x.device))
            audio_pad = (x, pos_emb)
        else:
            audio_pad = audio_pad + self.modality_encoding(torch.tensor(self.modality_to_id["audio"]).to(audio_pad.device))

        if isinstance(video_pad, tuple):
            x, pos_emb = video_pad
            x = x + self.modality_encoding(torch.tensor(self.modality_to_id["video"]).to(x.device))
            video_pad = (x, pos_emb)
        else:
            video_pad = video_pad + self.modality_encoding(torch.tensor(self.modality_to_id["video"]).to(video_pad.device))

        # -- tailored encoder blocks
        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            audio_pad, audio_masks, video_pad, video_masks = self.encoders(audio_pad, audio_masks, video_pad, video_masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                audio_pad, audio_masks, video_pad, video_masks = encoder_layer(audio_pad, audio_masks, video_pad, video_masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_audio_out = audio_pad
                    if isinstance(encoder_audio_out, tuple):
                        encoder_audio_out = encoder_audio_out[0]

                    encoder_video_out = video_pad
                    if isinstance(encoder_video_out, tuple):
                        encoder_video_out = encoder_video_out[0]

                    # -- -- intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_audio_out = self.after_norm(encoder_audio_out)
                        encoder_video_out = self.after_norm(encoder_video_out)

                    # -- -- audiovisual intermediate CTC loss
                    encoder_audiovisual_out, _ = audiovisual_fusion(
                        encoder_audio_out,
                        audio_masks,
                        encoder_video_out,
                        video_masks,
                    )
                    intermediate_outs.append((layer_idx + 1, encoder_audiovisual_out))

                    # -- -- intermediate CTC contidioning
                    if self.interctc_use_conditioning:
                        if self.audiovisual_interctc_conditioning:
                            ctc_audiovisual_out = ctc.softmax(encoder_audiovisual_out)
                            ctc_audio_out = ctc_audiovisual_out
                            ctc_video_out = ctc_audiovisual_out
                        else:
                            ctc_audio_out = ctc.softmax(encoder_audio_out)
                            ctc_video_out = ctc.softmax(encoder_video_out)

                        if isinstance(audio_pad, tuple):
                            audio, audio_pos_emb = audio_pad
                            audio = audio + self.conditioning_layer(ctc_audio_out)
                            audio_pad = (audio, audio_pos_emb)
                        else:
                            audio_pad = audio_pad + self.conditioning_layer(ctc_audio_out)

                        if isinstance(video_pad, tuple):
                            video, video_pos_emb = video_pad
                            video = video + self.conditioning_layer(ctc_video_out)
                            video_pad = (video, video_pos_emb)
                        else:
                            video_pad = video_pad + self.conditioning_layer(ctc_video_out)

        # -- output implementation details
        if isinstance(audio_pad, tuple):
            audio_pad = audio_pad[0]
        if isinstance(video_pad, tuple):
            video_pad = video_pad[0]

        if self.normalize_before:
            audio_pad = self.after_norm(audio_pad)
            video_pad = self.after_norm(video_pad)

        if len(intermediate_outs) > 0:
            return (audio_pad, intermediate_outs), audio_masks, video_pad, video_masks, None
        return audio_pad, audio_masks, video_pad, video_masks, None
