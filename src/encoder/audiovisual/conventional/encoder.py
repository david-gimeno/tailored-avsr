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

from espnet2.asr.ctc import CTC
from src.encoder.audiovisual.audiovisual_abs_encoder import AudioVisualAbsEncoder

from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)

from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from src.encoder.branchformer.encoder import MyBranchformerEncoder

from src.audiovisual_fusion.audiovisual_fusion_abs_module import AudioVisualFusionAbsModule

class ConventionalEncoder(AudioVisualAbsEncoder):
    """Wrapper Encoder allows to combine two different encoder branches.

    """
    def __init__(
        self,
        input_size,
        acoustic_encoder_conf,
        visual_encoder_conf,
        output_size: int = 256,
        embed_pos_enc_layer_type: str = "rel_pos",
        embed_rel_pos_type: str = "latest",
        interctc_use_conditioning: bool = False,
        audiovisual_interctc_conditioning: bool = False,
        interctc_layer_idx: List[int] = [],
    ):
        """Construct an Encoder object."""
        super().__init__()

        # -- sanity checks
        assert (
            embed_pos_enc_layer_type
            == acoustic_encoder_conf["pos_enc_layer_type"]
            == visual_encoder_conf["pos_enc_layer_type"]
        ), (embed_pos_enc_layer_type, visual_encoder_conf["pos_enc_layer_type"], visual_encoder_conf["pos_enc_layer_type"])

        assert (
            embed_rel_pos_type
            == acoustic_encoder_conf["rel_pos_type"]
            == visual_encoder_conf["rel_pos_type"]
        ), (embed_rel_pos_type, visual_encoder_conf["rel_pos_type"], visual_encoder_conf["res_pos_type"])

        # -- building encoders
        acoustic_encoder_class = self.get_encoder_class(acoustic_encoder_conf["encoder_class_type"])
        visual_encoder_class = self.get_encoder_class(visual_encoder_conf["encoder_class_type"])

        del acoustic_encoder_conf["encoder_class_type"]
        del visual_encoder_conf["encoder_class_type"]

        self.acoustic_encoder = acoustic_encoder_class(
            input_size=input_size,
            output_size=output_size,
            **acoustic_encoder_conf,
        )
        self.visual_encoder = visual_encoder_class(
            input_size=input_size,
            output_size=output_size,
            **visual_encoder_conf,
        )

        # -- sanity checks
        assert (
            len(self.acoustic_encoder.encoders) == len(self.visual_encoder.encoders)
        ), "Both encoders must have the same number of blocks."
        assert (
            self.acoustic_encoder.output_size() == self.visual_encoder.output_size()
        ), "Output size should be the same in both wrapped encoders."
        assert (
            self.acoustic_encoder.embed is None and self.visual_encoder.embed is None
        ), "The embedding layers of both encoders should be None."
        assert (
            len(self.acoustic_encoder.interctc_layer_idx) == 0 and len(self.visual_encoder.interctc_layer_idx) == 0
        ), "InterCTC loss must be defined in the WrapperEncoder."
        assert (
            self.acoustic_encoder.interctc_use_conditioning == False and self.visual_encoder.interctc_use_conditioning == False
        ), "InterCTC conditioning must be defined in the WrapperEncoder."

        num_blocks = len(self.acoustic_encoder.encoders)
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
        return self.acoustic_encoder.output_size()

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
        # -- embedding layers were applied before in model architeture encode method
        # -- wrapped encoder blocks
        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            audio_pad, audio_masks = self.acoustic_encoder.encoders(audio_pad, audio_masks)
            video_pad, video_masks = self.visual_encoder.encoders(video_pad, video_masks)
        else:
            for layer_idx, (acoustic_encoder_layer, visual_encoder_layer) in enumerate(zip(self.acoustic_encoder.encoders, self.visual_encoder.encoders)):
                audio_pad, audio_masks = acoustic_encoder_layer(audio_pad, audio_masks)
                video_pad, video_masks = visual_encoder_layer(video_pad, video_masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_audio_out = audio_pad
                    if isinstance(encoder_audio_out, tuple):
                        encoder_audio_out = encoder_audio_out[0]

                    encoder_video_out = video_pad
                    if isinstance(encoder_video_out, tuple):
                        encoder_video_out = encoder_video_out[0]

                    # -- -- intermediate outputs are also normalized
                    if self.acoustic_encoder.normalize_before:
                        encoder_audio_out = self.acoustic_encoder.after_norm(encoder_audio_out)
                    if self.visual_encoder.normalize_before:
                        encoder_video_out = self.visual_encoder.after_norm(encoder_video_out)

                    # -- -- audiovisual intermediate CTC loss
                    encoder_audiovisual_out, _ = audiovisual_fusion(
                        encoder_audio_out,
                        audio_masks,
                        encoder_video_out,
                        video_masks,
                    )
                    intermediate_outs.append((layer_idx + 1, encoder_audiovisual_out))

                    # -- -- intermediate CTC conditioning
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

        if self.acoustic_encoder.normalize_before:
            audio_pad = self.acoustic_encoder.after_norm(audio_pad)
        if self.visual_encoder.normalize_before:
            video_pad = self.visual_encoder.after_norm(video_pad)

        if len(intermediate_outs) > 0:
            return (audio_pad, intermediate_outs), audio_masks, video_pad, video_masks, None
        return audio_pad, audio_masks, video_pad, video_masks, None

    def get_encoder_class(self, encoder_class_type):
        if encoder_class_type == "conformer":
            return ConformerEncoder
        elif encoder_class_type == "my_e_branchformer":
            return MyEBranchformerEncoder
        elif encoder_class_type == "simt_conformer":
            return ConformerEncoderSimT
        elif encoder_class_type == "simt_my_e_branchformer":
            return MyEBranchformerEncoderSimT
        else:
            raise ValueError("unknown encoder_class_type: " + encoder_class_type)
