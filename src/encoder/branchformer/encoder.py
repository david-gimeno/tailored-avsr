#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging
from typing import List, Optional, Tuple, Union

import numpy
import torch
from typeguard import check_argument_types

from src.ctc.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.layers.cgmlp import ConvolutionalGatingMLP
from espnet2.asr.layers.fastformer import FastSelfAttention
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
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
    Conv1dSubsampling2,
    Conv1dSubsampling3,
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.nets_utils import rename_state_dict

from src.encoder.branchformer.encoder_layer import MyBranchformerEncoderLayer

class MyBranchformerEncoder(AbsEncoder):
    """Branchformer Encoder module.

    """
    def __init__(
        self,
        input_size=256,
        output_size=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        cgmlp_linear_units=2048,
        cgmlp_conv_kernel=31,
        cgmlp_weight=0.5,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        attn_branch_drop_rate=0.0,
        input_layer="conv3dresnet18",
        rel_pos_type="latest",
        pos_enc_layer_type="rel_pos",
        attention_layer_type="rel_selfattn",
        positionwise_layer_type="linear",
        ffn_activation_type="relu",
        merge_method="learned_ave",
        gate_activation="identity",
        ignore_id=-1,
        use_attn=True,
        use_cgmlp=True,
        macaron=True,
        zero_triu=False,
        normalize_before=True,
        use_linear_after_conv=False,
        interctc_use_conditioning: bool = False,
        interctc_layer_idx: List[int] = [],
        stochastic_depth_rate=0.0,
        max_pos_emb_len: int = 5000,
    ):
        """Construct an Encoder object."""
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if attention_layer_type == "rel_selfattn":
                attention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert attention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert attention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert attention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        # -- -- embedding layer
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer in ["conv1d", "conv3dresnet18"]:
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(512, output_size),
                pos_enc_class(output_size, positional_dropout_rate)
            )
        elif input_layer == "conv1d2":
            self.embed = Conv1dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv1d3":
            self.embed = Conv1dSubsampling3(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None:
            # if input_size == output_size:
            #     self.embed = torch.nn.Sequential(
            #         pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
            #     )
            # else:
            #     self.embed = torch.nn.Linear(input_size, output_size)
            self.embed = None
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.normalize_before = normalize_before
        activation = get_activation(ffn_activation_type)

        ## -- PositionWise Feed Forward Network Module
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

        ## -- Self-Attention Module
        if attention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif attention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
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
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        elif attention_layer_type == "fast_selfattn":
            assert pos_enc_layer_type in ["abs_pos", "scaled_abs_pos"]
            encoder_selfattn_layer = FastSelfAttention
            encoder_selfattn_layer_args = (
                output_size,
                attention_heads,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + attention_layer_type)

        ## -- cgMLP Module
        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
        )

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks
        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        if isinstance(cgmlp_weight, float):
            cgmlp_weight = [cgmlp_weight] * num_blocks
        if len(cgmlp_weight) != num_blocks:
            raise ValueError(
                f"Length of cgmlp_weight ({len(cgmlp_weight)}) should be equal to "
                f"num_blocks ({num_blocks})"
            )

        if isinstance(attn_branch_drop_rate, float):
            attn_branch_drop_rate = [attn_branch_drop_rate] * num_blocks
        if len(attn_branch_drop_rate) != num_blocks:
            raise ValueError(
                f"Length of attn_branch_drop_rate ({len(attn_branch_drop_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: MyBranchformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args) if use_attn else None,
                cgmlp_layer(*cgmlp_layer_args) if use_cgmlp else None,
                positionwise_layer(*positionwise_layer_args) if macaron else None,
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                merge_method,
                cgmlp_weight[lnum],
                attn_branch_drop_rate[lnum],
                stochastic_depth_rate[lnum],
            ),
        )

        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

    def output_size(self) -> int:
        return self._output_size

    def forward(self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
        max_layer: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
            ctc (CTC): Intermediate CTC module.
            max_layer (int): Layer depth below which InterCTC is applied.
        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv1dSubsampling2)
            or isinstance(self.embed, Conv1dSubsampling3)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            if max_layer is not None and 0 <= max_layer < len(self.encoders):
                for layer_idx, encoder_layer in enumerate(self.encoders):
                    xs_pad, masks = encoder_layer(xs_pad, masks)
                    if layer_idx >= max_layer:
                        break
            else:
                xs_pad, masks = self.encoders(xs_pad, masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x = x + self.conditioning_layer(ctc_out)
                            xs_pad = (x, pos_emb)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None
