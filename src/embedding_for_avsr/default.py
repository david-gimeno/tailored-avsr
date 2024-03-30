# Copyright 2023 David Gimeno-Gómez (PRHLT, UPV)

import logging
from typing import List, Optional, Tuple, Union

import copy
import numpy
import torch
import torch.nn.functional as F
from typeguard import check_argument_types
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from espnet.nets.pytorch_backend.transformer.embedding import (  # noqa: H301
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import Conv2dSubsamplingWOPosEnc

from src.embedding_for_avsr.embedding_abs_layer import EmbeddingForAVSRAbsLayer

class DefaultEmbeddingLayerForAVSR(EmbeddingForAVSRAbsLayer):
    """Default Embedding Layer for AVSR that allow us to temporally align the audio and video data streams before applying positional encoding.

    Args:
        input_size (int): Input dimension.
        output_size (int): Encoder dimension.
        pos_enc_layer_type (str): Positional encoder layer type.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        input_layer: Input embedding alyer type.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate for positional encoding.
        max_pos_emb_len (int): Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        pos_enc_layer_type: str = "rel_pos",
        rel_pos_type: str = "latest",
        input_layer: str = "conv2d",
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        max_pos_emb_len: int = 5000,
    ):
        super().__init__()
        self._output_size = output_size
        self._rel_pos_type = rel_pos_type
        self._pos_enc_layer_type = pos_enc_layer_type

        # -- embedding layer
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsamplingWOPosEnc(
                input_size,
                output_size,
                dropout_rate,
                kernels=[3,3],
                strides=[2,2],
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx)
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = input_layer
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        # -- positional encoding
        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
        elif rel_pos_type == "latest":
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        self.pos_enc = pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad,
        ilens,
    ) -> Tuple[Union[Tuple, torch.Tensor], torch.Tensor]:
        """Forward computatition.

        Args:
            xs_pad (torch.Tensor): Input data tensor.
            ilens (torch.Tensor): Input data length (#batch, time).

        Returns:
            torch.Tensor: Output data tensor w/ pos emb [(#batch, time, size), (1, time, size)].
            torch.Tensor: Output audiovisual length (#batch,).
        """
        # -- masks computation
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        # -- embed layer
        if isinstance(self.embed, Conv2dSubsamplingWOPosEnc):
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        # positional encoding
        xs_pad = self.pos_enc(xs_pad)

        return xs_pad, masks

    def apply_embed_layer(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # -- masks computation
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if isinstance(self.embed, Conv2dSubsamplingWOPosEnc):
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        return xs_pad, masks


    def apply_pos_enc(
        self,
        xs_pad: torch.Tensor,
    ) -> Tuple[Union[Tuple, torch.Tensor], Optional[torch.Tensor]]:

        return self.pos_enc(xs_pad)
