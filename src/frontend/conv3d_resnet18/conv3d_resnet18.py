#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Code mainly based on Pingchuan Ma's implementation,
# see Visual Speech Recognition for Multiple Languages Github's repository

# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import logging
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from espnet2.asr.frontend.abs_frontend import AbsFrontend

from src.frontend.conv3d_resnet18.modules.resnet import ResNet
from src.frontend.conv3d_resnet18.modules.resnet import BasicBlock
from espnet.nets.pytorch_backend.conformer.swish import Swish

# -- auxiliary function to re-shape a 3D tensor to a 2D tensor
def threeD_to_2D_tensor(x):
    """Reshape a 3D tensor to a 2D tensor.

    Args:
        x (torch.Tensor): video stream data (batch, channels, time, height, width).

    Returns:
        torch.Tensor: re-shaped video stream data. (batch*time, channels, height, width).
    """

    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)

    return x.reshape(n_batch*s_time, n_channels, sx, sy)

class Conv3dResNet18(AbsFrontend):
    """Conv3d+ResNet18 module.
    """

    def __init__(self, activation_type="swish"):
        """__init__.
        """
        super(Conv3dResNet18, self).__init__()

        # -- frontend3D
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=64,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False
            ),
            nn.BatchNorm3d(64),
            Swish(),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            )
        )

        # -- trunk2D
        self.trunk = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            activation_type=activation_type,
        )

    def output_size(self) -> int:
        return self.trunk.layer4[1].conv2.out_channels


    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            speech (torch.Tensor): batch of padded video data sequences.
            speech_lengths (torch.Tensor): batch of padded video data lengths.

        Returns:
            torch.Tensor, torch.Tensor: latent visual speech represention, video data lengths.
        """
        # -- including channel dimension
        speech = speech.unsqueeze(1)

        B, C, T, H, W = speech.size()
        speech = self.frontend3D(speech)
        speech = threeD_to_2D_tensor(speech)
        speech = self.trunk(speech)
        speech = speech.view(B, T, speech.size(-1))

        return speech, speech_lengths
