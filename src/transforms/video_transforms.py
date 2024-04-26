#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import torchaudio
import torch.nn as nn
from typing import List, Optional, Tuple, Union

class Compose(object):
    """Compose several preprocess together.
    """

    def __init__(self, preprocess):
        """__init__.

        Args:
            preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
        """
        self.preprocess = preprocess

    def __call__(self, video_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video_data (torch.Tensor): input video sequence.

        Returns:
            torch.Tensor: the preprocessed video sequence.
        """
        for p in self.preprocess:
            if p is not None:
                video_data = p(video_data)
        return video_data

class FunctionalModule(torch.nn.Module):
    """Compose several preprocess together.
    """
    def __init__(self, functional):
        """
        Args:
            functional (function): function to convert in a torch.Module.
        """
        super().__init__()
        self.functional = functional

    def forward(self, video_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video_data (torch.Tensor): input video sequence.

        Returns:
            torch.Tensor: the preprocessed video sequence.
        """
        return self.functional(video_data)

class Normalise(object):
    """Normalize a Tensor sequence of images with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video_data (Tensor): input video sequence to be normalised.

        Returns:
            torch.Tensor: Normalised video sequence.
        """
        return (video_data - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)

class TimeMasking(object):
    """Masking n consecutive frames with the mean frame of the sequence.
      We use one mask per second, and for each mask, we mask 0.4 seconds as a maximum
    """

    def __init__(self, fps=25.0, max_frames: Optional[int] = None, max_seconds: Optional[float] = None):
        """__init__.
          Args:
            fps (int): number of frames per second of the video sequence.
            max_frames (int): the maximum number of video frames to mask.
            max_seconds (int): the maximum number of seconds to mask.
        """
        assert max_frames or max_seconds
        self.fps = fps
        self.max_frames = max_frames
        self.max_seconds = max_seconds

    def __call__(self, video_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video_data (torch.Tensor): video sequence data to be masked.

        Returns:
            torch.Tensor: masked video sequence.
        """
        if self.max_frames is None:
            max_frames = int(self.fps * self.max_seconds)
        else:
            max_frames = self.max_frames

        video_length = video_data.shape[0]
        num_seconds = int(video_length / self.fps)
        video_mean_frame = video_data.mean(axis=0)
        for _ in range(num_seconds):
            mask_length = random.randint(0, max_frames)
            if mask_length > 0:
                offset = random.randint(0, video_length - mask_length)
                video_data[offset:(offset+mask_length)] = video_mean_frame

        return video_data

class CenterCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, crop_size):
        """__init__.

        Args:
            crop_size (tuple): size of the cropped video sequence.
        """
        self.crop_size = crop_size

    def __call__(self, video_data):
        """
        Args:
            video_data (torch.Tensor): video sequence to be cropped.

        Returns:
            torch.tensor: video cropped sequence.
        """
        frames, h, w = video_data.shape
        th, tw = self.crop_size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        return video_data[:, delta_h:delta_h+th, delta_w:delta_w+tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.crop_size)


class VideoSpeedRate(object):
    # Copyright 2021 Imperial College London (Pingchuan Ma)
    # Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
    """Subsample/Upsample the number of frames in a sequence.

    """

    def __init__(self, speed_rate=1.0):
        """__init__.

        :param speed_rate: float, the speed rate between the frame rate of \
            the input video and the frame rate used for training.
        """
        self._speed_rate = speed_rate

    def __call__(self, x):
         """
         Args:
             img (numpy.ndarray): sequence to be sampled.
         Returns:
             numpy.ndarray: sampled sequence.
         """
         if self._speed_rate <= 0:
             raise ValueError("speed_rate should be greater than zero.")
         if self._speed_rate == 1.:
             return x
         old_length = x.shape[0]
         new_length = int(old_length / self._speed_rate)
         old_indices = np.arange(old_length)
         new_indices = np.linspace(start=0, stop=old_length, num=new_length, endpoint=False)
         new_indices = list(map(int, new_indices))
         x = x[new_indices]
         return x
