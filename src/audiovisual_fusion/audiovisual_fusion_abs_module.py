from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

class AudioVisualFusionAbsModule(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        audio_pad: torch.Tensor,
        audio_masks: torch.Tensor,
        video_pad: torch.Tensor,
        video_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
