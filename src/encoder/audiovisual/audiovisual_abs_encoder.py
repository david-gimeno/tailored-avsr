from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

class AudioVisualAbsEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        audio_pad: torch.Tensor,
        audio_ilens: torch.Tensor,
        video_pad: torch.Tensor,
        video_ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
