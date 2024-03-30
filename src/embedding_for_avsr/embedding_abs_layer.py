from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch

class EmbeddingForAVSRAbsLayer(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[Union[Tuple, torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def apply_embed_layer(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def apply_pos_enc(
        self,
        xs_pad: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
