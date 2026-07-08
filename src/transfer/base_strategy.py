"""Base class for public TGSR-PINN transfer strategies."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTransferStrategy(ABC):
    """Abstract transfer strategy interface."""

    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def transfer(
        self,
        target_model: nn.Module,
        source_path: str,
        **kwargs,
    ) -> None:
        """Modify the target model using a source checkpoint."""
        raise NotImplementedError

    def _load_source_state(self, source_path: str) -> dict:
        """Load a source model state dict."""
        return torch.load(source_path, map_location=self.device, weights_only=False)
