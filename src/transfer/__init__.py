"""TGSR-PINN transfer strategy registry."""

import inspect

from .base_strategy import BaseTransferStrategy
from .tgsr_strategy import TGSRTransferStrategy, tgsr_transfer_weights


STRATEGY_REGISTRY = {
    "tgsr": TGSRTransferStrategy,
}


def get_transfer_strategy(mode: str, device, **kwargs):
    if mode not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown transfer mode: {mode}. "
            f"Available public modes: {list(STRATEGY_REGISTRY.keys())}"
        )

    strategy_class = STRATEGY_REGISTRY[mode]
    signature = inspect.signature(strategy_class.__init__)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )

    if accepts_var_kwargs:
        filtered_kwargs = kwargs
    else:
        allowed_keys = {
            name for name in parameters.keys()
            if name not in ("self", "device")
        }
        filtered_kwargs = {
            key: value for key, value in kwargs.items()
            if key in allowed_keys
        }

    return strategy_class(device, **filtered_kwargs)


__all__ = [
    "BaseTransferStrategy",
    "TGSRTransferStrategy",
    "tgsr_transfer_weights",
    "STRATEGY_REGISTRY",
    "get_transfer_strategy",
]
