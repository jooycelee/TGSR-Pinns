"""Utilities for transferring trainable physics-side neural branches.

This module is intentionally narrow. It supports function-valued inverse
experiments where the unknown coefficient/source is represented by a small
network inside the physics module, for example ``physics.q_net``.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


def resolve_physics_checkpoint(source_model_path: str) -> Optional[str]:
    """Return the sibling physics checkpoint for a source model checkpoint."""
    if not source_model_path:
        return None
    candidate = Path(source_model_path).with_name("physics_best.pth")
    return str(candidate) if candidate.exists() else None


def copy_named_physics_branches(
    target_physics: nn.Module,
    source_physics_path: str,
    branch_names: Iterable[str] = ("q_net",),
    device: Optional[torch.device] = None,
) -> Dict[str, int]:
    """Copy matching branch parameters from a saved physics state dict.

    Parameters are copied only when both the key and tensor shape match. The
    function returns a small report so scripts and tests can verify what
    happened without parsing console output.
    """
    report = {"transferred": 0, "skipped": 0, "missing": 0}
    if target_physics is None or not source_physics_path:
        report["missing"] += 1
        return report

    checkpoint = Path(source_physics_path)
    if not checkpoint.exists():
        report["missing"] += 1
        return report

    map_location = device if device is not None else "cpu"
    source_state = torch.load(str(checkpoint), map_location=map_location, weights_only=False)
    if isinstance(source_state, dict) and "state_dict" in source_state:
        source_state = source_state["state_dict"]

    target_state = target_physics.state_dict()
    branch_prefixes = tuple(f"{name}." for name in branch_names)

    with torch.no_grad():
        for key, source_value in source_state.items():
            if not key.startswith(branch_prefixes):
                continue
            if key not in target_state:
                report["skipped"] += 1
                continue
            if target_state[key].shape != source_value.shape:
                report["skipped"] += 1
                continue
            target_state[key].copy_(source_value.to(target_state[key].device))
            report["transferred"] += 1

    return report
