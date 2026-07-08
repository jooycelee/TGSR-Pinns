"""Paper-facing ablation score transforms for TGSR-PINN.

These helpers intentionally operate on neuron scores before the existing
GMM/rank soft-decay machinery. Default TGSR-PINN behavior is unchanged unless
the caller explicitly selects one of these ablation modes.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch.nn as nn


VALID_SCORE_ABLATION_MODES = {
    "none",
    "",
    "score_shuffled_tgsr",
    "random_soft_decay",
    "magnitude_soft_decay",
}


def normalize_ablation_mode(mode: Optional[str]) -> str:
    """Normalize user/environment input for score ablation mode."""
    normalized = (mode or "none").strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "default": "none",
        "tgsr": "none",
        "score_shuffled": "score_shuffled_tgsr",
        "shuffled": "score_shuffled_tgsr",
        "random": "random_soft_decay",
        "magnitude": "magnitude_soft_decay",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_SCORE_ABLATION_MODES:
        valid = ", ".join(sorted(m for m in VALID_SCORE_ABLATION_MODES if m))
        raise ValueError(f"Unknown score ablation mode '{mode}'. Valid modes: {valid}")
    return "none" if normalized == "" else normalized


def _copy_scores(layer_scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        layer_name: np.asarray(scores, dtype=float).copy()
        for layer_name, scores in layer_scores.items()
    }


def _normalize_unit_interval(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x.copy()
    finite = np.isfinite(x)
    if not finite.all():
        x = np.where(finite, x, 0.0)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if np.isclose(x_min, x_max):
        return np.ones_like(x, dtype=float)
    return (x - x_min) / (x_max - x_min)


def transform_scores_for_ablation(
    layer_scores: Dict[str, np.ndarray],
    mode: Optional[str],
    rng: np.random.Generator,
    magnitude_scores: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Return score dictionary transformed for a score ablation mode.

    Lower scores are interpreted by the existing TGSR-PINN code as lower
    adaptation evidence. The transforms therefore preserve that convention.
    """
    normalized_mode = normalize_ablation_mode(mode)
    if normalized_mode == "none":
        return _copy_scores(layer_scores), {"mode": "none"}

    transformed: Dict[str, np.ndarray] = {}
    meta = {"mode": normalized_mode, "layers": {}}

    for layer_name, scores in layer_scores.items():
        scores_arr = np.asarray(scores, dtype=float)
        if normalized_mode == "score_shuffled_tgsr":
            transformed_scores = rng.permutation(scores_arr)
        elif normalized_mode == "random_soft_decay":
            transformed_scores = rng.random(scores_arr.shape)
        elif normalized_mode == "magnitude_soft_decay":
            if magnitude_scores is None or layer_name not in magnitude_scores:
                raise ValueError(f"Magnitude scores missing for {layer_name}")
            magnitude_arr = np.asarray(magnitude_scores[layer_name], dtype=float)
            if magnitude_arr.shape != scores_arr.shape:
                raise ValueError(
                    f"Magnitude score shape mismatch for {layer_name}: "
                    f"expected {scores_arr.shape}, got {magnitude_arr.shape}"
                )
            transformed_scores = _normalize_unit_interval(magnitude_arr)
        else:  # pragma: no cover - normalize_ablation_mode guards this branch.
            raise AssertionError(normalized_mode)

        transformed[layer_name] = np.asarray(transformed_scores, dtype=float)
        meta["layers"][layer_name] = {
            "n": int(scores_arr.size),
            "mean": float(np.mean(transformed[layer_name])) if scores_arr.size else 0.0,
            "std": float(np.std(transformed[layer_name])) if scores_arr.size else 0.0,
        }

    return transformed, meta


def collect_incoming_weight_norm_scores(
    model: nn.Module,
    layer_names: Iterable[str],
) -> Dict[str, np.ndarray]:
    """Collect incoming weight-row L2 norms for hidden Linear layers.

    The final output Linear layer is excluded, matching the TGSR-PINN neuron
    scoring convention.
    """
    requested = list(layer_names)
    linear_modules = [module for module in model.modules() if isinstance(module, nn.Linear)]
    hidden_modules = linear_modules[:-1]
    scores: Dict[str, np.ndarray] = {}

    for idx, module in enumerate(hidden_modules):
        layer_name = f"layer_{idx}"
        if layer_name not in requested:
            continue
        weight = module.weight.detach().float().cpu().numpy()
        scores[layer_name] = np.linalg.norm(weight, axis=1)

    missing = [layer_name for layer_name in requested if layer_name not in scores]
    if missing:
        raise ValueError(f"Could not collect magnitude scores for layers: {missing}")
    return scores
