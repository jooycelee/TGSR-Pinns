import numpy as np
import pytest
import torch

from src.transfer.tgsr_strategy import TGSRTransferStrategy


def _strategy(monkeypatch, mapping=None, force_rank_only=None):
    if mapping is not None:
        monkeypatch.setenv("TGSR_SOFT_DECAY_MAPPING", mapping)
    else:
        monkeypatch.delenv("TGSR_SOFT_DECAY_MAPPING", raising=False)

    if force_rank_only is not None:
        monkeypatch.setenv("TGSR_GMM_FORCE_RANK_ONLY", force_rank_only)
    else:
        monkeypatch.delenv("TGSR_GMM_FORCE_RANK_ONLY", raising=False)

    return TGSRTransferStrategy(torch.device("cpu"))


def test_default_soft_decay_mapping_matches_existing_cubic_curve(monkeypatch):
    strategy = _strategy(monkeypatch)

    _, scales = strategy._build_gmm_soft_reset_plan(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    assert strategy.soft_decay_mapping == "cubic"
    assert np.allclose(scales, np.array([1.0, 0.98125, 0.85, 0.79375, 0.4]))


def test_linear_soft_decay_mapping_uses_simple_interval_scaling(monkeypatch):
    strategy = _strategy(monkeypatch, mapping="linear")

    _, scales = strategy._build_gmm_soft_reset_plan(np.array([0.0, 0.5, 1.0]))

    assert np.allclose(scales, np.array([1.0, 0.7, 0.4]))


def test_sigmoid_soft_decay_mapping_is_monotone_and_endpoint_normalized(monkeypatch):
    strategy = _strategy(monkeypatch, mapping="sigmoid")

    _, scales = strategy._build_gmm_soft_reset_plan(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    assert scales[0] == pytest.approx(1.0)
    assert scales[-1] == pytest.approx(0.4)
    assert np.all(np.diff(scales) < 0.0)


def test_hard_threshold_soft_decay_mapping_switches_at_midpoint(monkeypatch):
    strategy = _strategy(monkeypatch, mapping="hard_threshold")

    _, scales = strategy._build_gmm_soft_reset_plan(np.array([0.49, 0.5, 0.51]))

    assert np.allclose(scales, np.array([1.0, 0.4, 0.4]))


def test_force_rank_only_gmm_uses_rank_fallback_for_every_layer(monkeypatch):
    strategy = _strategy(monkeypatch, force_rank_only="1")

    strategy._identify_low_quality_neurons({"layer_0": np.array([0.1, 0.2, 0.3, 0.4])})

    debug = strategy._last_pruning_debug["layer_0"]
    assert strategy.gmm_force_rank_only is True
    assert debug["gmm_mode"] == "rank-only"
    assert debug["use_two_component"] is False
