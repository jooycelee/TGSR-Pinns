import os
import sys
import tempfile

import pytest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import FCN


class TestModelBasics:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def source_model(self, device):
        model = FCN([2, 50, 50, 1])
        model.to(device)
        return model

    def test_fcn_structure(self, source_model):
        assert hasattr(source_model, "linears")
        assert len(source_model.linears) == 3

    def test_save_load_model(self, source_model, device):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.pth")
            torch.save(source_model.state_dict(), path)
            loaded_state = torch.load(path, map_location=device)
            assert set(loaded_state.keys()) == set(source_model.state_dict().keys())

    def test_model_forward(self, source_model, device):
        x = torch.rand(10, 1, device=device)
        t = torch.rand(10, 1, device=device)
        output = source_model(x, t)
        assert output.shape == (10, 1)

    def test_weight_shapes(self, source_model):
        state = source_model.state_dict()
        assert state["linears.0.weight"].shape == (50, 2)
        assert state["linears.0.bias"].shape == (50,)
        assert state["linears.1.weight"].shape == (50, 50)
        assert state["linears.2.weight"].shape == (1, 50)


class TestStateDictCompatibility:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_strict_load_same_architecture(self, device):
        source = FCN([2, 50, 50, 1])
        target = FCN([2, 50, 50, 1])
        target.load_state_dict(source.state_dict(), strict=True)

        for (_, source_param), (_, target_param) in zip(source.named_parameters(), target.named_parameters()):
            assert torch.allclose(source_param, target_param)

    def test_partial_load_different_output_width(self, device):
        source = FCN([2, 50, 50, 1])
        target = FCN([2, 50, 50, 2])

        state = source.state_dict()
        del state["linears.2.weight"]
        del state["linears.2.bias"]

        target.load_state_dict(state, strict=False)
