"""Neuron-level target-score evaluator used by TGSR-PINN.

The evaluator collects hidden-layer pre-activation values and their loss
gradients, then combines first-order Taylor sensitivity with pre-activation
variance. The public methods are intentionally small because TGSR-PINN uses
this module inside transfer-time scoring batches.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class NeuroQualityEvaluator:
    """Compute neuron scores for hidden Linear layers."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        base_alpha: float = 0.5,
        alpha_range: float = 0.3,
        combination_method: str = "geometric",
    ):
        self.model = model
        self.device = device
        self.base_alpha = base_alpha
        self.alpha_range = alpha_range
        self.combination_method = combination_method

        valid_methods = {"additive", "geometric", "product", "harmonic"}
        if combination_method not in valid_methods:
            raise ValueError(f"Unknown combination method {combination_method}. Valid: {sorted(valid_methods)}")

        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.layer_records: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.hooks: List[Any] = []
        self._total_layers = self._count_hidden_layers()

    def _count_hidden_layers(self) -> int:
        if hasattr(self.model, "linears"):
            return max(1, len(self.model.linears) - 1)
        linear_modules = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        return max(1, len(linear_modules) - 1)

    def _iter_hidden_linear_layers(self):
        if hasattr(self.model, "linears"):
            for idx, module in enumerate(self.model.linears[:-1]):
                yield idx, module
            return

        linear_modules = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        target_modules = linear_modules[:-1] if len(linear_modules) > 1 else linear_modules
        for idx, module in enumerate(target_modules):
            yield idx, module

    def _register_hooks(self) -> None:
        """Register forward hooks that retain gradients on pre-activations."""

        def get_activation_hook(name: str) -> Callable:
            def hook(module, inputs, output):
                if not torch.is_tensor(output):
                    return

                activation = output.detach()
                self.activations[name] = activation

                if not output.requires_grad:
                    return

                def capture_grad(grad, layer_name=name, stored_activation=activation):
                    grad_detached = grad.detach()
                    self.gradients[layer_name] = grad_detached
                    self.layer_records.setdefault(layer_name, []).append(
                        (stored_activation, grad_detached)
                    )

                output.register_hook(capture_grad)

            return hook

        for idx, module in self._iter_hidden_linear_layers():
            hook = module.register_forward_hook(get_activation_hook(f"layer_{idx}"))
            self.hooks.append(hook)

    def _remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}
        self.layer_records = {}

    def _call_custom_loss(
        self,
        loss_fn: Callable,
        inputs: List[torch.Tensor],
        outputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_tuple = tuple(inputs)
        call_patterns = [
            lambda: loss_fn(model=self.model, inputs=input_tuple, outputs=outputs),
            lambda: loss_fn(self.model, input_tuple, outputs),
            lambda: loss_fn(input_tuple),
        ]
        if outputs is not None:
            call_patterns.append(lambda: loss_fn(outputs))

        last_error = None
        for call in call_patterns:
            try:
                return call()
            except TypeError as exc:
                last_error = exc
        raise last_error if last_error is not None else RuntimeError("Failed to call loss function")

    def _get_layer_alpha(self, layer_idx: int) -> float:
        if self._total_layers <= 1:
            return self.base_alpha + self.alpha_range / 2.0
        depth_ratio = layer_idx / (self._total_layers - 1)
        return self.base_alpha + self.alpha_range * depth_ratio

    def _combine_scores(
        self,
        importance: torch.Tensor,
        variance: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        eps = 1e-6
        if self.combination_method == "additive":
            return alpha * importance + (1.0 - alpha) * variance
        if self.combination_method == "geometric":
            log_s = alpha * torch.log(importance + eps) + (1.0 - alpha) * torch.log(variance + eps)
            return torch.exp(log_s)
        if self.combination_method == "product":
            return importance * variance
        if self.combination_method == "harmonic":
            return 2.0 * importance * variance / (importance + variance + eps)
        raise AssertionError(self.combination_method)

    def compute_neuron_importance(
        self,
        loss_fn: Optional[Callable] = None,
        data_inputs: Optional[List[torch.Tensor]] = None,
        num_samples: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """Return per-neuron scores for each hidden layer."""
        if data_inputs is None:
            if hasattr(self.model, "linears"):
                input_dims = self.model.linears[0].in_features
            else:
                input_dims = 2
            inputs = []
            for _ in range(input_dims):
                tensor = torch.rand(num_samples, 1, device=self.device) * 2.0 - 1.0
                tensor.requires_grad_(True)
                inputs.append(tensor)
        elif isinstance(data_inputs, torch.Tensor):
            inputs = [data_inputs]
        else:
            inputs = list(data_inputs)

        self._register_hooks()
        self.model.train()
        self.model.zero_grad()

        try:
            if loss_fn is not None:
                loss = self._call_custom_loss(loss_fn, inputs)
            else:
                outputs = self.model(*inputs)
                loss = outputs.mean()
            loss.backward()
        except Exception as exc:
            print(f"[ERROR] Neuron scoring failed: {exc}")
            self._remove_hooks()
            return {}

        neuron_scores: Dict[str, np.ndarray] = {}
        for layer_name, records in self.layer_records.items():
            if not records:
                continue
            layer_idx = int(layer_name.split("_")[1])
            activations = torch.cat([activation for activation, _ in records], dim=0)
            gradients = torch.cat([gradient for _, gradient in records], dim=0)

            importance = (activations * gradients).abs().mean(dim=0)
            variance = activations.var(dim=0)

            imp_norm = (importance - importance.mean()) / (importance.std() + 1e-6)
            var_norm = (variance - variance.mean()) / (variance.std() + 1e-6)
            imp_norm = torch.clamp(imp_norm, 0, None)
            var_norm = torch.clamp(var_norm, 0, None)
            imp_norm = imp_norm / (imp_norm.max() + 1e-6)
            var_norm = var_norm / (var_norm.max() + 1e-6)

            alpha = self._get_layer_alpha(layer_idx)
            score = self._combine_scores(imp_norm, var_norm, alpha)
            neuron_scores[layer_name] = score.detach().cpu().numpy()

        self._remove_hooks()
        return neuron_scores

    def _compute_kmeans_threshold(self, scores: np.ndarray) -> float:
        if len(scores) < 2:
            return 0.0

        c1 = float(np.min(scores))
        c2 = float(np.max(scores))
        for _ in range(10):
            dist1 = np.abs(scores - c1)
            dist2 = np.abs(scores - c2)
            mask1 = dist1 < dist2
            mask2 = ~mask1
            new_c1 = float(np.mean(scores[mask1])) if mask1.any() else c1
            new_c2 = float(np.mean(scores[mask2])) if mask2.any() else c2
            if abs(new_c1 - c1) < 1e-6 and abs(new_c2 - c2) < 1e-6:
                break
            c1, c2 = new_c1, new_c2
        return (c1 + c2) / 2.0

    def _suggest_threshold(self, scores: np.ndarray, mode: str = "percentile", value: float = 20.0) -> float:
        if mode == "kmeans":
            return self._compute_kmeans_threshold(scores)
        return float(np.percentile(scores, value))

    def evaluate_layers(self, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        neuron_scores = self.compute_neuron_importance()
        if not neuron_scores:
            return {}

        layer_quality: Dict[str, Dict[str, float]] = {}
        if verbose:
            print(f"\n{'Layer':<12} | {'Mean score':<15} | {'Low-score count':<15} | {'alpha':<8}")
            print("-" * 60)

        for name, scores in neuron_scores.items():
            layer_idx = int(name.split("_")[1])
            alpha = self._get_layer_alpha(layer_idx)
            threshold = self._suggest_threshold(scores, mode="percentile", value=5.0)
            low_count = int((scores < threshold).sum())
            total = len(scores)
            mean_score = float(scores.mean())
            layer_quality[name] = {
                "quality": mean_score,
                "dead_ratio": low_count / total if total else 0.0,
                "alpha": alpha,
            }
            if verbose:
                print(f"{name:<12} | {mean_score:<15.4f} | {low_count}/{total:<10} | {alpha:.2f}")
        return layer_quality

    def suggest_transfer_mask(
        self,
        keep_ratio: Optional[float] = None,
        dynamic_threshold: bool = True,
        loss_fn: Optional[Callable] = None,
        pruning_mode: str = "percentile",
    ) -> Dict[str, np.ndarray]:
        scores = self.compute_neuron_importance(loss_fn=loss_fn)
        masks: Dict[str, np.ndarray] = {}

        for name, score_vec in scores.items():
            n = len(score_vec)
            if dynamic_threshold:
                if pruning_mode == "kmeans":
                    threshold = self._suggest_threshold(score_vec, mode="kmeans")
                else:
                    threshold = float(np.mean(score_vec) - 0.5 * np.std(score_vec))

                mask_dynamic = score_vec > threshold
                mask_topk = np.zeros_like(score_vec, dtype=bool)
                if keep_ratio is not None and keep_ratio > 0:
                    k = max(1, int(n * keep_ratio))
                    mask_topk[np.argsort(score_vec)[-k:]] = True
                masks[name] = np.logical_or(mask_dynamic, mask_topk).astype(float)
            else:
                ratio = keep_ratio if keep_ratio is not None else 0.5
                k = max(1, int(n * ratio))
                mask = np.zeros_like(score_vec)
                mask[np.argsort(score_vec)[-k:]] = 1.0
                masks[name] = mask

        return masks

    def get_blend_weights(
        self,
        loss_fn: Optional[Callable] = None,
        min_weight: float = 0.1,
        max_weight: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        scores = self.compute_neuron_importance(loss_fn=loss_fn)
        blend_weights: Dict[str, np.ndarray] = {}

        for name, score_vec in scores.items():
            s_min = float(np.min(score_vec))
            s_max = float(np.max(score_vec))
            if s_max - s_min > 1e-8:
                normalized = (score_vec - s_min) / (s_max - s_min)
            else:
                normalized = np.ones_like(score_vec) * 0.5
            blend_weights[name] = min_weight + normalized * (max_weight - min_weight)

        return blend_weights
