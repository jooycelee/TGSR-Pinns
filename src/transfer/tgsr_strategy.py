"""
PINN V2 自适应两阶段迁移策略 (TGSR Two-Phase Transfer Strategy)

============================================================
核心创新: 两阶段评估 + 目标任务导向的神经元筛选
============================================================

与 SmartTransferStrategy 的区别:
- Smart: 在源模型上评估神经元质量，迁移前筛选
- TGSR: 先全量迁移，在目标任务上训练后再评估并修剪

理论优势:
- 评估的是神经元对*目标任务*的贡献，而非源任务
- 能检测到"负迁移"神经元并将其重置
- 更符合迁移学习的本质：利用有用的，丢弃有害的

流程:
    阶段1: 全量迁移 + 短期预热 (warmup_epochs)
    阶段2: 在目标任务上评估神经元质量
    阶段3: 重置低质量神经元 + 继续训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import Optional, Dict, Callable, Tuple, Any, List
import numpy as np

from .base_strategy import BaseTransferStrategy
from .score_ablation_modes import (
    collect_incoming_weight_norm_scores,
    normalize_ablation_mode,
    transform_scores_for_ablation,
)
from .physics_branch import copy_named_physics_branches, resolve_physics_checkpoint
from ..utils.neuro_evaluator import NeuroQualityEvaluator


VALID_SOFT_DECAY_MAPPINGS = {
    'cubic',
    'linear',
    'sigmoid',
    'hard_threshold',
}

VALID_LAYER_PROTECTION_MODES = {
    'depth_aware',
    'none',
    'uniform_matched',
}


class TGSRTransferStrategy(BaseTransferStrategy):
    """
    自适应两阶段迁移策略 (TGSR Two-Phase Transfer)

    创新点:
    1. 两阶段评估: 先迁移后评估，而非迁移前评估
    2. 目标任务导向: 使用目标任务的损失函数评估神经元重要性
    3. 动态修剪: 根据预热阶段表现决定保留/重置哪些神经元
    4. 负迁移检测: 识别并消除对目标任务有害的迁移特征

    Attributes:
        warmup_epochs: 预热训练轮数 (用于评估)
        prune_threshold: 修剪阈值 (基于百分位数)
        reset_mode: 重置模式 ('reinit' 或 'scale')
    """

    def __init__(
        self,
        device: torch.device,
        warmup_epochs_cap: int = 200,      # upper cap, actual warmup still follows warmup_ratio
        warmup_ratio: float = 0.2,         # 默认取主训练 epoch 的 20%
        prune_percentile: float = 20.0,
        reset_mode: str = 'scale',
        min_keep_ratio: float = 0.5,
        pruning_mode: str = 'gmm',         # 'gmm' / 'kmeans' / 'percentile'
        warmup_optimizer: str = 'lbfgs',   # 'adam' / 'lbfgs'，lbfgs 参数反演更准确
        warmup_l2_lambda: float = 0.0,     # L2 正则化强度 (借鉴 Ψ-NN)
        progressive_warmup: bool = False,  # 默认关闭，只有显式指定才开启
        warmup_epochs: Optional[int] = None,
        transfer_physics_branch: bool = False,
        reset_physics_branch: bool = False,
        physics_branch_names: Tuple[str, ...] = ("q_net",),
        score_alpha: Optional[float] = None,       # 评分 α (None=使用默认0.5)
        shield_beta: Optional[float] = None,       # 层级保护 β_h (None=使用默认0.55)
    ):
        """
        初始化策略

        Args:
            device: 计算设备
            warmup_epochs: 预热训练轮数 (默认 200)
            prune_percentile: 修剪的百分位阈值 (默认 20，即修剪最差的 20%)
            reset_mode: 重置模式
                - 'reinit': 完全重新初始化低质量神经元
                - 'scale': 缩放低质量神经元权重 (更保守)
            min_keep_ratio: 最低保留比例 (默认 0.5，至少保留 50% 神经元)
            pruning_mode: 神经元筛选模式
                - 'gmm': 基于 GMM 后验坏度的连续 soft gate (当前默认)
                - 'kmeans': 基于聚类的分级 soft reset
                - 'percentile': 基于固定百分位阈值的硬尾部筛选
            warmup_optimizer: 预热阶段优化器
                - 'lbfgs': 使用 L-BFGS (默认，适合参数反演问题，二阶优化更精准)
                - 'adam': 使用 Adam (适合快速验证)
            warmup_l2_lambda: Warmup 阶段 L2 正则化系数
                - 0.0: 不使用 (默认)
                - > 0: 促进权重聚拢，改善后续修剪质量 (借鉴 Ψ-NN)
                - 推荐值: 1e-4 ~ 1e-3
        """
        super().__init__(device)
        if warmup_epochs is not None:
            warmup_epochs_cap = warmup_epochs

        self.warmup_epochs_cap = int(warmup_epochs_cap)
        # Backward-compatible alias for older call sites and experiment configs.
        self.warmup_epochs = self.warmup_epochs_cap
        self.warmup_ratio = warmup_ratio
        self.prune_percentile = prune_percentile
        self.reset_mode = reset_mode
        self.min_keep_ratio = min_keep_ratio
        self.pruning_mode = pruning_mode
        self.score_num_batches = 3
        self.kmeans_scope = 'layerwise'
        self.kmeans_bad_ratio = 0.35
        self.soft_reset_min_scale = 0.4
        self.soft_reset_max_scale = 0.85
        self.gmm_shallow_protect = 0.55
        self.gmm_fallback_strength = 0.75
        self.gmm_force_rank_only = False
        self.layer_protection_mode = 'depth_aware'
        self.soft_decay_mapping = 'cubic'
        self.warmup_optimizer = warmup_optimizer.lower()
        self.warmup_l2_lambda = warmup_l2_lambda
        self.progressive_warmup = bool(progressive_warmup)
        self.transfer_physics_branch = bool(transfer_physics_branch)
        self.reset_physics_branch = bool(reset_physics_branch)
        self.physics_branch_names = tuple(physics_branch_names)
        self.score_alpha = score_alpha       # None = use default
        self.shield_beta = shield_beta       # None = use default
        if self.shield_beta is not None:
            self.gmm_shallow_protect = self.shield_beta
        self.score_ablation_mode = 'none'
        self.score_ablation_seed = 0
        self._progressive_warmup_mode = 'manual'
        self._last_reset_scales: Dict[str, np.ndarray] = {}
        self._cached_score_points_batches: Optional[List[Any]] = None
        self._last_pruning_debug: Dict[str, Dict[str, Any]] = {}
        self.warmup_capture_frames = 0
        self._warmup_neuron_snapshots: List[Dict[str, Any]] = []
        self._warmup_capture_epoch_points: set[int] = set()
        self._warmup_capture_total_epochs = 0

        # 支持通过环境变量覆盖超参数 (用于敏感性分析扫描)
        import os
        env_percentile = os.environ.get('PINN_PRUNE_PERCENTILE')
        if env_percentile is not None:
            self.prune_percentile = float(env_percentile)
            print(f"[ENV] PINN_PRUNE_PERCENTILE={self.prune_percentile}")
        env_warmup = os.environ.get('PINN_WARMUP_EPOCHS')
        if env_warmup is not None:
            self.warmup_epochs_cap = int(env_warmup)
            self.warmup_epochs = self.warmup_epochs_cap
            print(f"[ENV] PINN_WARMUP_EPOCHS={self.warmup_epochs_cap}")
        env_warmup_ratio = os.environ.get('PINN_WARMUP_RATIO')
        if env_warmup_ratio is not None:
            self.warmup_ratio = float(env_warmup_ratio)
            print(f"[ENV] PINN_WARMUP_RATIO={self.warmup_ratio}")
        env_progressive = os.environ.get('TGSR_PROGRESSIVE_WARMUP')
        if env_progressive is not None:
            self.progressive_warmup = env_progressive.lower() in ('1', 'true', 'yes')
            self._progressive_warmup_mode = 'env'
            print(f"[ENV] TGSR_PROGRESSIVE_WARMUP={self.progressive_warmup}")
        env_score_batches = os.environ.get('TGSR_SCORE_NUM_BATCHES')
        if env_score_batches is not None:
            self.score_num_batches = max(1, int(env_score_batches))
            print(f"[ENV] TGSR_SCORE_NUM_BATCHES={self.score_num_batches}")
        env_kmeans_scope = os.environ.get('TGSR_KMEANS_SCOPE')
        if env_kmeans_scope is not None:
            self.kmeans_scope = env_kmeans_scope.lower()
            print(f"[ENV] TGSR_KMEANS_SCOPE={self.kmeans_scope}")
        env_score_alpha = os.environ.get('PINN_SCORE_ALPHA')
        if env_score_alpha is not None:
            self.score_alpha = float(env_score_alpha)
            print(f"[ENV] PINN_SCORE_ALPHA={self.score_alpha}")
        env_shield_beta = os.environ.get('PINN_SHIELD_BETA')
        if env_shield_beta is not None:
            self.shield_beta = float(env_shield_beta)
            self.gmm_shallow_protect = self.shield_beta
            print(f"[ENV] PINN_SHIELD_BETA={self.shield_beta}")
        if env_kmeans_scope is not None:
            self.kmeans_scope = env_kmeans_scope.lower()
            print(f"[ENV] TGSR_KMEANS_SCOPE={self.kmeans_scope}")
        env_kmeans_bad_ratio = os.environ.get('TGSR_KMEANS_BAD_RATIO')
        if env_kmeans_bad_ratio is not None:
            self.kmeans_bad_ratio = float(env_kmeans_bad_ratio)
            print(f"[ENV] TGSR_KMEANS_BAD_RATIO={self.kmeans_bad_ratio}")
        env_gmm_shallow_protect = os.environ.get('TGSR_GMM_SHALLOW_PROTECT')
        if env_gmm_shallow_protect is not None:
            self.gmm_shallow_protect = float(env_gmm_shallow_protect)
            print(f"[ENV] TGSR_GMM_SHALLOW_PROTECT={self.gmm_shallow_protect}")
        env_gmm_fallback_strength = os.environ.get('TGSR_GMM_FALLBACK_STRENGTH')
        if env_gmm_fallback_strength is not None:
            self.gmm_fallback_strength = float(env_gmm_fallback_strength)
            print(f"[ENV] TGSR_GMM_FALLBACK_STRENGTH={self.gmm_fallback_strength}")
        env_gmm_force_rank_only = os.environ.get('TGSR_GMM_FORCE_RANK_ONLY')
        if env_gmm_force_rank_only is not None:
            self.gmm_force_rank_only = env_gmm_force_rank_only.strip().lower() in ('1', 'true', 'yes', 'on')
            print(f"[ENV] TGSR_GMM_FORCE_RANK_ONLY={self.gmm_force_rank_only}")
        env_layer_protection_mode = os.environ.get('TGSR_LAYER_PROTECTION_MODE')
        if env_layer_protection_mode is not None:
            self.layer_protection_mode = self._normalize_layer_protection_mode(env_layer_protection_mode)
            print(f"[ENV] TGSR_LAYER_PROTECTION_MODE={self.layer_protection_mode}")
        env_soft_decay_mapping = os.environ.get('TGSR_SOFT_DECAY_MAPPING')
        if env_soft_decay_mapping is not None:
            self.soft_decay_mapping = self._normalize_soft_decay_mapping(env_soft_decay_mapping)
            print(f"[ENV] TGSR_SOFT_DECAY_MAPPING={self.soft_decay_mapping}")
        env_warmup_capture_frames = os.environ.get('TGSR_WARMUP_CAPTURE_FRAMES')
        if env_warmup_capture_frames is not None:
            self.warmup_capture_frames = max(0, int(env_warmup_capture_frames))
            print(f"[ENV] TGSR_WARMUP_CAPTURE_FRAMES={self.warmup_capture_frames}")
        env_score_ablation = (
            os.environ.get('TGSR_SCORE_ABLATION_MODE')
            or os.environ.get('TGSR_ABLATION_MODE')
        )
        if env_score_ablation is not None:
            self.score_ablation_mode = normalize_ablation_mode(env_score_ablation)
            print(f"[ENV] TGSR_SCORE_ABLATION_MODE={self.score_ablation_mode}")
        env_score_ablation_seed = os.environ.get('TGSR_SCORE_ABLATION_SEED')
        if env_score_ablation_seed is not None:
            self.score_ablation_seed = int(env_score_ablation_seed)
            print(f"[ENV] TGSR_SCORE_ABLATION_SEED={self.score_ablation_seed}")

        # 存储评估结果 (用于后续分析)
        self.evaluation_results = {}

    @staticmethod
    def _normalize_soft_decay_mapping(mapping: str) -> str:
        """Normalize the soft-decay mapping selected for ablation studies."""
        normalized = (mapping or 'cubic').strip().lower().replace('-', '_')
        aliases = {
            'default': 'cubic',
            'piecewise_cubic': 'cubic',
            'piecewise': 'cubic',
            'hard': 'hard_threshold',
            'threshold': 'hard_threshold',
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in VALID_SOFT_DECAY_MAPPINGS:
            valid = ', '.join(sorted(VALID_SOFT_DECAY_MAPPINGS))
            raise ValueError(f"Unknown TGSR_SOFT_DECAY_MAPPING '{mapping}'. Valid: {valid}")
        return normalized

    @staticmethod
    def _normalize_layer_protection_mode(mode: str) -> str:
        """Normalize the layer-protection mode selected for diagnostic ablations."""
        normalized = (mode or 'depth_aware').strip().lower().replace('-', '_')
        aliases = {
            'default': 'depth_aware',
            'depth': 'depth_aware',
            'depthaware': 'depth_aware',
            'no': 'none',
            'off': 'none',
            'disabled': 'none',
            'uniform': 'uniform_matched',
            'matched': 'uniform_matched',
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in VALID_LAYER_PROTECTION_MODES:
            valid = ', '.join(sorted(VALID_LAYER_PROTECTION_MODES))
            raise ValueError(f"Unknown TGSR_LAYER_PROTECTION_MODE '{mode}'. Valid: {valid}")
        return normalized

    def transfer(
        self,
        target_model: nn.Module,
        source_path: str,
        physics_engine: Optional[Any] = None,
        trainer: Optional[Any] = None,
        **kwargs
    ) -> None:
        """
        执行自适应两阶段迁移

        Args:
            target_model: 目标模型 (将被修改)
            source_path: 源模型权重文件路径
            physics_engine: 物理引擎 (用于采样和计算残差)
            trainer: 训练器实例 (用于预热阶段)
        """
        print(f"\n{'='*60}")
        print(f"  [TGSR] 自适应两阶段迁移策略")
        print(f"{'='*60}")
        self._cached_score_points_batches = None
        self._warmup_neuron_snapshots = []
        self._warmup_capture_epoch_points = set()
        self._warmup_capture_total_epochs = 0

        runtime_warmup_epochs, runtime_warmup_optimizer = self._resolve_runtime_warmup(
            target_model, trainer
        )

        prog_str = ", 渐进式Warmup" if self.progressive_warmup else ""
        if self.pruning_mode == 'gmm':
            prune_desc = (
                "GMM soft gate "
                f"(shallow_protect={self.gmm_shallow_protect:.2f}, "
                f"fallback_strength={self.gmm_fallback_strength:.2f}, "
                f"mapping={self.soft_decay_mapping}"
                f"{', rank_only=True' if self.gmm_force_rank_only else ''})"
            )
        elif self.pruning_mode == 'kmeans':
            prune_desc = (
                f"KMeans soft reset (scope={self.kmeans_scope}, "
                f"bad_ratio={self.kmeans_bad_ratio:.2f})"
            )
        else:
            prune_desc = f"Percentile tail reset (P{self.prune_percentile:.1f})"

        print(
            f"[配置] 预热轮数: {runtime_warmup_epochs} (cap={self.warmup_epochs_cap}, ratio={self.warmup_ratio:.2f}), "
            f"筛选模式: {prune_desc}, 预热优化器: {runtime_warmup_optimizer.upper()}{prog_str}"
        )

        # ========== 阶段 0: 加载源模型 ==========
        print(f"\n[阶段 0] 加载源模型: {source_path}")
        source_state = self._load_source_weights(source_path)
        if source_state is None:
            print("[错误] 无法加载源模型，回退到随机初始化")
            return

        # ========== 阶段 1: 全量迁移 + 预热 ==========
        print(f"\n[阶段 1] 全量迁移 + 预热训练 ({runtime_warmup_epochs} epochs)")

        # 1.1 全量迁移
        self._full_transfer(target_model, source_state)
        physics_transfer_report: Dict[str, Any] = {}
        if self.transfer_physics_branch and trainer is not None:
            source_physics_path = resolve_physics_checkpoint(source_path)
            physics_transfer_report = self._transfer_physics_branch(
                target_physics=trainer.physics,
                source_physics_path=source_physics_path,
            )
            print(
                "[TGSR] physics-branch transfer: "
                f"transferred={physics_transfer_report.get('transferred', 0)}, "
                f"skipped={physics_transfer_report.get('skipped', 0)}, "
                f"missing={physics_transfer_report.get('missing', 0)}"
            )

        # 1.2 保存迁移后的初始状态 (用于后续比较)
        initial_state = copy.deepcopy(target_model.state_dict())
        pre_warmup_params = self._snapshot_physics_params(trainer)

        # 1.3 预热训练 (如果提供了 trainer)
        warmup_loss_history = []
        if trainer is not None and runtime_warmup_epochs > 0:
            warmup_loss_history = self._warmup_training(
                target_model, trainer, runtime_warmup_epochs, runtime_warmup_optimizer
            )
        else:
            print("[警告] 未提供 trainer，跳过预热阶段")
        post_warmup_params = self._snapshot_physics_params(trainer)

        # ========== 阶段 2: 评估神经元对目标任务的贡献 ==========
        print(f"\n[阶段 2] 评估神经元对目标任务的贡献")

        neuron_scores, prune_masks, prune_ratio = self._evaluate_and_prune(
            target_model=target_model,
            trainer=trainer,
        )

        # ========== 阶段 3: 修剪/重置低质量神经元 ==========
        print(f"\n[阶段 3] 修剪低质量神经元 (模式: {self.reset_mode})")

        if prune_masks:
            self._prune_neurons(target_model, initial_state, prune_masks)
        else:
            print("[警告] 本次未执行修剪")
        model_pruning_debug = copy.deepcopy(self._last_pruning_debug)
        physics_branch_prune_ratio = 0.0
        physics_branch_pruning_debug: Dict[str, Any] = {}
        if self.reset_physics_branch and trainer is not None and hasattr(trainer.physics, 'q_net'):
            print("\n[TGSR] Function-branch selective soft reset")
            _, physics_prune_masks, physics_branch_prune_ratio = self._evaluate_and_prune_physics_branch(
                branch_module=trainer.physics.q_net,
                trainer=trainer,
            )
            physics_branch_pruning_debug = copy.deepcopy(self._last_pruning_debug)
            if physics_prune_masks:
                self._prune_neurons(trainer.physics.q_net, None, physics_prune_masks)
            else:
                print("[WARN] No function-branch neurons selected for soft reset.")
        post_prune_params = self._snapshot_physics_params(trainer)

        # 保存评估结果
        self.evaluation_results = {
            'neuron_scores': neuron_scores,
            'prune_masks': prune_masks,
            'warmup_loss': warmup_loss_history,
            'prune_ratio': prune_ratio,
            'pruning_debug': self._to_serializable(model_pruning_debug),
            'physics_branch_transfer': self._to_serializable(physics_transfer_report),
            'physics_branch_prune_ratio': physics_branch_prune_ratio,
            'physics_branch_pruning_debug': self._to_serializable(physics_branch_pruning_debug),
        }
        self._record_transfer_metadata(
            trainer=trainer,
            runtime_warmup_epochs=runtime_warmup_epochs,
            runtime_warmup_optimizer=runtime_warmup_optimizer,
            warmup_loss_history=warmup_loss_history,
            pre_warmup_params=pre_warmup_params,
            post_warmup_params=post_warmup_params,
            post_prune_params=post_prune_params,
            prune_ratio=prune_ratio,
            pruning_debug=model_pruning_debug,
        )
        self._record_physics_branch_metadata(
            trainer=trainer,
            transfer_report=physics_transfer_report,
            prune_ratio=physics_branch_prune_ratio,
            pruning_debug=physics_branch_pruning_debug,
        )

        print(f"\n[完成] 自适应迁移完成!")
        print(f"{'='*60}\n")

    def _load_source_weights(self, source_path: str) -> Optional[Dict]:
        """加载源模型权重"""
        try:
            state_dict = torch.load(source_path, map_location=self.device, weights_only=False)
            return state_dict
        except Exception as e:
            print(f"[错误] 加载源模型失败: {e}")
            return None

    def _transfer_physics_branch(
        self,
        target_physics: Optional[nn.Module],
        source_physics_path: Optional[str],
    ) -> Dict[str, int]:
        """Copy trainable physics-side branches, e.g. q_net for function inversion."""
        return copy_named_physics_branches(
            target_physics=target_physics,
            source_physics_path=source_physics_path or "",
            branch_names=self.physics_branch_names,
            device=self.device,
        )

    def _full_transfer(self, target_model: nn.Module, source_state: Dict) -> None:
        """执行全量迁移 (处理维度不匹配)"""
        target_state = target_model.state_dict()

        transferred = 0
        skipped = 0

        with torch.no_grad():
            for name, param in source_state.items():
                if name in target_state:
                    if param.shape == target_state[name].shape:
                        target_state[name].copy_(param)
                        transferred += 1
                    else:
                        # 处理维度不匹配 (如 1D->2D)
                        if 'linears.0.weight' in name:
                            # 输入层处理
                            self._handle_input_layer_mismatch(
                                target_state[name], param
                            )
                            transferred += 1
                        else:
                            skipped += 1
                            print(f"  [跳过] {name}: 形状不匹配 {param.shape} vs {target_state[name].shape}")

        print(f"  -> 成功迁移 {transferred} 个参数, 跳过 {skipped} 个")

    def _handle_input_layer_mismatch(
        self,
        target_weight: torch.Tensor,
        source_weight: torch.Tensor
    ) -> None:
        """处理输入层维度不匹配"""
        # 复制公共维度
        min_in = min(target_weight.shape[1], source_weight.shape[1])
        target_weight[:, :min_in] = source_weight[:, :min_in]
        print(f"  [维度适配] 复制了前 {min_in} 个输入维度")

    def _compute_l2_reg(self, model: nn.Module) -> torch.Tensor:
        """计算 L2 正则化项 (借鉴 Ψ-NN 的权重聚拢思想)"""
        l2_reg = torch.tensor(0.0, device=self.device)
        for param in model.parameters():
            l2_reg = l2_reg + param.pow(2).sum()
        return l2_reg

    def _to_serializable(self, value: Any) -> Any:
        """Convert tensors / numpy values into JSON-safe Python objects."""
        if torch.is_tensor(value):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: self._to_serializable(val) for key, val in value.items()}
        if isinstance(value, tuple):
            return [self._to_serializable(item) for item in value]
        if isinstance(value, list):
            return [self._to_serializable(item) for item in value]
        return value

    def _snapshot_physics_params(self, trainer: Optional[Any]) -> Dict[str, Any]:
        """Capture current inverse-parameter values/errors for transfer-stage bookkeeping."""
        if trainer is None or not hasattr(trainer, 'physics') or trainer.physics is None:
            return {}
        if not hasattr(trainer.physics, 'get_param_errors'):
            return {}

        try:
            return self._to_serializable(trainer.physics.get_param_errors())
        except Exception as exc:
            return {'snapshot_error': str(exc)}

    def _record_transfer_metadata(
        self,
        trainer: Optional[Any],
        runtime_warmup_epochs: int,
        runtime_warmup_optimizer: str,
        warmup_loss_history: List[float],
        pre_warmup_params: Dict[str, Any],
        post_warmup_params: Dict[str, Any],
        post_prune_params: Dict[str, Any],
        prune_ratio: float,
        pruning_debug: Dict[str, Any],
    ) -> None:
        """Persist tgsr-transfer bookkeeping into logger history before main training starts."""
        if trainer is None or not hasattr(trainer, 'logger') or trainer.logger is None:
            return

        history = getattr(trainer.logger, 'history', None)
        if not isinstance(history, dict):
            return

        history['tgsr_transfer'] = {
            'recorded_epoch0_is_post_warmup': True,
            'runtime_warmup_epochs': int(runtime_warmup_epochs),
            'runtime_warmup_optimizer': str(runtime_warmup_optimizer),
            'warmup_loss': self._to_serializable(warmup_loss_history),
            'pre_warmup_params': self._to_serializable(pre_warmup_params),
            'post_warmup_params': self._to_serializable(post_warmup_params),
            'post_prune_params': self._to_serializable(post_prune_params),
            'prune_ratio': float(prune_ratio),
            'pruning_mode': self.pruning_mode,
            'reset_mode': self.reset_mode,
            'score_ablation_mode': self.score_ablation_mode,
            'score_ablation_seed': int(self.score_ablation_seed),
            'score_ablation_meta': self._to_serializable(
                self.evaluation_results.get('score_ablation_meta', {})
            ),
            'layer_pruning_debug': self._to_serializable(pruning_debug),
            'warmup_neuron_snapshots': self._to_serializable(self._warmup_neuron_snapshots),
        }
        if self.pruning_mode == 'gmm':
            history['tgsr_transfer']['gmm_settings'] = {
                'gmm_shallow_protect': float(self.gmm_shallow_protect),
                'gmm_fallback_strength': float(self.gmm_fallback_strength),
                'soft_reset_min_scale': float(self.soft_reset_min_scale),
                'soft_reset_max_scale': float(self.soft_reset_max_scale),
                'soft_decay_mapping': self.soft_decay_mapping,
                'gmm_force_rank_only': bool(self.gmm_force_rank_only),
                'layer_protection_mode': self.layer_protection_mode,
            }

    def _record_physics_branch_metadata(
        self,
        trainer: Optional[Any],
        transfer_report: Dict[str, Any],
        prune_ratio: float,
        pruning_debug: Dict[str, Any],
    ) -> None:
        """Append function-branch transfer/reset metadata to tgsr history."""
        if trainer is None or not hasattr(trainer, 'logger') or trainer.logger is None:
            return
        history = getattr(trainer.logger, 'history', None)
        if not isinstance(history, dict):
            return
        tgsr_history = history.setdefault('tgsr_transfer', {})
        tgsr_history['physics_branch'] = {
            'enabled_transfer': bool(self.transfer_physics_branch),
            'enabled_reset': bool(self.reset_physics_branch),
            'branch_names': list(self.physics_branch_names),
            'transfer_report': self._to_serializable(transfer_report),
            'prune_ratio': float(prune_ratio),
            'layer_pruning_debug': self._to_serializable(pruning_debug),
        }

    def _resolve_runtime_warmup(
        self,
        target_model: nn.Module,
        trainer: Optional[Any]
    ) -> Tuple[int, str]:
        """Tie warmup length to main training epochs while preserving optimizer choice."""
        runtime_warmup_epochs = self.warmup_epochs_cap
        runtime_warmup_optimizer = self.warmup_optimizer
        total_train_epochs = None

        if trainer is not None:
            train_cfg = trainer.config.get('training', {})
            total_train_epochs = int(train_cfg.get('epochs_adam', 0)) + int(train_cfg.get('epochs_lbfgs', 0))
            if total_train_epochs > 0:
                ratio_epochs = max(1, int(np.ceil(total_train_epochs * self.warmup_ratio)))
                original_epochs = runtime_warmup_epochs
                runtime_warmup_epochs = min(runtime_warmup_epochs, ratio_epochs)
                if runtime_warmup_epochs != original_epochs:
                    print(
                        "[TGSR] warmup tied to main training epochs "
                        f"({total_train_epochs} * {self.warmup_ratio:.2f} -> {runtime_warmup_epochs})"
                    )

        return runtime_warmup_epochs, runtime_warmup_optimizer

    def _collect_warmup_params(
        self,
        model: nn.Module,
        trainer: Any,
        trainable_model_only: bool = False
    ) -> List[nn.Parameter]:
        """Collect warmup parameters from both network and physics modules."""
        if trainable_model_only:
            params = [p for p in model.parameters() if p.requires_grad]
        else:
            params = list(model.parameters())

        if trainer is not None and hasattr(trainer, 'physics') and trainer.physics is not None:
            params.extend([p for p in trainer.physics.parameters() if p.requires_grad])

        # Remove duplicates while preserving order.
        deduped = []
        seen = set()
        for param in params:
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            deduped.append(param)
        return deduped

    def _clone_points(self, points: Any) -> Any:
        """Clone tensors while preserving requires_grad flags."""
        if torch.is_tensor(points):
            cloned = points.detach().clone().to(self.device)
            cloned.requires_grad_(points.requires_grad)
            return cloned
        if isinstance(points, tuple):
            return tuple(self._clone_points(item) for item in points)
        if isinstance(points, list):
            return [self._clone_points(item) for item in points]
        if isinstance(points, dict):
            return {key: self._clone_points(value) for key, value in points.items()}
        return copy.deepcopy(points)

    def _extract_evaluation_inputs(self, points: Tuple[Any, Any, List[Any], Tuple[Any, Any]]) -> Optional[List[torch.Tensor]]:
        """Extract real target-domain inputs for evaluator fallback."""
        if not points:
            return None

        pde_inputs = points[0]
        if pde_inputs is None:
            return None
        if torch.is_tensor(pde_inputs):
            return [pde_inputs]
        if isinstance(pde_inputs, (tuple, list)):
            return list(pde_inputs)
        return None

    def _get_cached_score_points_batches(self, trainer: Any) -> List[Any]:
        """Cache fixed scoring batches within one transfer for stable neuron evaluation."""
        expected = max(1, self.score_num_batches)
        if self._cached_score_points_batches is None or len(self._cached_score_points_batches) != expected:
            self._cached_score_points_batches = [
                self._clone_points(trainer._sample_all()) for _ in range(expected)
            ]
            print(f"[INFO] Cached {expected} fixed scoring batch(es) for this transfer.")
        return self._cached_score_points_batches

    def _build_warmup_capture_schedule(self, total_epochs: int) -> set[int]:
        """Build sparse checkpoint epochs for optional warm-up neuron snapshots."""
        if self.warmup_capture_frames <= 0 or total_epochs <= 0:
            return set()

        frame_count = min(total_epochs + 1, max(2, self.warmup_capture_frames))
        epoch_points = {
            int(round(value))
            for value in np.linspace(0, total_epochs, num=frame_count, endpoint=True)
        }
        epoch_points.add(0)
        epoch_points.add(total_epochs)
        return {point for point in epoch_points if 0 <= point <= total_epochs}

    def _sorted_layer_names(self, layer_names: List[str]) -> List[str]:
        """Sort layer names like layer_0, layer_1, ... in numeric order."""
        def sort_key(name: str) -> Tuple[int, str]:
            try:
                return int(name.split('_')[1]), name
            except (IndexError, ValueError):
                return 10**9, name

        return sorted(layer_names, key=sort_key)

    def _capture_warmup_neuron_snapshot(
        self,
        model: nn.Module,
        trainer: Any,
        warmup_epoch: int,
        total_epochs: int,
        label: str,
        loss_value: Optional[float] = None,
    ) -> None:
        """
        Optionally capture layer-wise neuron badness snapshots during warm-up.

        Red/blue style post-processing can later use gate strength:
            gate_strength = 1 - reset_scale
        """
        if trainer is None or self.warmup_capture_frames <= 0:
            return
        if warmup_epoch not in self._warmup_capture_epoch_points:
            return

        neuron_scores, prune_masks, prune_ratio = self._evaluate_and_prune(
            target_model=model,
            trainer=trainer,
        )
        if not neuron_scores:
            return

        snapshot_layers: Dict[str, Any] = {}
        for layer_name in self._sorted_layer_names(list(neuron_scores.keys())):
            scores = np.asarray(neuron_scores[layer_name], dtype=float)
            mask = np.asarray(prune_masks.get(layer_name, np.zeros_like(scores)), dtype=float)
            reset_scales = np.asarray(
                self._last_reset_scales.get(layer_name, np.ones_like(scores)),
                dtype=float,
            )
            snapshot_layers[layer_name] = {
                'scores': self._to_serializable(scores),
                'prune_mask': self._to_serializable(mask),
                'reset_scales': self._to_serializable(reset_scales),
                'gate_strength': self._to_serializable(1.0 - reset_scales),
            }

        snapshot = {
            'epoch': int(warmup_epoch),
            'total_epochs': int(total_epochs),
            'progress': float(warmup_epoch / max(1, total_epochs)),
            'label': str(label),
            'loss_value': float(loss_value) if loss_value is not None else None,
            'prune_ratio': float(prune_ratio),
            'layers': snapshot_layers,
        }
        self._warmup_neuron_snapshots.append(snapshot)
        print(
            f"      [Warmup Snapshot] epoch={warmup_epoch}/{total_epochs}, "
            f"layers={len(snapshot_layers)}, prune_ratio={prune_ratio:.1f}%"
        )

    def _maybe_capture_warmup_snapshot(
        self,
        model: nn.Module,
        trainer: Any,
        global_epoch: int,
        loss_value: Optional[float],
        label: str,
    ) -> None:
        """Capture a warm-up snapshot if the epoch is in the sparse schedule."""
        if global_epoch not in self._warmup_capture_epoch_points:
            return
        self._capture_warmup_neuron_snapshot(
            model=model,
            trainer=trainer,
            warmup_epoch=global_epoch,
            total_epochs=self._warmup_capture_total_epochs,
            label=label,
            loss_value=loss_value,
        )

    def _warmup_training(
        self,
        model: nn.Module,
        trainer: Any,
        epochs: int,
        warmup_optimizer: Optional[str] = None
    ) -> list:
        """
        预热训练阶段

        根据 warmup_optimizer 配置选择优化器:
        - 'adam': 使用 Adam (快速但可能引入参数偏差)
        - 'lbfgs': 使用 L-BFGS (二阶优化，更精确但更慢)

        可选 L2 正则化 (warmup_l2_lambda > 0):
        借鉴 Ψ-NN 论文，L2 正则化迫使权重聚拢，
        使有用/无用权重形成更清晰的二分类，改善后续修剪质量。

        可选渐进式 warm-up (progressive_warmup = True):
        Phase 1: 冻结浅层，仅训练最后 2 层 (epochs // 2)
        Phase 2: 全部解冻，训练所有层 (剩余 epochs)
        这保护了浅层通用特征，借鉴 Progressive (ULMFiT) 的思想。
        """
        optimizer_name = (warmup_optimizer or self.warmup_optimizer).lower()
        l2_info = f", L2 λ={self.warmup_l2_lambda}" if self.warmup_l2_lambda > 0 else ""
        prog_info = ", 渐进式" if self.progressive_warmup else ""
        print(f"  开始预热训练 (优化器: {optimizer_name.upper()}{l2_info}{prog_info})...")

        model.train()
        self._warmup_capture_total_epochs = int(max(0, epochs))
        self._warmup_capture_epoch_points = self._build_warmup_capture_schedule(self._warmup_capture_total_epochs)
        if self._warmup_capture_epoch_points:
            self._capture_warmup_neuron_snapshot(
                model=model,
                trainer=trainer,
                warmup_epoch=0,
                total_epochs=self._warmup_capture_total_epochs,
                label="pre-warmup",
                loss_value=None,
            )

        if self.progressive_warmup:
            # ===== 渐进式 Warm-up =====
            loss_history = self._progressive_warmup_training(model, trainer, epochs, optimizer_name)
        elif optimizer_name == 'lbfgs':
            # ===== 标准 L-BFGS 预热 =====
            loss_history = self._lbfgs_warmup_with_params(
                model,
                trainer,
                epochs,
                self._collect_warmup_params(model, trainer),
                "LBFGS",
                capture_epoch_offset=0,
                capture_total_epochs=self._warmup_capture_total_epochs,
            )
        else:
            # ===== 标准 Adam 预热 =====
            loss_history = self._adam_warmup_with_params(
                model,
                trainer,
                epochs,
                self._collect_warmup_params(model, trainer),
                "Adam",
                capture_epoch_offset=0,
                capture_total_epochs=self._warmup_capture_total_epochs,
            )

        print(f"  预热完成! 最终 Loss: {loss_history[-1]:.6f}")
        return loss_history

    def _progressive_warmup_training(
        self,
        model: nn.Module,
        trainer: Any,
        epochs: int,
        warmup_optimizer: Optional[str] = None
    ) -> list:
        """
        渐进式 Warm-up: 分阶段解冻层

        Phase 1: 冻结浅层, 仅训练最后 n_unfreeze 层 (epochs // 2)
        Phase 2: 全部解冻, 训练所有层 (剩余 epochs)

        原理: Progressive 策略在参数反演上表现优异，
        因为它保护了浅层的通用特征表示。将此思想融入
        TGSR 的 warm-up 阶段，使 Taylor importance
        评估在更稳定的特征基础上进行。
        """
        optimizer_name = (warmup_optimizer or self.warmup_optimizer).lower()
        # 确定模型层数和解冻数量
        n_layers = len(model.linears) if hasattr(model, 'linears') else 0
        n_unfreeze_phase1 = min(2, n_layers)  # Phase 1: 最后 2 层

        epochs_phase1 = epochs // 2
        epochs_phase2 = epochs - epochs_phase1

        loss_history = []

        if n_layers > 0 and n_unfreeze_phase1 < n_layers:
            # ===== Phase 1: 冻结浅层 =====
            print(f"    [Phase 1] 冻结前 {n_layers - n_unfreeze_phase1} 层, "
                  f"训练最后 {n_unfreeze_phase1} 层 ({epochs_phase1} epochs)")

            # 冻结所有层
            for param in model.parameters():
                param.requires_grad = False

            # 解冻最后 n_unfreeze_phase1 层
            for i, layer in enumerate(model.linears):
                if i >= n_layers - n_unfreeze_phase1:
                    for param in layer.parameters():
                        param.requires_grad = True

            # Phase 1 训练
            phase1_history = self._single_phase_warmup(
                model,
                trainer,
                epochs_phase1,
                "Phase1",
                optimizer_name,
                capture_epoch_offset=0,
                capture_total_epochs=epochs,
            )
            loss_history.extend(phase1_history)

            # ===== Phase 2: 全部解冻 =====
            print(f"    [Phase 2] 全部解冻, 训练所有层 ({epochs_phase2} epochs)")
            for param in model.parameters():
                param.requires_grad = True

            phase2_history = self._single_phase_warmup(
                model,
                trainer,
                epochs_phase2,
                "Phase2",
                optimizer_name,
                capture_epoch_offset=epochs_phase1,
                capture_total_epochs=epochs,
            )
            loss_history.extend(phase2_history)
        else:
            # 模型不支持逐层操作，回退到标准warm-up
            print(f"    [回退] 模型无 linears 属性，使用标准 warm-up")
            if optimizer_name == 'lbfgs':
                loss_history = self._lbfgs_warmup_with_params(
                    model,
                    trainer,
                    epochs,
                    self._collect_warmup_params(model, trainer),
                    "LBFGS",
                    capture_epoch_offset=0,
                    capture_total_epochs=epochs,
                )
            else:
                loss_history = self._adam_warmup_with_params(
                    model,
                    trainer,
                    epochs,
                    self._collect_warmup_params(model, trainer),
                    "Adam",
                    capture_epoch_offset=0,
                    capture_total_epochs=epochs,
                )

        return loss_history

    def _single_phase_warmup(
        self,
        model: nn.Module,
        trainer: Any,
        epochs: int,
        phase_label: str = "",
        warmup_optimizer: Optional[str] = None,
        capture_epoch_offset: int = 0,
        capture_total_epochs: Optional[int] = None,
    ) -> list:
        """单阶段 warm-up 训练 (仅训练 requires_grad=True 的参数)"""
        trainable_params = self._collect_warmup_params(model, trainer, trainable_model_only=True)
        if not trainable_params:
            print(f"      [{phase_label}] 无可训练参数, 跳过")
            return []

        optimizer_name = (warmup_optimizer or self.warmup_optimizer).lower()

        if optimizer_name == 'lbfgs':
            return self._lbfgs_warmup_with_params(
                model,
                trainer,
                epochs,
                trainable_params,
                phase_label,
                capture_epoch_offset=capture_epoch_offset,
                capture_total_epochs=capture_total_epochs,
            )
        else:
            return self._adam_warmup_with_params(
                model,
                trainer,
                epochs,
                trainable_params,
                phase_label,
                capture_epoch_offset=capture_epoch_offset,
                capture_total_epochs=capture_total_epochs,
            )

    def _lbfgs_warmup_with_params(
        self,
        model,
        trainer,
        epochs,
        params,
        label="",
        capture_epoch_offset: int = 0,
        capture_total_epochs: Optional[int] = None,
    ) -> list:
        """L-BFGS warm-up (指定可训练参数)"""
        optimizer = optim.LBFGS(
            params, lr=0.1, max_iter=20,
            history_size=50, line_search_fn='strong_wolfe'
        )
        loss_history = []
        for epoch in range(epochs):
            points = trainer._sample_all()
            def closure():
                optimizer.zero_grad()
                loss, _ = trainer.calculate_loss(points)
                if self.warmup_l2_lambda > 0:
                    loss = loss + self.warmup_l2_lambda * self._compute_l2_reg(model)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            loss_val = loss.item()
            loss_history.append(loss_val)
            self._warmup_capture_total_epochs = int(capture_total_epochs or epochs)
            self._maybe_capture_warmup_snapshot(
                model=model,
                trainer=trainer,
                global_epoch=capture_epoch_offset + epoch + 1,
                loss_value=loss_val,
                label=label,
            )
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"      [{label}] Epoch {epoch+1}/{epochs}: Loss = {loss_val:.6f}")
        return loss_history

    def _adam_warmup_with_params(
        self,
        model,
        trainer,
        epochs,
        params,
        label="",
        capture_epoch_offset: int = 0,
        capture_total_epochs: Optional[int] = None,
    ) -> list:
        """Adam warm-up (指定可训练参数)"""
        optimizer = optim.Adam(params, lr=0.001)
        loss_history = []
        for epoch in range(epochs):
            points = trainer._sample_all()
            optimizer.zero_grad()
            loss, _ = trainer.calculate_loss(points)
            if self.warmup_l2_lambda > 0:
                loss = loss + self.warmup_l2_lambda * self._compute_l2_reg(model)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            loss_history.append(loss_val)
            self._warmup_capture_total_epochs = int(capture_total_epochs or epochs)
            self._maybe_capture_warmup_snapshot(
                model=model,
                trainer=trainer,
                global_epoch=capture_epoch_offset + epoch + 1,
                loss_value=loss_val,
                label=label,
            )
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"      [{label}] Epoch {epoch+1}/{epochs}: Loss = {loss_val:.6f}")
        return loss_history

    def _create_target_loss_fn(self, trainer: Any, evaluation_points: Tuple[Any, Any, List[Any], Tuple[Any, Any]]) -> Callable:
        """Create a real target-task loss closure for neuron evaluation."""
        def loss_fn(model=None, inputs=None, outputs=None):
            original_model = trainer.model
            try:
                if model is not None:
                    trainer.model = model
                trainer.model.zero_grad(set_to_none=True)
                if hasattr(trainer, 'physics') and trainer.physics is not None:
                    trainer.physics.zero_grad(set_to_none=True)
                points = self._clone_points(evaluation_points)
                loss, _ = trainer.calculate_loss(points)
                return loss
            finally:
                trainer.model = original_model
        return loss_fn

    def _create_physics_branch_loss_fn(
        self,
        trainer: Any,
        evaluation_points: Tuple[Any, Any, List[Any], Tuple[Any, Any]],
    ) -> Callable:
        """Create a loss closure that evaluates hooks inside physics-side branches.

        The state network remains ``trainer.model``. The supplied ``model`` is
        the already-attached physics branch, for example ``trainer.physics.q_net``.
        """
        def loss_fn(model=None, inputs=None, outputs=None):
            trainer.model.zero_grad(set_to_none=True)
            if hasattr(trainer, 'physics') and trainer.physics is not None:
                trainer.physics.zero_grad(set_to_none=True)
            points = self._clone_points(evaluation_points)
            loss, _ = trainer.calculate_loss(points)
            return loss
        return loss_fn

    def _extract_physics_branch_inputs(
        self,
        points: Tuple[Any, Any, List[Any], Tuple[Any, Any]]
    ) -> Optional[List[torch.Tensor]]:
        """Extract x-like inputs for q_net-style branch scoring."""
        if not points:
            return None
        pde_inputs = points[0]
        if torch.is_tensor(pde_inputs):
            return [pde_inputs]
        if isinstance(pde_inputs, (tuple, list)) and pde_inputs:
            first = pde_inputs[0]
            if torch.is_tensor(first):
                return [first]
        return None

    def _aggregate_neuron_scores(
        self,
        score_batches: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Average neuron scores across multiple sampled target-task batches."""
        if not score_batches:
            return {}

        aggregated = {}
        reference = score_batches[0]
        for layer_name, ref_scores in reference.items():
            expected_shape = ref_scores.shape
            layer_scores = []
            for batch_scores in score_batches:
                scores = batch_scores.get(layer_name)
                if scores is None or scores.shape != expected_shape:
                    raise ValueError(f"Inconsistent neuron score shape for {layer_name}")
                layer_scores.append(scores)
            aggregated[layer_name] = np.mean(np.stack(layer_scores, axis=0), axis=0)
        return aggregated

    def _compute_kmeans_threshold(self, scores: np.ndarray) -> float:
        """Stable 1D two-cluster threshold for one layer of neuron scores."""
        if len(scores) < 2:
            return float(scores[0]) if len(scores) == 1 else 0.0

        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        if np.isclose(score_min, score_max):
            return score_min

        c1, c2 = np.percentile(scores, [25, 75])
        for _ in range(50):
            if c1 > c2:
                c1, c2 = c2, c1

            dist1 = np.abs(scores - c1)
            dist2 = np.abs(scores - c2)
            mask1 = dist1 <= dist2
            mask2 = ~mask1

            new_c1 = np.mean(scores[mask1]) if mask1.any() else c1
            new_c2 = np.mean(scores[mask2]) if mask2.any() else c2

            if np.isclose(new_c1, c1, atol=1e-8) and np.isclose(new_c2, c2, atol=1e-8):
                c1, c2 = new_c1, new_c2
                break
            c1, c2 = new_c1, new_c2

        if c1 > c2:
            c1, c2 = c2, c1
        return float((c1 + c2) / 2.0)

    def _compute_kmeans_low_cluster_mask(
        self,
        scores: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Return low-quality cluster membership from 1D two-cluster k-means."""
        n = len(scores)
        if n < 2:
            return np.zeros(n, dtype=float), 0.0, 0.0

        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        if np.isclose(score_min, score_max):
            return np.zeros(n, dtype=float), score_min, score_max

        c1, c2 = np.percentile(scores, [25, 75])
        for _ in range(50):
            if c1 > c2:
                c1, c2 = c2, c1

            dist1 = np.abs(scores - c1)
            dist2 = np.abs(scores - c2)
            mask1 = dist1 <= dist2
            mask2 = ~mask1

            new_c1 = np.mean(scores[mask1]) if mask1.any() else c1
            new_c2 = np.mean(scores[mask2]) if mask2.any() else c2

            if np.isclose(new_c1, c1, atol=1e-8) and np.isclose(new_c2, c2, atol=1e-8):
                c1, c2 = new_c1, new_c2
                break
            c1, c2 = new_c1, new_c2

        if c1 > c2:
            c1, c2 = c2, c1

        low_mask = (np.abs(scores - c1) <= np.abs(scores - c2)).astype(float)
        return low_mask, float(c1), float(c2)

    def _compute_gmm_bad_probability(
        self,
        scores: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float, float, float, bool]:
        """
        Fit a two-component 1D GMM in log-score space and return a calibrated
        badness probability.

        Returns:
            bad_prob: calibrated badness probability derived from posterior excess
                over the low-quality mixture prior, modulated by component separation
            low_mean: mean of the low-quality component (mapped back to score space)
            high_mean: mean of the high-quality component (mapped back to score space)
            low_weight: mixture weight of the low-quality component
            separation_conf: confidence from component separation, in [0, 1]
            bic_gain: BIC improvement of 2-component model over 1-component model
            use_two_component: whether the layer has enough evidence to justify
                a bad-subgroup interpretation
        """
        x = np.asarray(scores, dtype=float)
        n = len(x)
        if n < 2:
            return np.zeros(n, dtype=float), 0.0, 0.0, 0.0, 0.0, 0.0, False

        score_min = float(np.min(x))
        score_max = float(np.max(x))
        if np.isclose(score_min, score_max):
            return np.zeros(n, dtype=float), score_min, score_max, 0.0, 0.0, 0.0, False

        # Score distributions are typically highly right-skewed. Modeling them
        # directly in raw space makes the low-quality component dominate too
        # aggressively, while hard floor-clipping in pure log space tends to
        # create near-degenerate point masses for very small scores. Therefore
        # we use a smooth log1p compression with a TGSR scale.
        eps = 1e-12
        positive_scores = x[x > 0.0]
        score_scale = float(np.median(positive_scores)) if positive_scores.size else 1.0
        score_scale = max(score_scale, eps)
        x_model = np.log1p(np.maximum(x, 0.0) / score_scale)

        means = np.percentile(x_model, [25, 75]).astype(float)
        variances = np.array(
            [max(np.var(x_model), 1e-10), max(np.var(x_model), 1e-10)],
            dtype=float
        )
        weights = np.array([0.5, 0.5], dtype=float)
        responsibilities = np.full((n, 2), 0.5, dtype=float)
        prev_ll = -np.inf

        for _ in range(100):
            pdf = []
            for k in range(2):
                coef = weights[k] / np.sqrt(2.0 * np.pi * variances[k])
                expo = np.exp(-0.5 * ((x_model - means[k]) ** 2) / variances[k])
                pdf.append(coef * expo)
            pdf = np.stack(pdf, axis=1)
            total = np.sum(pdf, axis=1, keepdims=True) + eps
            responsibilities = pdf / total

            nk = np.sum(responsibilities, axis=0) + eps
            weights = nk / n
            means = np.sum(responsibilities * x_model[:, None], axis=0) / nk
            variances = np.sum(
                responsibilities * ((x_model[:, None] - means) ** 2),
                axis=0
            ) / nk
            variances = np.maximum(variances, 1e-10)

            ll = float(np.sum(np.log(total[:, 0])))
            if np.isclose(ll, prev_ll, atol=1e-8):
                break
            prev_ll = ll

        low_idx = int(np.argmin(means))
        high_idx = 1 - low_idx
        raw_bad_prob = responsibilities[:, low_idx]
        low_weight = float(weights[low_idx])

        mean_single = float(np.mean(x_model))
        var_single = float(max(np.var(x_model), 1e-10))
        ll_single = float(np.sum(
            -0.5 * np.log(2.0 * np.pi * var_single)
            - 0.5 * ((x_model - mean_single) ** 2) / var_single
        ))
        ll_double = prev_ll
        bic_single = -2.0 * ll_single + 2.0 * np.log(max(n, 2))
        bic_double = -2.0 * ll_double + 5.0 * np.log(max(n, 2))
        bic_gain = float(bic_single - bic_double)

        low_var = float(variances[low_idx])
        high_var = float(variances[high_idx])
        low_std = float(np.sqrt(low_var))
        high_std = float(np.sqrt(high_var))
        mean_gap = float(abs(means[high_idx] - means[low_idx]))
        denom = (low_std ** 2 + high_std ** 2) + eps
        bc = np.sqrt((2.0 * low_std * high_std) / denom) * np.exp(-(mean_gap ** 2) / (4.0 * denom))
        bc = float(np.clip(bc, 0.0, 1.0))
        separation_conf = 1.0 - bc

        # Calibrate posterior saturation:
        # - If the low-quality component itself occupies a large prior mass, then
        #   moderate posterior values should not be treated as strongly bad.
        # - Only the posterior "excess" over the layer-wise low-component prior is
        #   converted into badness, and this is further discounted when the two
        #   mixture components overlap heavily.
        use_two_component = (bic_gain > 10.0) and (low_weight <= 0.5)
        if use_two_component:
            posterior_excess = np.clip(
                (raw_bad_prob - low_weight) / max(1.0 - low_weight, eps),
                0.0,
                1.0
            )
            bad_prob = posterior_excess * separation_conf
        else:
            # If two-component evidence is weak or the low-score component is not
            # a minority subgroup, treat the layer as unimodal and avoid strong
            # data-driven pruning decisions on this layer.
            bad_prob = np.zeros(n, dtype=float)

        low_mean = float(score_scale * np.expm1(means[low_idx]))
        high_mean = float(score_scale * np.expm1(means[high_idx]))
        return bad_prob, low_mean, high_mean, low_weight, separation_conf, bic_gain, use_two_component

    def _build_gmm_soft_reset_plan(
        self,
        bad_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build continuous soft-reset scales from posterior bad probabilities.

        GMM branch now uses a true continuous gate:
        - every neuron receives a scale factor determined by posterior badness
        - no hard decision such as p_bad >= 0.5 controls whether the neuron is
          softened or not
        - a reporting mask is derived only from whether the resulting gate
          causes a meaningful attenuation

        The default cubic mapping is kept for backward compatibility. Other
        mappings are enabled only for controlled ablation runs via
        TGSR_SOFT_DECAY_MAPPING.
        """
        bad_prob = np.clip(np.asarray(bad_prob, dtype=float), 0.0, 1.0)
        reset_scales = np.ones_like(bad_prob, dtype=float)

        if self.soft_decay_mapping == 'cubic':
            lower_region = bad_prob < 0.5
            if np.any(lower_region):
                lower_norm = bad_prob[lower_region] / 0.5
                reset_scales[lower_region] = (
                    1.0 - (lower_norm ** 3) * (1.0 - self.soft_reset_max_scale)
                )

            upper_region = ~lower_region
            if np.any(upper_region):
                upper_norm = (bad_prob[upper_region] - 0.5) / 0.5
                reset_scales[upper_region] = (
                    self.soft_reset_max_scale
                    - (upper_norm ** 3)
                    * (self.soft_reset_max_scale - self.soft_reset_min_scale)
                )
        elif self.soft_decay_mapping == 'linear':
            reset_scales = 1.0 - bad_prob * (1.0 - self.soft_reset_min_scale)
        elif self.soft_decay_mapping == 'sigmoid':
            steepness = 8.0
            sigmoid = 1.0 / (1.0 + np.exp(steepness * (bad_prob - 0.5)))
            sigmoid_at_zero = 1.0 / (1.0 + np.exp(steepness * (0.0 - 0.5)))
            sigmoid_at_one = 1.0 / (1.0 + np.exp(steepness * (1.0 - 0.5)))
            unit_keep = (sigmoid - sigmoid_at_one) / max(sigmoid_at_zero - sigmoid_at_one, 1e-12)
            reset_scales = self.soft_reset_min_scale + unit_keep * (1.0 - self.soft_reset_min_scale)
        elif self.soft_decay_mapping == 'hard_threshold':
            reset_scales = np.where(bad_prob < 0.5, 1.0, self.soft_reset_min_scale)
        else:  # pragma: no cover - constructor validation guards this branch.
            raise AssertionError(self.soft_decay_mapping)

        reset_scales = np.clip(reset_scales, self.soft_reset_min_scale, 1.0)
        # Reporting-only activity mask: a neuron is considered "actively
        # softened" when the gate reduces its scale by at least 2%.
        gate_strength = 1.0 - reset_scales
        reset_mask = (gate_strength >= 0.02).astype(float)
        return reset_mask, reset_scales

    def _compute_rank_bad_probability(self, scores: np.ndarray) -> np.ndarray:
        """
        Build a smooth fallback badness signal from within-layer score ranking.

        Lower-scored neurons receive higher badness, but a power transform keeps
        the response smoother than a hard tail-selection rule.
        """
        x = np.asarray(scores, dtype=float)
        n = len(x)
        if n <= 1:
            return np.zeros(n, dtype=float)

        order = np.argsort(x)
        rank = np.empty(n, dtype=float)
        rank[order] = np.linspace(0.0, 1.0, num=n, endpoint=True)
        poorness = 1.0 - rank
        return np.power(np.clip(poorness, 0.0, 1.0), 1.5)

    def _compute_gmm_layer_prior(
        self,
        layer_idx: int,
        total_layers: int
    ) -> Tuple[float, float]:
        """
        Layer-depth prior for GMM branch.

        Assumption:
        - layers closer to the input contain more reusable physical structure and
          should be protected more strongly;
        - layers closer to the output are more task-specific and can tolerate a
          stronger gate when badness evidence appears.

        Returns:
            physics_weight: importance prior in [0, 1], larger for shallow layers
            badness_multiplier: attenuation factor applied to badness, smaller for
                shallow layers and closer to 1 for deeper layers
        """
        if total_layers <= 1:
            if self.layer_protection_mode == 'none':
                return 1.0, 1.0
            return 1.0, max(0.0, 1.0 - self.gmm_shallow_protect)

        if self.layer_protection_mode == 'none':
            depth_ratio = layer_idx / (total_layers - 1)
            return 1.0 - depth_ratio, 1.0

        depth_ratio = layer_idx / (total_layers - 1)
        physics_weight = 1.0 - depth_ratio
        if self.layer_protection_mode == 'uniform_matched':
            eta_values = [
                1.0 - self.gmm_shallow_protect * (1.0 - idx / (total_layers - 1))
                for idx in range(total_layers)
            ]
            badness_multiplier = float(np.mean(eta_values))
        else:
            badness_multiplier = 1.0 - self.gmm_shallow_protect * physics_weight
        return physics_weight, float(np.clip(badness_multiplier, 0.0, 1.0))

    def _build_soft_reset_plan(
        self,
        scores: np.ndarray,
        candidate_mask: np.ndarray,
        low_center: float,
        high_center: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a graded soft-reset plan inside the low-quality cluster.

        Returns:
            reset_mask: binary mask of neurons that will be attenuated
            reset_scales: per-neuron scale factors, 1.0 means unchanged
        """
        n_neurons = len(scores)
        reset_mask = np.zeros(n_neurons, dtype=float)
        reset_scales = np.ones(n_neurons, dtype=float)

        candidate_indices = np.where(candidate_mask > 0.5)[0]
        if candidate_indices.size == 0:
            return reset_mask, reset_scales

        max_prune = int(n_neurons * (1 - self.min_keep_ratio))
        if max_prune <= 0:
            return reset_mask, reset_scales

        candidate_scores = scores[candidate_indices]
        dist_low = np.abs(candidate_scores - low_center)
        dist_high = np.abs(candidate_scores - high_center)
        bad_conf = (dist_high - dist_low) / (dist_high + dist_low + 1e-12)

        select_count = max(1, int(np.ceil(candidate_indices.size * self.kmeans_bad_ratio)))
        select_count = min(select_count, max_prune, candidate_indices.size)

        order = np.lexsort((candidate_scores, -bad_conf))
        ranked_indices = candidate_indices[order]
        active_indices = ranked_indices[:select_count]
        if active_indices.size == 0:
            return reset_mask, reset_scales

        reset_mask[active_indices] = 1.0

        if active_indices.size == 1:
            reset_scales[active_indices[0]] = self.soft_reset_min_scale
            return reset_mask, reset_scales

        tier_scales = np.array([
            self.soft_reset_min_scale,
            0.55,
            0.70,
            self.soft_reset_max_scale,
        ], dtype=float)
        tier_edges = np.linspace(0, active_indices.size, len(tier_scales) + 1, dtype=int)
        for tier_idx, scale in enumerate(tier_scales):
            start = tier_edges[tier_idx]
            end = tier_edges[tier_idx + 1]
            if start < end:
                reset_scales[active_indices[start:end]] = scale
        return reset_mask, reset_scales

    def _apply_score_ablation(
        self,
        target_model: nn.Module,
        neuron_scores: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply optional score-ablation transform before low-quality estimation."""
        if self.score_ablation_mode == 'none':
            return neuron_scores

        rng = np.random.default_rng(int(self.score_ablation_seed))
        magnitude_scores = None
        if self.score_ablation_mode == 'magnitude_soft_decay':
            magnitude_scores = collect_incoming_weight_norm_scores(
                target_model,
                neuron_scores.keys(),
            )

        transformed_scores, meta = transform_scores_for_ablation(
            neuron_scores,
            mode=self.score_ablation_mode,
            rng=rng,
            magnitude_scores=magnitude_scores,
        )
        self.evaluation_results['score_ablation_meta'] = meta
        print(
            f"[Score Ablation] score mode={self.score_ablation_mode}, "
            f"seed={self.score_ablation_seed}"
        )
        return transformed_scores

    def _evaluate_and_prune(
        self,
        target_model: nn.Module,
        trainer: Optional[Any],
        initial_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float]:
        """Evaluate neurons on real target-task loss and prune low-quality units."""
        if trainer is None:
            print("[WARN] Trainer missing, skip evaluation and pruning.")
            return {}, {}, 0.0

        score_batches = []
        for batch_idx, cached_points in enumerate(self._get_cached_score_points_batches(trainer), start=1):
            evaluator = NeuroQualityEvaluator(
                target_model, self.device,
                base_alpha=self.score_alpha if self.score_alpha is not None else 0.5,
                alpha_range=0.3
            )
            evaluation_points = self._clone_points(cached_points)
            evaluation_inputs = self._extract_evaluation_inputs(evaluation_points)
            loss_fn = self._create_target_loss_fn(trainer, evaluation_points)
            batch_scores = evaluator.compute_neuron_importance(
                loss_fn=loss_fn,
                data_inputs=evaluation_inputs,
            )
            if not batch_scores:
                print(f"[WARN] Empty neuron-score batch {batch_idx}, skipped.")
                continue

            has_non_finite = any(not np.isfinite(scores).all() for scores in batch_scores.values())
            if has_non_finite:
                print(f"[WARN] Non-finite neuron-score batch {batch_idx}, skipped.")
                continue

            score_batches.append(batch_scores)

        if not score_batches:
            print("[WARN] All neuron-score batches failed, skip pruning.")
            return {}, {}, 0.0

        neuron_scores = self._aggregate_neuron_scores(score_batches)
        print(f"[INFO] Averaged neuron scores over {len(score_batches)} sampled batches.")

        if not neuron_scores:
            print("[警告] 神经元评分为空，跳过修剪")
            return {}, {}, 0.0

        has_non_finite = any(not np.isfinite(scores).all() for scores in neuron_scores.values())
        if has_non_finite:
            print("[警告] 神经元评分出现 NaN/Inf，跳过修剪以保护当前模型")
            return neuron_scores, {}, 0.0

        neuron_scores = self._apply_score_ablation(target_model, neuron_scores)
        prune_masks = self._identify_low_quality_neurons(neuron_scores)

        total_neurons = sum(len(scores) for scores in neuron_scores.values())
        pruned_neurons = sum(mask.sum() for mask in prune_masks.values())
        prune_ratio = pruned_neurons / total_neurons * 100 if total_neurons > 0 else 0.0

        print(f"[统计] 总神经元: {total_neurons}, 标记修剪: {int(pruned_neurons)} ({prune_ratio:.1f}%)")
        return neuron_scores, prune_masks, prune_ratio

    def _evaluate_and_prune_physics_branch(
        self,
        branch_module: nn.Module,
        trainer: Optional[Any],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float]:
        """Evaluate and soft-reset a trainable physics branch such as q_net."""
        if trainer is None:
            print("[WARN] Trainer missing, skip physics-branch evaluation.")
            return {}, {}, 0.0

        score_batches = []
        for batch_idx, cached_points in enumerate(self._get_cached_score_points_batches(trainer), start=1):
            evaluator = NeuroQualityEvaluator(branch_module, self.device)
            evaluation_points = self._clone_points(cached_points)
            evaluation_inputs = self._extract_physics_branch_inputs(evaluation_points)
            loss_fn = self._create_physics_branch_loss_fn(trainer, evaluation_points)
            batch_scores = evaluator.compute_neuron_importance(
                loss_fn=loss_fn,
                data_inputs=evaluation_inputs,
            )
            if not batch_scores:
                print(f"[WARN] Empty physics-branch score batch {batch_idx}, skipped.")
                continue
            if any(not np.isfinite(scores).all() for scores in batch_scores.values()):
                print(f"[WARN] Non-finite physics-branch score batch {batch_idx}, skipped.")
                continue
            score_batches.append(batch_scores)

        if not score_batches:
            print("[WARN] All physics-branch scoring batches failed, skip reset.")
            return {}, {}, 0.0

        neuron_scores = self._aggregate_neuron_scores(score_batches)
        prune_masks = self._identify_low_quality_neurons(neuron_scores)
        total_neurons = sum(len(scores) for scores in neuron_scores.values())
        pruned_neurons = sum(mask.sum() for mask in prune_masks.values())
        prune_ratio = pruned_neurons / total_neurons * 100 if total_neurons > 0 else 0.0
        print(
            f"[TGSR] function branch neurons: {total_neurons}, "
            f"soft-reset marked: {int(pruned_neurons)} ({prune_ratio:.1f}%)"
        )
        return neuron_scores, prune_masks, prune_ratio

    def _identify_low_quality_neurons(
        self,
        neuron_scores: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        识别低质量神经元

        使用基于百分位数的阈值:
        - 得分低于 prune_percentile 的神经元被标记为低质量
        - 但保证至少保留 min_keep_ratio 的神经元

        返回:
            Dict[layer_name, binary_mask]
            其中 1 表示需要修剪的神经元
        """
        prune_masks = {}
        self._last_reset_scales = {}
        self._last_pruning_debug = {}
        all_scores = np.concatenate([s for s in neuron_scores.values()])
        global_threshold = None
        global_masks = {}
        global_low_center = None
        global_high_center = None

        if self.pruning_mode not in ('kmeans', 'gmm'):
            global_threshold = np.percentile(all_scores, self.prune_percentile)
            print(f"  Global pruning threshold (P{self.prune_percentile}): {global_threshold:.4f}")
        elif self.kmeans_scope == 'global':
            global_low_mask, low_center, high_center = self._compute_kmeans_low_cluster_mask(all_scores)
            global_low_center = low_center
            global_high_center = high_center
            print(
                f"  Global K-Means clusters: low_center={low_center:.4f}, "
                f"high_center={high_center:.4f}, low_cluster={int(global_low_mask.sum())}/{len(global_low_mask)}"
            )
            offset = 0
            for layer_name, scores in neuron_scores.items():
                count = len(scores)
                global_masks[layer_name] = global_low_mask[offset:offset + count].copy()
                offset += count

        layer_indices = []
        for layer_name in neuron_scores.keys():
            try:
                layer_indices.append(int(layer_name.split('_')[1]))
            except (IndexError, ValueError):
                layer_indices.append(0)
        total_layers = max(1, max(layer_indices) + 1) if layer_indices else 1

        for layer_name, scores in neuron_scores.items():
            n_neurons = len(scores)
            try:
                layer_idx = int(layer_name.split('_')[1])
            except (IndexError, ValueError):
                layer_idx = 0
            layer_debug: Dict[str, Any] = {
                'mode': self.pruning_mode,
                'layer_index': int(layer_idx),
                'n_neurons': int(n_neurons),
            }

            if self.pruning_mode == 'gmm':
                bad_prob, low_mean, high_mean, low_weight, separation_conf, bic_gain, use_two_component = self._compute_gmm_bad_probability(scores)
                physics_weight, badness_multiplier = self._compute_gmm_layer_prior(layer_idx, total_layers)
                if self.gmm_force_rank_only:
                    fallback_bad = self._compute_rank_bad_probability(scores)
                    bad_prob = np.clip(
                        fallback_bad * self.gmm_fallback_strength * badness_multiplier,
                        0.0,
                        1.0
                    )
                    use_two_component = False
                    gmm_mode = 'rank-only'
                elif use_two_component:
                    bad_prob = np.clip(bad_prob * badness_multiplier, 0.0, 1.0)
                    gmm_mode = '2comp'
                else:
                    fallback_bad = self._compute_rank_bad_probability(scores)
                    bad_prob = np.clip(
                        fallback_bad * self.gmm_fallback_strength * badness_multiplier,
                        0.0,
                        1.0
                    )
                    gmm_mode = 'fallback-depth'
                mask, reset_scales = self._build_gmm_soft_reset_plan(bad_prob)
                mean_gate = float(np.mean(1.0 - reset_scales))
                layer_debug.update({
                    'low_mean': float(low_mean),
                    'high_mean': float(high_mean),
                    'low_weight': float(low_weight),
                    'separation_conf': float(separation_conf),
                    'bic_gain': float(bic_gain),
                    'physics_weight': float(physics_weight),
                    'badness_multiplier': float(badness_multiplier),
                    'use_two_component': bool(use_two_component),
                    'gmm_mode': gmm_mode,
                    'mean_badness': float(np.mean(bad_prob)),
                    'max_badness': float(np.max(bad_prob)) if len(bad_prob) > 0 else 0.0,
                    'mean_gate': mean_gate,
                    'soft_decay_mapping': self.soft_decay_mapping,
                    'gmm_force_rank_only': bool(self.gmm_force_rank_only),
                    'layer_protection_mode': self.layer_protection_mode,
                })
                print(
                    f"  {layer_name} GMM: low_mean={low_mean:.4f}, high_mean={high_mean:.4f}, "
                    f"low_weight={low_weight:.2f}, sep_conf={separation_conf:.3f}, "
                    f"bic_gain={bic_gain:.2f}, physics_weight={physics_weight:.2f}, "
                    f"bad_mult={badness_multiplier:.2f}, mode={gmm_mode}, "
                    f"active_gate={int(mask.sum())}/{n_neurons}, "
                    f"mean_badness={bad_prob.mean():.3f}, mean_gate={mean_gate:.3f}"
                )
            elif self.pruning_mode == 'kmeans' and self.kmeans_scope == 'layerwise':
                low_mask, low_center, high_center = self._compute_kmeans_low_cluster_mask(scores)
                print(
                    f"  {layer_name} clusters (Layerwise K-Means): "
                    f"low_center={low_center:.4f}, high_center={high_center:.4f}, "
                    f"low_cluster={int(low_mask.sum())}/{n_neurons}"
                )
                layer_debug.update({
                    'low_center': float(low_center),
                    'high_center': float(high_center),
                    'low_cluster_count': int(low_mask.sum()),
                    'kmeans_scope': 'layerwise',
                })
                mask, reset_scales = self._build_soft_reset_plan(
                    scores, low_mask, low_center, high_center
                )
            elif self.pruning_mode == 'kmeans':
                low_mask = global_masks[layer_name]
                layer_debug.update({
                    'low_center': float(global_low_center),
                    'high_center': float(global_high_center),
                    'low_cluster_count': int(low_mask.sum()),
                    'kmeans_scope': self.kmeans_scope,
                })
                mask, reset_scales = self._build_soft_reset_plan(
                    scores, low_mask, global_low_center, global_high_center
                )
            else:
                threshold = global_threshold
                mask = (scores < threshold).astype(float)
                reset_scales = (1.0 - mask) + mask * 0.5
                layer_debug.update({
                    'threshold': float(threshold),
                    'prune_percentile': float(self.prune_percentile),
                })

                n_pruned = int(mask.sum())
                max_prune = int(n_neurons * (1 - self.min_keep_ratio))

                if n_pruned > max_prune:
                    sorted_indices = np.argsort(scores)
                    mask = np.zeros(n_neurons)
                    mask[sorted_indices[:max_prune]] = 1.0
                    reset_scales = (1.0 - mask) + mask * 0.5

            prune_masks[layer_name] = mask
            self._last_reset_scales[layer_name] = reset_scales
            layer_debug.update({
                'prune_count': int(mask.sum()),
                'prune_ratio': float(mask.sum() / max(1, n_neurons)),
                'scale_min': float(np.min(reset_scales)) if len(reset_scales) > 0 else 1.0,
                'scale_max': float(np.max(reset_scales)) if len(reset_scales) > 0 else 1.0,
                'active_gate_count': int(mask.sum()),
            })
            self._last_pruning_debug[layer_name] = layer_debug
            if mask.sum() > 0:
                active_scales = reset_scales[mask > 0.5]
                print(
                    f"    {layer_name}: prune {int(mask.sum())}/{n_neurons} neurons, "
                    f"scale_range={active_scales.min():.2f}~{active_scales.max():.2f}"
                )
            else:
                print(f"    {layer_name}: prune 0/{n_neurons} neurons")

        return prune_masks

    def _prune_neurons(
        self,
        target_model: nn.Module,
        initial_state: Dict,
        prune_masks: Dict[str, np.ndarray]
    ) -> None:
        """
        修剪/重置低质量神经元

        根据 reset_mode:
        - 'reinit': 使用 Xavier 初始化重置权重
        - 'scale': 将权重缩放 50% (更保守)
        """
        with torch.no_grad():
            linear_modules = [m for m in target_model.modules() if isinstance(m, nn.Linear)]
            last_linear_idx = len(linear_modules) - 1
            linear_idx = 0

            for name, module in target_model.named_modules():
                if isinstance(module, nn.Linear):
                    if linear_idx == last_linear_idx:
                        linear_idx += 1
                        continue

                    layer_name = f"layer_{linear_idx}"

                    if layer_name in prune_masks:
                        mask = prune_masks[layer_name]
                        mask_tensor = torch.tensor(mask, device=self.device).float()

                        if self.reset_mode == 'reinit':
                            # 完全重新初始化
                            reinit_weight = torch.empty_like(module.weight)
                            nn.init.xavier_uniform_(reinit_weight)
                            reinit_bias = torch.zeros_like(module.bias)

                            # 混合: 保留好的，重置差的
                            keep_mask = 1 - mask_tensor
                            module.weight.data = (
                                module.weight.data * keep_mask.view(-1, 1) +
                                reinit_weight * mask_tensor.view(-1, 1)
                            )
                            module.bias.data = (
                                module.bias.data * keep_mask +
                                reinit_bias * mask_tensor
                            )

                        elif self.reset_mode == 'scale':
                            # 分级 soft reset: 差得越多，缩放越强
                            scale_values = self._last_reset_scales.get(layer_name)
                            if scale_values is not None:
                                scale_mask = torch.tensor(scale_values, device=self.device).float()
                            else:
                                scale_factor = 0.5
                                keep_mask = 1 - mask_tensor
                                scale_mask = keep_mask + mask_tensor * scale_factor

                            module.weight.data *= scale_mask.view(-1, 1)
                            module.bias.data *= scale_mask

                    linear_idx += 1


# =============================================================================
# 兼容性函数 (供 Trainer 直接调用)
# =============================================================================

def tgsr_transfer_weights(
    model: nn.Module,
    source_path: str,
    device: torch.device,
    physics_engine: Any = None,
    trainer: Any = None,
    warmup_epochs_cap: int = 200,
    warmup_ratio: float = 0.2,
    warmup_epochs: Optional[int] = None,
    prune_percentile: float = 20.0
) -> nn.Module:
    """
    自适应两阶段迁移的便捷函数

    Args:
        model: 目标模型
        source_path: 源模型路径
        device: 计算设备
        physics_engine: 物理引擎
        trainer: 训练器
        warmup_epochs: 预热轮数
        prune_percentile: 修剪百分位

    Returns:
        修改后的模型
    """
    strategy = TGSRTransferStrategy(
        device=device,
        warmup_epochs_cap=warmup_epochs_cap,
        warmup_ratio=warmup_ratio,
        warmup_epochs=warmup_epochs,
        prune_percentile=prune_percentile
    )

    strategy.transfer(
        target_model=model,
        source_path=source_path,
        physics_engine=physics_engine,
        trainer=trainer
    )

    return model
