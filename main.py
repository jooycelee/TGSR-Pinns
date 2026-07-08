import argparse
import os
import random
import sys

import numpy as np
import torch

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import load_config
from src.core.trainer import Trainer
from src.models.fcn import FCN
from src.utils.logger import Logger

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def _check_cuda_compatibility() -> bool:
    """Return True if CUDA can run kernels on current hardware."""
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device='cuda')
            del test_tensor
            return True
        except RuntimeError as exc:
            if 'no kernel image is available' in str(exc):
                print('[WARN] CUDA architecture is not supported by current PyTorch, fallback to CPU.')
                return False
            raise
    return False


_cuda_works = _check_cuda_compatibility()
device = torch.device('cuda' if _cuda_works else 'cpu')


def print_device_info() -> None:
    print('\n' + '=' * 50)
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f'  [GPU] {gpu_name} ({gpu_mem:.1f} GB)')
        print(f'  CUDA Version: {torch.version.cuda}')
    else:
        if torch.cuda.is_available():
            print('  [CPU] CUDA device exists but is not supported by this PyTorch build')
        else:
            print('  [CPU] No CUDA available')
    print('=' * 50 + '\n')


def _coerce_seed(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('-'):
            digits = stripped[1:]
        else:
            digits = stripped
        if digits.isdigit():
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def _resolve_seed(config, cli_seed=None, env_seed=None):
    training_cfg = config.get('training', {})
    runtime_cfg = config.get('runtime', {})
    candidates = [
        ('cli', cli_seed),
        ('env', env_seed),
        ('config.seed', config.get('seed')),
        ('config.training.seed', training_cfg.get('seed')),
        ('config.runtime.seed', runtime_cfg.get('seed')),
    ]
    for source, value in candidates:
        parsed = _coerce_seed(value)
        if parsed is not None:
            return parsed, source
    return None, None


def _apply_global_seed(seed):
    if seed is None:
        return

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:
        print(f'[WARN] Failed to enable deterministic algorithms: {exc}')


def _parse_key_value_overrides(items, value_cast=float, label='override'):
    overrides = {}
    if not items:
        return overrides

    for raw in items:
        if raw is None:
            continue
        text = raw.strip()
        if not text:
            continue
        if '=' not in text:
            raise ValueError(f'Invalid {label}: {raw}. Expected key=value.')
        key, value_text = text.split('=', 1)
        key = key.strip()
        value_text = value_text.strip()
        if not key:
            raise ValueError(f'Invalid {label}: {raw}. Empty key.')
        overrides[key] = value_cast(value_text)
    return overrides


def main():
    parser = argparse.ArgumentParser(description='PINN runner')
    parser.add_argument('--config', type=str, required=True, help='Config file path (.json)')
    parser.add_argument('--load_model', type=str, default=None, help='Pretrained model path override')

    from src.transfer import STRATEGY_REGISTRY

    valid_modes = ['none'] + list(STRATEGY_REGISTRY.keys())
    parser.add_argument('--transfer_mode', type=str, choices=valid_modes, default='none',
                        help=f'Transfer mode: {valid_modes}')
    parser.add_argument('--transfer_ratio', type=float, default=None, help='Transfer ratio override')
    parser.add_argument('--learning_rate', type=float, default=None, help='Adam learning rate override')
    parser.add_argument('--batch_size_data', type=int, default=None, help='Data batch size override')
    parser.add_argument('--noise_level', type=float, default=None, help='Observation noise override')
    parser.add_argument('--fixed_data', type=str, choices=['true', 'false'], default=None,
                        help='Use a fixed noisy observation pool during training')
    parser.add_argument('--fixed_data_size', type=int, default=None,
                        help='Fixed noisy observation pool size override')
    parser.add_argument('--weight_override', action='append', default=None,
                        help='Loss weight override(s), e.g. data=75 or bc=20')
    parser.add_argument('--physics_param_override', action='append', default=None,
                        help='Physics parameter init override(s), e.g. nu=0.01 or alpha=0.02')
    parser.add_argument('--use_tgsr', type=str, choices=['true', 'false'], default=None,
                        help='TGSR mode override')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name override used for results path')
    parser.add_argument('--pruning_mode', type=str, choices=['percentile', 'kmeans', 'gmm'], default='gmm',
                        help='TGSR pruning mode')
    parser.add_argument('--score_num_batches', type=int, default=None, help='TGSR scoring batch count')
    parser.add_argument('--kmeans_scope', type=str, choices=['global', 'layerwise'], default=None,
                        help='TGSR kmeans threshold scope')
    parser.add_argument('--seed', type=int, default=None, help='Random seed override')
    args = parser.parse_args()

    config = load_config(args.config)

    if 'training' not in config:
        config['training'] = {}
    if 'runtime' not in config:
        config['runtime'] = {}

    if args.load_model:
        config['training']['load_model_path'] = args.load_model
        print(f'[INFO] CLI override: load_model_path = {args.load_model}')

    if args.transfer_mode != 'none':
        config['training']['transfer_mode'] = args.transfer_mode
        print(f'[INFO] CLI override: transfer_mode = {args.transfer_mode}')

    if args.transfer_ratio is not None:
        config['training']['transfer_ratio'] = args.transfer_ratio
        print(f'[INFO] CLI override: transfer_ratio = {args.transfer_ratio}')

    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
        print(f'[INFO] CLI override: learning_rate = {args.learning_rate}')

    if args.batch_size_data is not None:
        config['training']['batch_size_data'] = args.batch_size_data
        print(f'[INFO] CLI override: batch_size_data = {args.batch_size_data}')

    if args.noise_level is not None:
        config['physics']['noise_level'] = args.noise_level
        print(f'[INFO] CLI override: noise_level = {args.noise_level}')

    if args.fixed_data is not None:
        config['training']['fixed_data_loss'] = (args.fixed_data.lower() == 'true')
        print(f"[INFO] CLI override: fixed_data_loss = {config['training']['fixed_data_loss']}")

    if args.fixed_data_size is not None:
        config['training']['fixed_data_size'] = args.fixed_data_size
        print(f'[INFO] CLI override: fixed_data_size = {args.fixed_data_size}')

    weight_overrides = _parse_key_value_overrides(args.weight_override, value_cast=float, label='weight_override')
    if weight_overrides:
        config['training'].setdefault('weights', {})
        for key, value in weight_overrides.items():
            config['training']['weights'][key] = value
            print(f'[INFO] CLI override: weights.{key} = {value}')

    physics_param_overrides = _parse_key_value_overrides(
        args.physics_param_override,
        value_cast=float,
        label='physics_param_override'
    )
    if physics_param_overrides:
        config['physics'].setdefault('params', {})
        for key, value in physics_param_overrides.items():
            config['physics']['params'][key] = value
            print(f'[INFO] CLI override: physics.params.{key} = {value}')

    if args.use_tgsr is not None:
        config['training']['use_tgsr'] = (args.use_tgsr.lower() == 'true')
        print(f"[INFO] CLI override: use_tgsr = {config['training']['use_tgsr']}")

    env_seed = os.environ.get('PINN_SEED')
    resolved_seed, seed_source = _resolve_seed(config, cli_seed=args.seed, env_seed=env_seed)
    if resolved_seed is not None:
        config['seed'] = resolved_seed
        config['training']['seed'] = resolved_seed
        config['runtime']['seed'] = resolved_seed
        config['runtime']['seed_source'] = seed_source
        _apply_global_seed(resolved_seed)
        print(f'[INFO] Seed fixed: seed = {resolved_seed} (source: {seed_source})')
    else:
        print('[WARN] No random seed detected. This run is non-deterministic.')

    if args.experiment_name is not None:
        config['experiment_name'] = args.experiment_name
        print(f'[INFO] CLI override: experiment_name = {args.experiment_name}')

    print_device_info()
    print(f"[INFO] Current experiment: {config['experiment_name']}")

    logger = Logger(config['experiment_name'])
    logger.log_config(config)

    model = FCN(config['model']['layers'])
    model.to(device)

    DEFERRED_STRATEGIES = {'tgsr'}

    load_path = config['training'].get('load_model_path')
    if load_path:
        if os.path.exists(load_path):
            print(f'[INFO] Loading pretrained model: {load_path}')
            try:
                mode = config['training'].get('transfer_mode', 'none')

                if mode in DEFERRED_STRATEGIES:
                    print(f"[INFO] Strategy '{mode}' requires trainer context, defer transfer...")
                    config['training']['_deferred_transfer'] = True
                    config['training']['_deferred_source_path'] = load_path
                    config['training']['_deferred_mode'] = mode

                elif mode != 'none':
                    print(f"[WARN] Unknown transfer mode '{mode}', fallback to standard state_dict load")
                    state_dict = torch.load(load_path, map_location=device)
                    model.load_state_dict(state_dict, strict=False)

                print(f'[INFO] Model loaded (mode: {mode}).')

            except Exception as exc:
                print(f'[ERROR] Failed to load model: {exc}')
                import traceback
                traceback.print_exc()
    else:
        print('[INFO] No pretrained model specified. Train from scratch.')

    from src.physics import get_physics_engine

    physics = get_physics_engine(config['physics'])
    print(f'[INFO] Physics engine: {physics.__class__.__name__}')
    physics.to(device)

    print('[INFO] Initializing trainer...')
    trainer = Trainer(model, physics, config, logger)

    if config['training'].get('_deferred_transfer', False):
        source_path = config['training']['_deferred_source_path']
        mode = config['training']['_deferred_mode']
        print(f'[INFO] Execute deferred transfer: {mode}')

        from src.transfer import get_transfer_strategy

        strategy_kwargs = {}

        if 'tgsr_warmup_epochs_cap' in config['training']:
            strategy_kwargs['warmup_epochs_cap'] = config['training']['tgsr_warmup_epochs_cap']
        elif 'tgsr_warmup_epochs' in config['training']:
            strategy_kwargs['warmup_epochs_cap'] = config['training']['tgsr_warmup_epochs']
        elif 'warmup_epochs_cap' in config['training']:
            strategy_kwargs['warmup_epochs_cap'] = config['training']['warmup_epochs_cap']
        elif 'warmup_epochs' in config['training']:
            strategy_kwargs['warmup_epochs_cap'] = config['training']['warmup_epochs']

        for key in [
            'warmup_ratio',
            'warmup_optimizer',
            'warmup_l2_lambda',
            'progressive_warmup',
            'prune_percentile',
            'reset_mode',
            'min_keep_ratio',
            'score_num_batches',
            'kmeans_scope',
        ]:
            if key in config['training']:
                strategy_kwargs[key] = config['training'][key]

        if args.pruning_mode is not None:
            strategy_kwargs['pruning_mode'] = args.pruning_mode
        if args.score_num_batches is not None:
            strategy_kwargs['score_num_batches'] = args.score_num_batches
        if args.kmeans_scope is not None:
            strategy_kwargs['kmeans_scope'] = args.kmeans_scope

        strategy = get_transfer_strategy(mode, device, **strategy_kwargs)
        strategy.transfer(
            target_model=model,
            source_path=source_path,
            physics_engine=physics,
            trainer=trainer,
        )

    print(f"[INFO] Start training: {config['experiment_name']}")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print('\n[WARN] Training interrupted by user.')
        logger.save_history()


if __name__ == '__main__':
    main()
