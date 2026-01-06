# Model模块初始化文件
# 重构后的CircuitTransformer模型组件

from .core import CircuitTransformer
from .training import train
from .inference import optimize, optimize_batch, _batch_estimate_policy, _batch_estimate_v_value_via_simulation_kvcache
from .mcts_optimization import _batch_MCTS_policy_with_leaf_parallelization
from .ppo_training import (
    train_ppo, _circuit_iterator, _create_circuit_env,
    _prepare_batch_input, _sample_action, _process_trajectories, _update_policy,
    _update_value_function, _log_prob, _batch_step, value_forward_pass, policy_forward_pass
)
from .model_refactored import CircuitTransformerRefactored

__all__ = [
    'CircuitTransformer',
    'CircuitTransformerRefactored',
    'train',
    'optimize',
    'optimize_batch',
    'train_ppo'
] 