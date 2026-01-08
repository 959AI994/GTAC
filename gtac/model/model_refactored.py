from .core import CircuitTransformer
from .training import train
from .inference import optimize, optimize_batch, _batch_estimate_policy, _batch_estimate_v_value_via_simulation_kvcache
from .mcts_optimization import _batch_MCTS_policy_with_leaf_parallelization
from .ppo_training import (
    train_ppo, _circuit_iterator, _create_circuit_env, 
    _prepare_batch_input, _sample_action, _process_trajectories, _update_policy, _optimize_with_ppo_policy,
    _update_value_function, _log_prob, _batch_step, value_forward_pass, policy_forward_pass, _calculate_comprehensive_reward,
    _optimize_batch_with_ppo_policy
)


class CircuitTransformerRefactored(CircuitTransformer):
    """Refactored CircuitTransformer class that integrates all split methods"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    train = train
    train_ppo = train_ppo
    
    optimize = optimize
    optimize_batch = optimize_batch
    _batch_estimate_policy = _batch_estimate_policy
    _batch_estimate_v_value_via_simulation_kvcache = _batch_estimate_v_value_via_simulation_kvcache
    
    _batch_MCTS_policy_with_leaf_parallelization = _batch_MCTS_policy_with_leaf_parallelization
    
    _circuit_iterator = _circuit_iterator
    _optimize_with_ppo_policy = _optimize_with_ppo_policy
    _create_circuit_env = _create_circuit_env
    _prepare_batch_input = _prepare_batch_input
    _sample_action = _sample_action
    
    _process_trajectories = _process_trajectories
    _update_policy = _update_policy
    _update_value_function = _update_value_function
    _log_prob = _log_prob
    _batch_step = _batch_step
    value_forward_pass = value_forward_pass
    policy_forward_pass = policy_forward_pass
    _calculate_comprehensive_reward = _calculate_comprehensive_reward
    _optimize_batch_with_ppo_policy = _optimize_batch_with_ppo_policy


# Provide an alias for backward compatibility
CircuitTransformer = CircuitTransformerRefactored 