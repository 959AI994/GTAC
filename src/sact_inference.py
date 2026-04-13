"""
Inference methods for ScalableCircuitTransformer

This module contains all inference-related methods adapted for dynamic encoding.
These methods are bound to ScalableCircuitTransformer instances at initialization.
"""

import time
import copy
import numpy as np
from scalable_circuit_transformer_refdfs.dynamic_encoding import DynamicEncoder
import scipy.special as special
import bitarray
from scalable_circuit_transformer_refdfs.utils import *
from scalable_circuit_transformer_refdfs.environment import LogicNetworkEnv
from scalable_circuit_transformer_refdfs.mcts import MCTSNode
from scalable_circuit_transformer_refdfs.encoding import stack_to_encoding


def _select_tokens_from_masked_logits(
    masked_logits: np.ndarray,
    inf_T: float,
    rng: np.random.Generator,
    top_k: int = 0,
    sample_when_temp: bool = True,
    low_conf_margin: float = 0.0,
) -> np.ndarray:
    """
    Decode helper with optional top-k and confidence-triggered stochastic fallback.
    masked_logits: invalid actions should already be -inf or very negative.
    """
    tokens = np.zeros(masked_logits.shape[0], dtype=np.int32)
    for i in range(masked_logits.shape[0]):
        row = masked_logits[i].astype(np.float64)
        valid_idx = np.where(np.isfinite(row) & (row > np.finfo(np.float32).min / 2))[0]
        if len(valid_idx) == 0:
            tokens[i] = 0
            continue
        row_valid = row[valid_idx]
        # Optional top-k truncation
        if top_k and top_k > 0 and len(valid_idx) > top_k:
            keep_local = np.argpartition(row_valid, -top_k)[-top_k:]
            keep_idx = valid_idx[keep_local]
            mask = np.zeros_like(row, dtype=bool)
            mask[keep_idx] = True
            row = np.where(mask, row, -np.inf)
            valid_idx = keep_idx
            row_valid = row[valid_idx]

        # Margin-based fallback: when confidence is low, sample from restricted set.
        do_sample = False
        if len(row_valid) >= 2 and low_conf_margin > 0:
            best2 = np.sort(row_valid)[-2:]
            margin = best2[-1] - best2[-2]
            if margin < low_conf_margin:
                do_sample = True
        if inf_T != 1.0 and sample_when_temp:
            do_sample = True

        if not do_sample:
            tokens[i] = int(np.nanargmax(row))
            continue

        t = max(float(inf_T), 1e-6)
        logits_t = row[valid_idx] / t
        logits_t = logits_t - np.max(logits_t)
        probs = np.exp(logits_t)
        probs = probs / (np.sum(probs) + 1e-12)
        tokens[i] = int(rng.choice(valid_idx, p=probs))
    return tokens


def _batch_estimate_policy(self, envs: list, src_tokens, src_pos_enc, src_action_mask, action_masks, cache):
    """
    Estimate policy for a batch of environments.
    
    Args:
        envs: List of LogicNetworkEnv environments
        src_tokens: Source tokens (encoder inputs)
        src_pos_enc: Source positional encodings
        src_action_mask: Source action masks
        action_masks: Current action masks for each environment
        cache: KV cache from previous inference
    
    Returns:
        Tuple of (policy, cache) where policy is softmax probabilities
    """
    start_time = time.time()
    indices = [len(env.tokens) for env in envs]
    max_token_length = np.max(indices) + 1
    
    # Get vocab size for each environment (use max for padding)
    max_vocab_size = self.max_vocab_size  # Fixed vocab size for all circuits
    
    tgt_tokens = np.stack(
        [np.array(env.tokens + [0] * (max_token_length - len(env.tokens)), dtype=np.int32) for env in envs],
        axis=0)
    tgt_pos_enc = np.stack(
        [np.stack(
            env.positional_encodings + [np.zeros(self.max_tree_depth * 2, dtype=np.float32)] * (
                    max_token_length - len(env.tokens)),
            axis=0) for env in envs]
        , axis=0)
    tgt_action_mask = np.stack(
        [np.concatenate([np.stack(env.action_masks),
                        np.ones((max_token_length - len(env.action_masks), self.max_vocab_size), dtype=bool)], axis=0)
        for env in envs]
    )
    inputs = {'inputs': src_tokens, 'enc_pos_encoding': src_pos_enc,
              'targets': tgt_tokens, 'dec_pos_encoding': tgt_pos_enc,
              'enc_action_mask': src_action_mask, 'dec_action_mask': tgt_action_mask}
    if cache is not None:
        inputs['cache'] = cache
    policy, cache = self._transformer_inference(inputs)
    policy = np.stack([policy_i[j] for policy_i, j in zip(policy, indices)], axis=0)
    if self.verbose > 0:
        print("policy estimation time: %.2f" % (time.time() - start_time))
    return special.softmax(np.where(action_masks, policy / self.policy_temperature_in_mcts, np.finfo(np.float32).min), axis=1), cache


def _batch_estimate_v_value_via_simulation_kvcache(self, envs: list, src_tokens, src_pos_enc, src_action_mask,
                                                   max_inference_seq_length, cache=None, num_leaf_parallelization=1,
                                                   return_debug: bool = False):
    """
    Estimate value function via simulation with KV cache.
    
    Args:
        envs: List of LogicNetworkEnv environments
        src_tokens: Source tokens (encoder inputs)
        src_pos_enc: Source positional encodings
        src_action_mask: Source action masks
        max_inference_seq_length: Maximum sequence length for inference
        cache: KV cache from previous inference
        num_leaf_parallelization: Number of parallel leaf evaluations
    
    Returns:
        Tuple of (values, successes, cache)
    """
    total_time = time.time()
    envs = self._copy_env(envs)
    copy_time = time.time() - total_time
    batch_size = len(envs)
    v = np.zeros(len(envs), dtype=np.float64)
    inputs = {'inputs': src_tokens, 'enc_pos_encoding': src_pos_enc, 'enc_action_mask': src_action_mask, 'cache': cache}
    targets = np.zeros((batch_size, 1), dtype=np.int32)
    pos_dim = self.max_tree_depth * 2
    dec_pos_encoding = np.zeros((batch_size, 1, pos_dim), dtype=np.float32)
    for j, env in enumerate(envs):
        if env.positional_encodings:
            pos0 = np.array(env.positional_encodings[-1], dtype=np.float32)[:pos_dim]
        else:
            pos0 = stack_to_encoding(env.tree_stack, env.cur_root_id, self.max_tree_depth, max_outputs=getattr(env, 'max_outputs', 256))
            pos0 = np.array(pos0, dtype=np.float32)[:pos_dim]
        if len(pos0) < pos_dim:
            pos0 = np.concatenate([pos0, np.zeros(pos_dim - len(pos0), dtype=np.float32)])
        dec_pos_encoding[j, 0, :] = pos0
    if cache is not None and num_leaf_parallelization > 1:
        cache['encoder_outputs'] = np.concatenate([cache['encoder_outputs']] * num_leaf_parallelization, axis=0)
    transformer_time = 0.
    action_mask_time = 0.
    step_time = 0.
    inf_T = max(1e-5, getattr(self, 'policy_temperature_in_mcts', 1.0))
    decode_top_k = int(getattr(self, 'decode_top_k_in_inference', 0) or 0)
    sample_when_temp = bool(getattr(self, 'sample_when_temp_in_inference', True))
    low_conf_margin = float(getattr(self, 'decode_low_conf_margin', 0.0) or 0.0)
    rng = np.random.default_rng(getattr(self, 'tt_seed', 42))

    steps_run = 0
    for i in range(max_inference_seq_length):
        steps_run = i + 1
        inputs['targets'], inputs['dec_pos_encoding'] = targets, dec_pos_encoding

        # generate action mask
        start_time = time.time()
        # REF-aware indexing: always use latest env mask.
        action_masks = np.stack([e.action_masks[-1] for e in envs], axis=0)
        padded_action_masks = np.zeros((batch_size, self.max_vocab_size), dtype=bool)
        for j, env in enumerate(envs):
            env_mask_size = min(len(action_masks[j]), self.max_vocab_size)
            padded_action_masks[j, :env_mask_size] = action_masks[j, :env_mask_size]
            # Position-token masks may exceed regular valid token space, so only apply valid_mask
            # when env is not currently expecting REF position.
            if not getattr(env, '_expecting_ref_position', False):
                valid_mask = self.encoder.get_valid_token_mask(env.num_inputs)
                padded_action_masks[j] = np.logical_and(padded_action_masks[j], valid_mask)
        inputs['dec_action_mask'] = np.expand_dims(padded_action_masks, axis=1)
        action_mask_time += time.time() - start_time

        start_time = time.time()
        policy, cache = self._transformer_inference(inputs, return_kv_cache=True, return_last_token=True)
        inputs['cache'] = cache
        transformer_time += time.time() - start_time

        start_time = time.time()
        masked_policy = np.where(padded_action_masks, policy, np.finfo(np.float32).min)
        tokens = _select_tokens_from_masked_logits(
            masked_policy,
            inf_T=inf_T,
            rng=rng,
            top_k=decode_top_k,
            sample_when_temp=sample_when_temp,
            low_conf_margin=low_conf_margin,
        )
        rewards, dones = zip(*[e.step(int(token)) if not e.is_finished else (0, True) for token, e in zip(tokens, envs)])
        step_time += time.time() - start_time

        dec_pos_encoding = [np.array(e.positional_encodings[-1], dtype=np.float32)[:pos_dim] for e in envs]
        for k, pe in enumerate(dec_pos_encoding):
            if len(pe) < pos_dim:
                dec_pos_encoding[k] = np.concatenate([pe, np.zeros(pos_dim - len(pe), dtype=np.float32)])

        v += np.array(rewards)
        if all([e.is_finished for e in envs]):
            break

        pos_encodings = np.expand_dims(np.stack(dec_pos_encoding, axis=0),
                                       axis=1)  # [batch_size, 1, max_tree_depth * 2]
        targets_new = np.expand_dims(tokens, axis=1)
        if self.use_kv_cache:
            targets = targets_new
            dec_pos_encoding = pos_encodings
        else:
            targets = np.concatenate([targets, targets_new], axis=1)
            dec_pos_encoding = np.concatenate([dec_pos_encoding, pos_encodings], axis=1)
    if cache is not None:
        cache['kv_cache'] = None
        if num_leaf_parallelization > 1:
            cache['encoder_outputs'] = cache['encoder_outputs'][:(batch_size // num_leaf_parallelization)]
    debug_infos = []
    if return_debug:
        for idx, e in enumerate(envs):
            final_valid_actions = 0
            if len(e.action_masks) > 0:
                final_valid_actions = int(np.sum(np.asarray(e.action_masks[-1], dtype=bool)))
            last_reward = float(e.rewards[-1]) if len(e.rewards) > 0 else 0.0
            hit_unfinished_penalty = bool(
                e.is_finished and (last_reward <= float(getattr(e, 'unfinished_penalty', -10)) + 1e-8)
            )
            if e.success:
                reason = "success_eos"
            elif hit_unfinished_penalty:
                reason = "unfinished_penalty"
            elif e.is_finished and getattr(e, 't', 0) >= getattr(e, 'max_length', 0) - 1:
                reason = "max_length_reached_unfinished"
            elif final_valid_actions <= 0 and not e.success:
                reason = "no_valid_action_mask"
            elif steps_run >= max_inference_seq_length and not e.success:
                reason = "rollout_budget_exhausted"
            elif e.is_finished and not e.success:
                reason = "finished_without_eos"
            else:
                reason = "stopped"
            debug_infos.append({
                "reason": reason,
                "steps_run": int(steps_run),
                "rollout_reward_sum": float(v[idx]),
                "cumulative_reward": float(e.cumulative_reward),
                "success": bool(e.success),
                "is_finished": bool(e.is_finished),
                "gen_eos": bool(getattr(e, 'gen_eos', False)),
                "final_valid_actions": int(final_valid_actions),
                "last_reward": float(last_reward),
                "hit_unfinished_penalty": bool(hit_unfinished_penalty),
            })
    if self.verbose > 0:
        print("simulation time: total %f, copy %.2f, step %.2f, transformer %.2f, action mask %.2f; #(steps) = %d" %
              (time.time() - total_time, copy_time, step_time, transformer_time, action_mask_time, i))
    if return_debug:
        return v.tolist(), [e.success for e in envs], cache, debug_infos
    return v.tolist(), [e.success for e in envs], cache


def optimize(self,
             aigs: list,
             context_num_inputs=None,
             input_tts: list = None,
             care_set_tts=None,
             ffws=None,
             context_hash_list=None,
             num_mcts_steps=0,
             num_leaf_parallelization=8,
             num_mcts_playouts_per_step=10,
             max_inference_seq_length=None,
             max_inference_reward_list=None,
             max_mcts_inference_seq_length=None,
             use_controllability_dont_cares=True,
             tts_compressed_list=None,
             overflow_option='origin',
             return_envs=False,
             w_gate=1,
             w_delay=0):
    """
    Optimize a list of circuits (AIGs).
    
    Args:
        aigs: List of AIG circuits (can be file paths, AIG strings, or root nodes)
        context_num_inputs: Number of inputs in context (if different from circuit)
        input_tts: List of input truth tables (one per circuit)
        care_set_tts: List of care set truth tables
        ffws: List of feed-forward mappings
        context_hash_list: List of context hashes
        num_mcts_steps: Number of MCTS steps
        num_leaf_parallelization: Number of parallel leaf evaluations
        num_mcts_playouts_per_step: Number of MCTS playouts per step
        max_inference_seq_length: Maximum inference sequence length
        max_inference_reward_list: List of max inference rewards
        max_mcts_inference_seq_length: Maximum MCTS inference sequence length
        use_controllability_dont_cares: Use controllability don't cares
        tts_compressed_list: List of compressed truth tables
        overflow_option: What to do on overflow ('origin' or other)
        return_envs: Whether to return environments
        w_gate: Gate weight for reward
        w_delay: Delay weight for reward
    
    Returns:
        List of optimized circuits
    """
    if self.ckpt_path is None:
        print("Warning: no checkpoint loaded. Make sure to load a checkpoint before optimization.")

    if max_inference_seq_length is None:
        max_inference_seq_length = self.max_seq_length
    if max_mcts_inference_seq_length is None:
        max_mcts_inference_seq_length = max_inference_seq_length

    optimized_aigs = []
    for i in range(0, len(aigs), self.inference_batch_size):
        aigs_batch = aigs[i: i + self.inference_batch_size]
        care_set_tts_batch = care_set_tts[i: i + self.inference_batch_size] if care_set_tts is not None else None
        ffws_batch = ffws[i: i + self.inference_batch_size] if ffws is not None else None
        input_tts_batch = input_tts[i: i + self.inference_batch_size] if input_tts is not None else None
        context_hash_list_batch = context_hash_list[i: i + self.inference_batch_size] if context_hash_list is not None else None
        tts_compressed_batch = tts_compressed_list[i: i + self.inference_batch_size] if tts_compressed_list is not None else None
        max_inference_reward_list_batch = max_inference_reward_list[i: i + self.inference_batch_size] if max_inference_reward_list is not None else None
        optimized_aigs += self.optimize_batch(aigs_batch,
                                              max_inference_seq_length,
                                              max_mcts_inference_seq_length,
                                              context_num_inputs,
                                              input_tts_batch,
                                              care_set_tts_batch,
                                              ffws_batch,
                                              context_hash_list_batch,
                                              max_inference_reward_list_batch,
                                              num_mcts_steps,
                                              num_leaf_parallelization,
                                              num_mcts_playouts_per_step,
                                              use_controllability_dont_cares,
                                              tts_compressed_batch,
                                              overflow_option,
                                              return_envs,
                                              w_gate=w_gate,
                                              w_delay=w_delay)
    return optimized_aigs


def optimize_batch(self,
                   aigs: list,
                   max_inference_seq_length,
                   max_mcts_inference_seq_length=None,
                   context_num_inputs=None,
                   input_tts: list = None,
                   care_set_tts=None,
                   ffws=None,
                   context_hash_list=None,
                   max_inference_reward_list=None,
                   num_mcts_steps=0,
                   num_leaf_parallelization=8,
                   num_mcts_playouts_per_step=10,
                   use_controllability_dont_cares=True,
                   tts_compressed=None,
                   overflow_option='origin',
                   return_envs=False,
                   return_mcts_roots=False,
                   return_input_encodings=False,
                   puct_explore_ratio=1.,
                   w_gate=1,
                   w_delay=0):
    """
    Optimize a batch of circuits using dynamic encoding.
    
    This is the core optimization method, adapted for ScalableCircuitTransformer
    with support for variable input/output sizes.
    """
    from scalable_circuit_transformer_refdfs.monte_carlo_tt import compute_tts_adaptive
    
    total_time = time.time()
    start_time = time.time()
    encoded_aigs = []
    aigs = aigs.copy()
    if max_mcts_inference_seq_length is None:
        max_mcts_inference_seq_length = max_inference_seq_length
    tts_list = []
    enc_action_masks = []
    orig_aig_size = []
    num_inputs_list = []  # Store num_inputs for each circuit
    
    # Process each circuit with dynamic encoding
    for i, aig in enumerate(aigs):
        # Parse AIG if needed
        if isinstance(aig, str):
            # Check if it's a file path or AIG string
            if aig.endswith('.aig') or aig.endswith('.aag'):
                aigs[i], info = read_aiger(aig)
                num_inputs, num_outputs = info[1], info[3]
            else:
                # Assume it's an AIG string
                aigs[i], info = read_aiger(aig_str=aig)
                num_inputs, num_outputs = info[1], info[3]
        else:
            # Already parsed, infer num_inputs by finding max input variable index
            if isinstance(aigs[i], list):
                # Multiple outputs - need to infer from circuit
                num_outputs = len(aigs[i])
            else:
                num_outputs = 1
                aigs[i] = [aigs[i]]
            
            # Find max input variable index in all roots
            from scalable_circuit_transformer_refdfs.utils import get_inputs_rec
            inputs_dict = get_inputs_rec(aigs[i])
            # Remove constant nodes (var == -1)
            input_vars = [var for var in inputs_dict.keys() if var >= 0]
            num_inputs = max(input_vars) + 1 if input_vars else self.default_num_inputs
        
        num_inputs_list.append(num_inputs)
        
        # Check output count (can be more flexible now with scalable model)
        if len(aigs[i]) > self.max_outputs:
            if self.verbose > 0:
                print(f"Warning: circuit {i} has {len(aigs[i])} outputs, exceeding max_outputs {self.max_outputs}")
        
        orig_aig_size.append(count_num_ands(aigs[i]))
        
        # Encode using dynamic encoder
        seq_enc_raw, pos_enc_raw = self.encoder.encode_aig(aigs[i], num_inputs)
        
        # Get input truth table for this circuit size
        # Use same seed as training to ensure consistency between input_tt and tts sampling
        input_tt = self.get_input_tt(num_inputs, seed=self.tt_seed) if input_tts is None else input_tts[i]
        
        # Compute truth tables (adaptive: exact for small, approximate for large)
        # Use same seed as training to ensure consistency with input_tt sampling
        tts = compute_tts_adaptive(
            aigs[i], num_inputs, input_tt=input_tt,
            n_samples=self.tt_num_samples,
            threshold=12,
            seed=self.tt_seed
        )
        
        # Post-process sequence (adds EOS, truncates, pads)
        seq_enc, pos_enc = self._encode_postprocess(seq_enc_raw, pos_enc_raw, num_inputs)
        
        # Generate action masks using dynamic vocab size
        enc_action_mask = self.generate_action_masks(
            tts,
            input_tt,
            None if care_set_tts is None else care_set_tts[i],
            seq_enc_raw,  # Use raw sequence before padding for mask generation
            use_controllability_dont_care=use_controllability_dont_cares,
            num_inputs=num_inputs,
            tts_compressed=None if tts_compressed is None else tts_compressed[i],
            ffw=None if ffws is None else ffws[i]
        )
        
        if enc_action_mask is None:
            if self.verbose > 0:
                print(f"Warning: Failed to generate action masks for circuit {i}, skipping...")
            # Use valid token mask as fallback
            vocab_size = self.max_vocab_size
            valid_mask = self.encoder.get_valid_token_mask(num_inputs)
            enc_action_mask = np.zeros((self.max_seq_length, self.max_vocab_size), dtype=bool)
            # Apply valid_mask to all positions
            for j in range(self.max_seq_length):
                enc_action_mask[j] = valid_mask
        
        enc_action_masks.append(enc_action_mask)
        encoded_aigs.append((seq_enc, pos_enc))
        tts_list.append(tts)
    
    enc_action_masks = np.stack(enc_action_masks)
    seq_enc, pos_enc = tuple(map(lambda x: np.stack(x, axis=0), zip(*encoded_aigs)))
    batch_size = len(aigs)

    inputs = {'inputs': seq_enc, 'enc_pos_encoding': pos_enc, 'enc_action_mask': enc_action_masks}
    targets = np.zeros((batch_size, 1), dtype=np.int32)
    dec_pos_encoding = np.zeros((batch_size, 1, self.max_tree_depth * 2), dtype=np.float32)
    
    # Create environments with correct num_inputs for each circuit
    # Use same seed as training to ensure input_tt consistency with tts sampling
    envs = [LogicNetworkEnv(
        tts=tts_list[i],
        num_inputs=num_inputs_list[i],  # Use dynamic num_inputs
        context_num_inputs=context_num_inputs,
        input_tt=self.get_input_tt(num_inputs_list[i], seed=self.tt_seed) if input_tts is None else input_tts[i],
        init_care_set_tt=None if care_set_tts is None else care_set_tts[i],
        ffw=None if ffws is None else ffws[i],
        error_rate_threshold=self.error_rate_threshold,
        context_hash=None if context_hash_list is None else context_hash_list[i],
        max_tree_depth=self.max_tree_depth,
        max_length=max_inference_seq_length,
        max_inference_reward=None if max_inference_reward_list is None else max_inference_reward_list[i],
        use_controllability_dont_cares=use_controllability_dont_cares,
        tts_compressed=None if tts_compressed is None else tts_compressed[i],
        eos_id=self.eos_id,
        pad_id=self.pad_id,
        w_gate=w_gate,
        w_delay=w_delay,
        verbose=self.verbose,
        use_monte_carlo_tt=num_inputs_list[i] > 12,  # Use MC for large circuits
        mc_tt_threshold=12,
        mc_tt_n_samples=self.tt_num_samples,
        use_memory_dfs=getattr(self, 'use_memory_dfs', False))  # Enable REF_TOKEN support if model uses memory DFS
        for i, aig in enumerate(aigs)]
    
    init_mcts_roots = [MCTSNode(None, 0, None, info={'env': env, 'reward': None, 'done': None, 'rollout_success': None}, puct_explore_ratio=puct_explore_ratio) for env in self._copy_env(envs)]
    # First-step decoder position: must match training (dec_pos[0] = initial stack) and test_autoregressive_inference.
    pos_dim = self.max_tree_depth * 2
    for j, env in enumerate(envs):
        pos0 = stack_to_encoding(
            env.tree_stack, env.cur_root_id, self.max_tree_depth,
            max_outputs=getattr(env, 'max_outputs', 256))
        pos0 = np.array(pos0, dtype=np.float32)
        if len(pos0) > pos_dim:
            pos0 = pos0[:pos_dim]
        elif len(pos0) < pos_dim:
            pos0 = np.concatenate([pos0, np.zeros(pos_dim - len(pos0), dtype=np.float32)])
        dec_pos_encoding[j, 0, :] = pos0
    transformer_time = 0.
    action_mask_time = 0.
    step_time = 0.
    init_time = time.time() - start_time
    if self.verbose > 0:
        print("optimization initialized, time cost %.2f" % init_time)
    
    # Import MCTS method if available
    try:
        from scalable_circuit_transformer_refdfs.mcts_optimization import _batch_MCTS_policy_with_leaf_parallelization
        if not hasattr(self, '_batch_MCTS_policy_with_leaf_parallelization'):
            self._batch_MCTS_policy_with_leaf_parallelization = _batch_MCTS_policy_with_leaf_parallelization.__get__(self, self.__class__)
    except ImportError:
        if self.verbose > 0:
            print("Warning: MCTS optimization not available, MCTS steps will be skipped")
    
    best_token_seqs = None
    mcts_roots = None
    
    # Check if model uses memory DFS (for REF_TOKEN support)
    use_memory_dfs = getattr(self, 'use_memory_dfs', False)
    # Get REF_TOKEN ID - should match training code: ref_token_id = self.max_vocab_size - 1
    # But encoder.REF_TOKEN = self.max_vocab_size, so we need to check which one is correct
    # From training code in scalable_model.py line 369: ref_token_id = self.max_vocab_size - 1
    # But from dynamic_encoding.py: REF_TOKEN = self.max_vocab_size
    # Actually, if use_memory_dfs=True, max_vocab_size already includes REF_TOKEN
    # So REF_TOKEN = max_vocab_size - 1 (the last position)
    ref_token_id = self.max_vocab_size - 1 if use_memory_dfs else None
    
    # Track actual_node_counter for each environment (like in training)
    # This is needed to properly map nodes to seq_position_to_decoded_idx
    actual_node_counters = [0] * len(envs)
    if use_memory_dfs:
        for env in envs:
            env.current_seq_position = 0

    # Decode controls (all optional, default keeps prior behavior)
    inf_T = max(1e-5, getattr(self, 'policy_temperature_in_mcts', 1.0))
    decode_top_k = int(getattr(self, 'decode_top_k_in_inference', 0) or 0)
    sample_when_temp = bool(getattr(self, 'sample_when_temp_in_inference', True))
    low_conf_margin = float(getattr(self, 'decode_low_conf_margin', 0.0) or 0.0)
    decode_beam_size = int(getattr(self, 'decode_beam_size_in_inference', 1) or 1)
    decode_beam_until = int(getattr(self, 'decode_beam_until_step', 0) or 0)
    decode_beam_lookahead_weight = float(getattr(self, 'decode_beam_lookahead_weight', 0.15) or 0.15)
    decode_beam_lookahead_steps = int(getattr(self, 'decode_beam_lookahead_steps_in_inference', 1) or 1)
    decode_beam_logprob_weight = float(getattr(self, 'decode_beam_logprob_weight', 1.0) or 1.0)
    decode_beam_delay_delta_weight = float(getattr(self, 'decode_beam_delay_delta_weight', 0.0) or 0.0)
    decode_beam_step_reward_weight = float(getattr(self, 'decode_beam_step_reward_weight', 0.0) or 0.0)
    decode_beam_area_growth_penalty = float(getattr(self, 'decode_beam_area_growth_penalty', 0.0) or 0.0)
    decode_beam_ref_token_penalty = float(getattr(self, 'decode_beam_ref_token_penalty', 0.0) or 0.0)
    rng = np.random.default_rng(getattr(self, 'tt_seed', 42))

    def _set_memory_state_before_step(env, token, anc):
        """Mirror training/inference state updates for memory DFS."""
        if not use_memory_dfs:
            return anc
        constant_1_id = self.max_vocab_size - 2
        constant_0_id = self.max_vocab_size - 3
        nand_token_id = self.max_vocab_size - 4
        and_token_id = self.max_vocab_size - 5
        input_end = 2 + env.num_inputs * 2
        is_input = (2 <= token < input_end)
        is_gate = (token in [and_token_id, nand_token_id])
        is_const = (token in [constant_0_id, constant_1_id])
        is_ref_token = (token == ref_token_id)
        if is_ref_token:
            env._current_actual_node_pos = anc
            anc += 2
        elif is_input or is_gate or is_const:
            env._current_actual_node_pos = anc
            anc += 1
            env.current_seq_position = anc
        else:
            env._current_actual_node_pos = -1
        return anc

    def _pick_with_lookahead_beam(env_idx, env, masked_logits_row, anc, step_idx, is_ref_position=False):
        """
        Lightweight beam with normalized multi-objective score:
        - keep top-N candidates by current score
        - optional n-step partial rollout reward
        - normalized area/delay deltas for comparable scales.
        """
        finite_idx = np.where(np.isfinite(masked_logits_row) & (masked_logits_row > (np.finfo(np.float32).min / 2)))[0]
        if len(finite_idx) == 0:
            return 0
        enable_beam = (
            decode_beam_size > 1 and
            ((decode_beam_until > 0 and step_idx < decode_beam_until) or is_ref_position)
        )
        if not enable_beam:
            return int(np.argmax(masked_logits_row))
        k = min(decode_beam_size, len(finite_idx))
        top_idx = np.argpartition(masked_logits_row[finite_idx], -k)[-k:]
        candidates = finite_idx[top_idx]
        base_num_ands = count_num_ands(env.roots)
        base_delay = compute_critical_path(env.roots)
        # Candidate prior from log-prob under current masked distribution (temperature-adjusted).
        row_vals = masked_logits_row[finite_idx].astype(np.float64)
        row_vals = row_vals / max(1e-6, inf_T)
        row_vals = row_vals - np.max(row_vals)
        row_probs = np.exp(row_vals)
        row_probs = row_probs / (np.sum(row_probs) + 1e-12)
        cand_logp = {int(tok): float(np.log(max(1e-12, row_probs[idx]))) for idx, tok in enumerate(finite_idx)}
        best_token = int(candidates[np.argmax(masked_logits_row[candidates])])
        best_score = -np.inf
        for tok in candidates:
            tok = int(tok)
            score = decode_beam_logprob_weight * cand_logp.get(tok, -50.0)
            is_ref_token = bool(use_memory_dfs and (ref_token_id is not None) and (tok == ref_token_id) and (not is_ref_position))
            try:
                env_tmp = copy.deepcopy(env)
                anc_tmp = anc
                if is_ref_position:
                    env_tmp._current_actual_node_pos = -1
                    env_tmp.current_seq_position = anc_tmp
                else:
                    anc_tmp = _set_memory_state_before_step(env_tmp, tok, anc_tmp)
                step_reward, _ = env_tmp.step(tok)
                rollout_reward = float(step_reward)
                if decode_beam_lookahead_steps > 1 and not env_tmp.is_finished:
                    # n-step partial rollout value from this candidate state.
                    rem_steps = max(1, decode_beam_lookahead_steps - 1)
                    rem_v, _, _ = self._batch_estimate_v_value_via_simulation_kvcache(
                        [env_tmp],
                        seq_enc[env_idx:env_idx + 1],
                        pos_enc[env_idx:env_idx + 1],
                        enc_action_masks[env_idx:env_idx + 1],
                        max_inference_seq_length=rem_steps,
                        cache=None,
                        num_leaf_parallelization=1,
                    )
                    rollout_reward += float(rem_v[0])
                if env_tmp.is_finished:
                    next_valid = 1
                elif len(env_tmp.action_masks) == 0:
                    next_valid = 0
                else:
                    next_valid = int(np.sum(np.asarray(env_tmp.action_masks[-1], dtype=bool)))
                if decode_beam_step_reward_weight != 0.0:
                    reward_norm = rollout_reward / (
                        max(1.0, float(base_num_ands)) * max(1.0, float(decode_beam_lookahead_steps))
                    )
                    score += decode_beam_step_reward_weight * reward_norm
                if decode_beam_area_growth_penalty > 0.0:
                    next_num_ands = count_num_ands(env_tmp.roots)
                    and_growth_norm = max(0.0, float(next_num_ands - base_num_ands)) / max(1.0, float(base_num_ands))
                    score -= decode_beam_area_growth_penalty * and_growth_norm
                if decode_beam_delay_delta_weight != 0.0:
                    next_delay = compute_critical_path(env_tmp.roots)
                    delay_improve_norm = float(base_delay - next_delay) / max(1.0, float(base_delay))
                    score += decode_beam_delay_delta_weight * delay_improve_norm
                if decode_beam_ref_token_penalty > 0.0 and is_ref_token:
                    score -= decode_beam_ref_token_penalty
                if next_valid <= 0 and not env_tmp.is_finished:
                    score -= 1e3
                else:
                    next_valid_norm = np.log(float(next_valid) + 1.0) / np.log(float(self.max_vocab_size) + 1.0)
                    score += decode_beam_lookahead_weight * next_valid_norm
            except Exception:
                score -= 1e3
            if score > best_score:
                best_score = score
                best_token = tok
        return int(best_token)
    
    for i in range(max_inference_seq_length):
        if all([e.is_finished for e in envs]):
            break
        inputs['targets'], inputs['dec_pos_encoding'] = targets, dec_pos_encoding

        # generate action mask
        start_time = time.time()
        # Use action_masks[-1] (autoregressive decode always uses the latest mask).
        # One outer iteration = one forward + one emitted token + one env.step (REF and ref_position are separate iterations).
        action_masks = np.stack([e.action_masks[-1] for e in envs], axis=0)
        # Pad action masks to max_vocab_size using valid token mask
        padded_action_masks = np.zeros((action_masks.shape[0], self.max_vocab_size), dtype=bool)
        for j, env in enumerate(envs):
            # Copy action mask from environment (which may be smaller)
            env_mask_size = len(action_masks[j])
            if env_mask_size <= self.max_vocab_size:
                padded_action_masks[j, :env_mask_size] = action_masks[j, :env_mask_size]
            else:
                padded_action_masks[j] = action_masks[j, :self.max_vocab_size]
            # Position-token indices can lie outside the structural valid_token mask; keep env mask only
            if not (use_memory_dfs and getattr(env, '_expecting_ref_position', False)):
                valid_mask = self.encoder.get_valid_token_mask(env.num_inputs)
                padded_action_masks[j] = np.logical_and(padded_action_masks[j], valid_mask)
        inputs['dec_action_mask'] = np.expand_dims(padded_action_masks, axis=1)
        action_mask_time += time.time() - start_time

        start_time = time.time()
        policy, cache = self._transformer_inference(inputs, return_kv_cache=True, return_last_token=True)
        transformer_time += time.time() - start_time
        inputs['cache'] = cache

        start_time = time.time()
        # Mask out invalid tokens before selecting
        masked_policy = np.where(padded_action_masks, policy, np.finfo(np.float32).min)
        beam_score_policy = masked_policy / inf_T if inf_T != 1.0 else masked_policy
        # Base selector with optional sampling.
        tokens = _select_tokens_from_masked_logits(
            masked_policy,
            inf_T=inf_T,
            rng=rng,
            top_k=decode_top_k,
            sample_when_temp=sample_when_temp,
            low_conf_margin=low_conf_margin,
        )
        # Optional lightweight beam override for early / REF-sensitive steps.
        if decode_beam_size > 1:
            for j, env in enumerate(envs):
                if env.is_finished:
                    continue
                is_ref_pos = bool(use_memory_dfs and getattr(env, '_expecting_ref_position', False))
                tokens[j] = _pick_with_lookahead_beam(
                    j,
                    env,
                    beam_score_policy[j],
                    actual_node_counters[j],
                    step_idx=i,
                    is_ref_position=is_ref_pos,
                )
        
        # Set _current_actual_node_pos for each environment before step (like in training)
        # This is crucial for memory DFS to properly track node positions
        if use_memory_dfs:
            # Define token IDs (matching training code)
            constant_1_id = self.max_vocab_size - 2
            constant_0_id = self.max_vocab_size - 3
            nand_token_id = self.max_vocab_size - 4
            and_token_id = self.max_vocab_size - 5
            
            for j, (token, env) in enumerate(zip(tokens, envs)):
                if env.is_finished:
                    continue
                
                # Ref position token (next iteration after REF): training sets state before step(ref_pos)
                if getattr(env, '_expecting_ref_position', False):
                    env._current_actual_node_pos = -1
                    env.current_seq_position = actual_node_counters[j]
                    continue
                
                # Check token type
                input_end = 2 + env.num_inputs * 2
                is_input = (2 <= token < input_end)
                is_gate = (token in [and_token_id, nand_token_id])
                is_const = (token in [constant_0_id, constant_1_id])
                is_ref_token = (ref_token_id is not None) and (token == ref_token_id)
                
                if is_ref_token:
                    # REF_TOKEN: set actual_node_counter BEFORE incrementing (matching training code)
                    env._current_actual_node_pos = actual_node_counters[j]
                    # Increment BEFORE step (matching training code line 406)
                    actual_node_counters[j] += 2  # REF_TOKEN + position = 2 positions
                elif is_input or is_gate or is_const:
                    # Regular node: set actual_node_counter BEFORE incrementing (matching training code)
                    env._current_actual_node_pos = actual_node_counters[j]
                    # Increment BEFORE step (matching training code line 436)
                    actual_node_counters[j] += 1
                    env.current_seq_position = actual_node_counters[j]  # Set to incremented value
                else:
                    # Other tokens (EOS, PAD, etc.): reset
                    env._current_actual_node_pos = -1

        if num_mcts_steps > 0 and hasattr(self, '_batch_MCTS_policy_with_leaf_parallelization'):
            if i < num_mcts_steps:
                best_token_seqs, mcts_roots = self._batch_MCTS_policy_with_leaf_parallelization(envs,
                                                                      max_inference_seq_length=max_mcts_inference_seq_length,
                                                                      num_leaf_parallelizations=num_leaf_parallelization,
                                                                      num_playouts=num_mcts_playouts_per_step,
                                                                      src_tokens=seq_enc,
                                                                      src_pos_enc=pos_enc,
                                                                      src_action_mask=enc_action_masks,
                                                                      roots=init_mcts_roots if i == 0 else mcts_roots,
                                                                      orig_aigs_size=orig_aig_size,
                                                                      puct_explore_ratio=puct_explore_ratio)
                tokens = [b[0] for b in best_token_seqs]
            else:
                if i == num_mcts_steps:
                    if self.verbose > 1:
                        print("best_token_seqs", best_token_seqs)
                if best_token_seqs is not None:
                    for j, b in enumerate(best_token_seqs):
                        if len(b) >= i - num_mcts_steps + 2:
                            tokens[j] = b[i - num_mcts_steps + 1]

        # One env.step per outer iteration; REF consumes one step, ref_position is emitted on the next iteration.
        rewards, dones = zip(*[e.step(int(token)) if not e.is_finished else (0, True) for token, e in zip(tokens, envs)])
        
        pos_encodings = [e.positional_encodings[-1] for e in envs]
        
        # Truncate to max_tree_depth * 2 to match model expectations
        # (stack_to_encoding returns max_tree_depth * 2 + 8, but model expects max_tree_depth * 2)
        pos_encodings = [pos_enc[:self.max_tree_depth * 2] for pos_enc in pos_encodings]

        pos_encodings = np.expand_dims(np.stack(pos_encodings, axis=0),
                                       axis=1)  # [batch_size, 1, max_tree_depth * 2]
        targets_new = np.expand_dims(tokens, axis=1)
        if self.use_kv_cache:
            targets = targets_new
            dec_pos_encoding = pos_encodings
        else:
            targets = np.concatenate([targets, targets_new], axis=1)
            dec_pos_encoding = np.concatenate([dec_pos_encoding, pos_encodings], axis=1)
        step_time += time.time() - start_time
        if self.verbose > 0:
            print(i, tokens.tolist() if hasattr(tokens, 'tolist') else tokens)
    
    if return_envs and self.verbose == 0:
        return envs
    
    optimized_aigs = []
    num_succeed_aigs = 0
    total_gain, seq_total_gain_for_succeeded_aig, seq_total_gain = 0, 0, 0
    seq_opt_total_gain_for_succeeded_aig = 0
    for i, (aig, env) in enumerate(zip(aigs, envs)):
        orig_num_ands = count_num_ands(aig)
        # if self.verbose > 1:
        seq_roots, info = sequential_synthesis(aig, num_inputs=num_inputs_list[i])
        orig_resyn_num_ands = info[4]
        seq_total_gain += orig_num_ands - orig_resyn_num_ands
        
        # Detailed failure diagnosis
        num_roots_match = len(env.roots) == len(aig)
        integrity_ok = check_integrity(env.roots)
        expected_success = num_roots_match and integrity_ok
        
        assert env.success == expected_success, \
            f"Success flag mismatch: env.success={env.success}, " \
            f"expected={expected_success} (roots match: {num_roots_match}, " \
            f"integrity: {integrity_ok}, num_roots: {len(env.roots)}, expected: {len(aig)})"
        
        if env.success:
            num_succeed_aigs += 1
            num_ands = count_num_ands(env.roots)
            opt_resyn_roots, opt_resyn_info = sequential_synthesis(
                env.roots, title=f"sequential_opt_{i}", num_inputs=num_inputs_list[i]
            )
            opt_resyn_num_ands = opt_resyn_info[4]
            if not return_envs:
                optimized_aigs.append(opt_resyn_roots)
            if self.verbose > 0:
                total_gain += max(orig_num_ands - num_ands, 0)
                seq_opt_total_gain_for_succeeded_aig += orig_num_ands - opt_resyn_num_ands
                print("aig #%d successfully optimized, #(AND) from %d to %d, cumulative reward %d, gain = %d" %
                      (i, orig_num_ands, num_ands, env.cumulative_reward, orig_num_ands - num_ands),
                      end="" if self.verbose > 1 else "\n")
                # if self.verbose > 1:
                seq_total_gain_for_succeeded_aig += orig_num_ands - orig_resyn_num_ands
                print(" (orig=%d, orig+resyn2=%d, optimize=%d, optimize+resyn2=%d)" %
                      (orig_num_ands, orig_resyn_num_ands, num_ands, opt_resyn_num_ands))
                
                # 计算优化后电路的真值表 (使用 Monte Carlo 或 精确计算，取决于输入规模)
                current_num_inputs = num_inputs_list[i]
                current_input_tt = self.get_input_tt(current_num_inputs, seed=self.tt_seed) if input_tts is None else input_tts[i]
                
                opt_tts = compute_tts_adaptive(
                    opt_resyn_roots, current_num_inputs, input_tt=current_input_tt,
                    n_samples=self.tt_num_samples,
                    threshold=12,
                    seed=self.tt_seed
                )
                
                # 原电路的真值表已经在 tts_list[i] 中
                orig_tts = tts_list[i]
                
                if len(orig_tts) > 0:
                    length = len(orig_tts[0])
                    diff_map = bitarray.bitarray(length)
                    diff_map.setall(0)
                    
                    # 比较每一路输出
                    for t, o in zip(orig_tts, opt_tts):
                        diff_map |= (t ^ o)
                    
                    num_diff_tt = diff_map.count(1)
                    error_rate = num_diff_tt / length
                    
                    print(f"Total columns (tt): {length}")
                    print(f"Different columns: {num_diff_tt}")
                    print(f"Error Rate: {error_rate:.4f} ({error_rate * 100:.2f}%)")
        else:
            if self.verbose > 0:
                failure_reason = []
                if len(env.roots) != len(aig):
                    failure_reason.append(f"root count mismatch ({len(env.roots)} != {len(aig)})")
                if not check_integrity(env.roots):
                    failure_reason.append("integrity check failed")
                if not env.gen_eos:
                    failure_reason.append("EOS not generated")
                reason_str = f" ({', '.join(failure_reason)})" if failure_reason else ""
                print("aig #%d (#(AND) = %d) failed to be optimized%s%s" %
                      (i, orig_num_ands, ', use original aig instead' if overflow_option == 'origin' else '', reason_str))
                if self.verbose > 1:
                    print(" (resyn2: %d)" % info[4])
            if not return_envs:
                optimized_aigs.append(aig if overflow_option == 'origin' else None)
    
    if self.verbose > 0:
        print(
            "%d out of %d aigs successfully optimized, total time %.2f, init time %.2f, transformer time %.2f, action mask time %.2f, step time %.2f" %
            (num_succeed_aigs, len(aigs), time.time() - total_time, init_time, transformer_time, action_mask_time, step_time))
        if num_succeed_aigs > 0:
            print("average gain %.3f for successfully optimized aigs" % (total_gain / num_succeed_aigs))
        print("average gain %.3f for all aigs (failed aigs correspond to zero gain)" % (total_gain / len(aigs)))
        if self.verbose > 1:
            if num_succeed_aigs > 0:
                print("resyn2 (orig): %.3f / %.3f" % (
                    seq_total_gain_for_succeeded_aig / num_succeed_aigs, seq_total_gain / len(aigs)))
                print("resyn2 (optimize): %.3f" % (
                    seq_opt_total_gain_for_succeeded_aig / num_succeed_aigs))
    
    if not return_envs and not return_mcts_roots:
        return optimized_aigs
    
    ret = []
    if return_envs:
        ret.append(envs)
    if return_mcts_roots:
        ret.append(init_mcts_roots)
    if return_input_encodings:
        ret.append({'inputs': seq_enc, 'enc_pos_encoding': pos_enc, 'enc_action_mask': enc_action_masks})
    return tuple(ret) if len(ret) > 1 else ret[0]

