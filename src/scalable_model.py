"""
Scalable Circuit Transformer Model

This module implements a circuit transformer that can handle variable input/output sizes
through dynamic vocabulary and adaptive attention masking.

Key Features:
- Dynamic vocabulary size based on circuit complexity
- Efficient embedding with padding for unused tokens
- Adaptive attention masking
- Backward compatible with fixed-size models
"""

from __future__ import annotations
import os
from typing import Optional
import numpy as np
import tensorflow as tf
import tf_keras as keras
import npn
from official.nlp import modeling as nlp
from scalable_circuit_transformer_refdfs.dynamic_encoding import DynamicEncoder
from scalable_circuit_transformer_refdfs.tensorflow_transformer import Seq2SeqTransformer
from scalable_circuit_transformer_refdfs.utils import *
from scalable_circuit_transformer_refdfs.environment import LogicNetworkEnv, ActionMaskTimeoutError
from scalable_circuit_transformer_refdfs.encoding import get_pos_encoding_n_vars
from scalable_circuit_transformer_refdfs.monte_carlo_tt import (
    compute_tts_adaptive,
    compute_input_tt_approximate,
    generate_input_samples
)


class ScalableCircuitTransformer:
    """
    A circuit transformer that dynamically adapts to different circuit sizes.

    Unlike the fixed CircuitTransformer which only supports 8 inputs, this model
    can handle arbitrary numbers of inputs and outputs by using:
    1. Dynamic vocabulary size calculation
    2. Embedding padding for unused tokens
    3. Runtime attention masking based on actual circuit size
    """

    def __init__(self,
                 max_inputs=256,
                 max_outputs=256,
                 embedding_width=512,
                 num_layers=12,
                 num_attention_heads=8,
                 intermediate_size=2048,
                 max_tree_depth=128,
                 max_seq_length=500,
                 inference_batch_size=512,
                 eos_id=1,
                 pad_id=0,
                 verbose=0,
                 mixed_precision=True,
                 ckpt_path=None,
                 batch_size=8,
                 add_action_mask_to_input=False,
                 policy_temperature_in_mcts=1.,
                 w_gate=1,
                 w_delay=0,
                 default_num_inputs=8,
                 use_memory_dfs=False,
                 tt_num_samples=4096,
                 error_rate_threshold=0.1,
                 tt_seed=42,
                 decode_top_k_in_inference=0,
                 sample_when_temp_in_inference=True,
                 decode_low_conf_margin=0.0,
                 decode_beam_size_in_inference=1,
                 decode_beam_until_step=0,
                 decode_beam_lookahead_weight=0.15,
                 decode_beam_lookahead_steps_in_inference=1,
                 decode_beam_logprob_weight=1.0,
                 decode_beam_delay_delta_weight=0.0,
                 decode_beam_step_reward_weight=0.0,
                 decode_beam_area_growth_penalty=0.0,
                 decode_beam_ref_token_penalty=0.0):
        """
        Initialize Scalable Circuit Transformer.

        Args:
            max_inputs: Maximum number of inputs supported (default: 64)
            max_outputs: Maximum number of outputs supported (default: 64)
            embedding_width: Embedding dimension for transformer (default: 512)
            num_layers: Number of transformer layers (default: 12)
            num_attention_heads: Number of attention heads (default: 8)
            intermediate_size: FFN intermediate size (default: 2048)
            max_tree_depth: Maximum tree depth for positional encoding (default: 32)
            max_seq_length: Maximum sequence length (default: 500)
            inference_batch_size: Batch size for inference (default: 512)
            eos_id: End-of-sequence token ID (default: 1)
            pad_id: Padding token ID (default: 0)
            verbose: Verbosity level (default: 0)
            mixed_precision: Use mixed precision training (default: True)
            ckpt_path: Path to checkpoint for loading weights (default: None)
            batch_size: Training batch size (default: 8)
            add_action_mask_to_input: Add action mask to input embeddings (default: False)
            policy_temperature_in_mcts: Temperature for policy in MCTS (default: 1.0)
            w_gate: Gate weight for reward (default: 1)
            w_delay: Delay weight for reward (default: 0)
            default_num_inputs: Default number of inputs for backward compatibility (default: 8)
            use_memory_dfs: Whether to use memory DFS encoding (default: False)
        """
        # Store configuration
        self.max_inputs = max_inputs
        self.max_outputs = max_outputs
        self.default_num_inputs = default_num_inputs
        # Base vocab size: PAD, EOS, inputs, gates, constants
        # If using memory DFS, add 1 for REF_TOKEN
        base_vocab_size = 2 + 2 * max_inputs + 4
        self.max_vocab_size = base_vocab_size + (1 if use_memory_dfs else 0)  # Add REF_TOKEN if using memory DFS
        self.embedding_width = embedding_width
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_tree_depth = max_tree_depth
        self.max_seq_length = max_seq_length
        self.inference_batch_size = inference_batch_size
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.verbose = verbose
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size
        self.add_action_mask_to_input = add_action_mask_to_input
        self.policy_temperature_in_mcts = policy_temperature_in_mcts
        self.w_gate = w_gate
        self.w_delay = w_delay
        self.error_rate_threshold = error_rate_threshold
        # Approximate TT parameters
        self.tt_threshold_inputs = 50  # Use exact TT if num_inputs <= 12, approximate if > 12
        self.tt_num_samples = tt_num_samples  # Number of samples for Monte Carlo approximation
        self.tt_seed = tt_seed  # Random seed for Monte Carlo sampling (ensures consistency)
        # Inference decode controls (used by sact_inference.py)
        self.decode_top_k_in_inference = decode_top_k_in_inference
        self.sample_when_temp_in_inference = sample_when_temp_in_inference
        self.decode_low_conf_margin = decode_low_conf_margin
        self.decode_beam_size_in_inference = decode_beam_size_in_inference
        self.decode_beam_until_step = decode_beam_until_step
        self.decode_beam_lookahead_weight = decode_beam_lookahead_weight
        self.decode_beam_lookahead_steps_in_inference = decode_beam_lookahead_steps_in_inference
        self.decode_beam_logprob_weight = decode_beam_logprob_weight
        self.decode_beam_delay_delta_weight = decode_beam_delay_delta_weight
        self.decode_beam_step_reward_weight = decode_beam_step_reward_weight
        self.decode_beam_area_growth_penalty = decode_beam_area_growth_penalty
        self.decode_beam_ref_token_penalty = decode_beam_ref_token_penalty

        # Initialize dynamic encoder
        self.use_memory_dfs = use_memory_dfs
        self.encoder = DynamicEncoder(
            max_inputs=max_inputs,
            max_outputs=max_outputs,
            max_seq_length=max_seq_length,
            use_memory_dfs=use_memory_dfs
        )

        # Mixed precision setup
        if mixed_precision:
            keras.mixed_precision.set_global_policy('mixed_float16')

        # Build transformer
        self._transformer = self._build_transformer()

        # Load checkpoint if provided
        if ckpt_path is not None:
            self.load(ckpt_path)

        # Cache for different input sizes
        self._input_tt_cache = {}
        
        # Setup transformer inference method with KV cache support
        import types
        @tf.function(reduce_retracing=True)
        def _transformer_inference_graph(self, inputs, return_kv_cache=False, return_last_token=False):
            policy, cache = self._transformer(inputs, return_kv_cache=return_kv_cache, return_last_token=return_last_token)
            return policy, cache

        def _transformer_inference(self, inputs, return_kv_cache=False, return_last_token=False):
            policy, cache = _transformer_inference_graph(self, inputs, return_kv_cache=return_kv_cache, return_last_token=return_last_token)
            return policy.numpy(), cache

        self._transformer_inference = types.MethodType(_transformer_inference, self)
        self._transformer.return_cache = True
        self.use_kv_cache = True

        # Bind inference methods from sact_inference module
        from scalable_circuit_transformer_refdfs.sact_inference import (
            _batch_estimate_policy,
            _batch_estimate_v_value_via_simulation_kvcache,
            optimize,
            optimize_batch
        )
        self._batch_estimate_policy = types.MethodType(_batch_estimate_policy, self)
        self._batch_estimate_v_value_via_simulation_kvcache = types.MethodType(_batch_estimate_v_value_via_simulation_kvcache, self)
        self.optimize = types.MethodType(optimize, self)
        self.optimize_batch = types.MethodType(optimize_batch, self)

        if self.verbose > 0:
            print(f"Initialized ScalableCircuitTransformer:")
            print(f"  Max inputs: {max_inputs}, Max outputs: {max_outputs}")
            print(f"  Max vocab size: {self.max_vocab_size}")
            print(f"  Max sequence length: {max_seq_length}")
            print(f"  Embedding width: {embedding_width}")

    def _build_transformer(self) -> Seq2SeqTransformer:
        """Build the underlying transformer model."""
        transformer = Seq2SeqTransformer(
            enc_vocab_size=self.max_vocab_size,
            dec_vocab_size=self.max_vocab_size,
            embedding_width=self.embedding_width,
            encoder_layer=nlp.models.TransformerEncoder(
                num_layers=self.num_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size
            ),
            decoder_layer=nlp.models.TransformerDecoder(
                num_layers=self.num_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size
            ),
            eos_id=self.eos_id,
            max_tree_depth=self.max_tree_depth,
            add_action_mask_to_inputs=self.add_action_mask_to_input
        )
        return transformer

    def get_vocab_size(self, num_inputs: int) -> int:
        """
        Get vocabulary size for a given number of inputs.
        
        Note: Returns max_vocab_size (fixed) regardless of num_inputs.
        The actual valid tokens are determined by get_valid_token_mask().
        If using memory DFS, max_vocab_size already includes REF_TOKEN.
        """
        return self.max_vocab_size
    
    def get_constant_token_ids(self, num_inputs: int) -> tuple[int, int]:
        """Get token IDs for constant 0 and constant 1 (fixed positions)."""
        constant_0_id = self.max_vocab_size - 2
        constant_1_id = self.max_vocab_size - 1
        return constant_0_id, constant_1_id

    def get_input_tt(self, num_inputs: int, seed: Optional[int] = None):
        """
        Get or compute input truth table for num_inputs.

        Uses exact computation for small circuits (<=12 inputs),
        approximate computation for larger circuits.
        
        For Monte Carlo approximation, seed ensures consistency with output TT.
        """
        if num_inputs <= 12:
            # Use exact computation - can cache
            if num_inputs not in self._input_tt_cache:
                self._input_tt_cache[num_inputs] = compute_input_tt(num_inputs)
            return self._input_tt_cache[num_inputs]
        else:
            # Use approximate computation with sampling
            # Don't cache Monte Carlo TT - regenerate with same seed to ensure consistency
            if seed is None:
                seed = self.tt_seed
            return compute_input_tt_approximate(num_inputs, self.tt_num_samples, seed=seed)

    def _encode_postprocess(self, seq_enc: list[int], pos_enc: list[int], num_inputs: int):
        """
        Postprocess encoded sequence and position encoding.

        Args:
            seq_enc: Token sequence
            pos_enc: Position encoding
            num_inputs: Number of inputs (for validation)

        Returns:
            Tuple of (padded_seq_enc, padded_pos_enc)
        """
        # Add EOS token
        seq_enc.append(self.eos_id)

        # Warn if sequence is too long
        if self.verbose > 0 and len(seq_enc) > self.max_seq_length:
            print(f"Warning: seq_enc length {len(seq_enc)} > max seq length ({self.max_seq_length})")

        # Truncate to max length
        seq_enc = seq_enc[:self.max_seq_length]
        pos_enc = pos_enc[:self.max_seq_length]

        # Convert position encoding to binary representation
        from scalable_circuit_transformer_refdfs.encoding import get_pos_encoding_n_vars, _int_to_binary_lsb, _NPN_INT_TO_TT_MAX_VARS
        
        n_vars = get_pos_encoding_n_vars(self.max_tree_depth, self.max_outputs)

        new_pos_enc = []
        for pos_val in pos_enc:
            if n_vars <= _NPN_INT_TO_TT_MAX_VARS:
                binary_list = npn.int_to_tt(pos_val, n_vars)
            else:
                binary_list = _int_to_binary_lsb(pos_val, n_vars)
            
            new_pos_enc.append(list(reversed(binary_list)))
        
        pos_enc = np.stack(new_pos_enc, axis=0)

        seq_enc = np.array(seq_enc + [0] * (self.max_seq_length - len(seq_enc)), dtype=np.int32)
        pos_enc = np.concatenate(
            [pos_enc, np.zeros((self.max_seq_length - len(pos_enc), self.max_tree_depth * 2))],
            axis=0,
            dtype=np.float32
        )

        return seq_enc, pos_enc

    def encode_circuit(self, roots: list[NodeWithInv], num_inputs: int) -> tuple:
        """
        Encode a circuit to model inputs.

        Args:
            roots: List of output nodes
            num_inputs: Number of inputs in the circuit

        Returns:
            Tuple of (seq_enc, pos_enc)
        """
        seq_enc, pos_enc = self.encoder.encode_aig(roots, num_inputs)
        # print(len(seq_enc))
        # print(len(pos_enc))
        return self._encode_postprocess(seq_enc, pos_enc, num_inputs)

    def _format_tree_stack(self, tree_stack):
        """Format tree_stack for debugging output."""
        if not tree_stack:
            return "[]"
        result = []
        for i, node in enumerate(tree_stack):
            if node.is_leaf():
                result.append(f"Leaf(var={node.var}, inv={node.inverted})")
            else:
                left_str = "None" if node.left is None else ("Leaf" if node.left.is_leaf() else "Node")
                right_str = "None" if node.right is None else ("Leaf" if node.right.is_leaf() else "Node")
                result.append(f"Node(left={left_str}, right={right_str}, inv={node.inverted})")
        return "[" + ", ".join(result) + "]"
    
    def _format_roots(self, roots):
        """Format roots for debugging output."""
        if not roots:
            return "[]"
        result = []
        for i, root in enumerate(roots):
            if root is None:
                result.append("None")
            elif root.is_leaf():
                result.append(f"Leaf(var={root.var}, inv={root.inverted})")
            else:
                left_str = "None" if root.left is None else ("Leaf" if root.left.is_leaf() else "Node")
                right_str = "None" if root.right is None else ("Leaf" if root.right.is_leaf() else "Node")
                result.append(f"Node(left={left_str}, right={right_str}, inv={root.inverted})")
        return "[" + ", ".join(result) + "]"
    
    def _format_sequence(self, seq_enc, max_show=50):
        """Format sequence for debugging output."""
        if len(seq_enc) <= max_show:
            return str(seq_enc.tolist() if isinstance(seq_enc, np.ndarray) else list(seq_enc))
        else:
            first_part = seq_enc[:max_show//2]
            last_part = seq_enc[-max_show//2:]
            return f"{first_part.tolist() if isinstance(first_part, np.ndarray) else list(first_part)} ... ({len(seq_enc) - max_show} tokens) ... {last_part.tolist() if isinstance(last_part, np.ndarray) else list(last_part)}"

    def generate_action_masks(self, tts, input_tt, care_set_tts, seq_enc,
                                use_controllability_dont_care, num_inputs,
                                tts_compressed=None, ffw=None):
            """
            Generate action masks for a given sequence with corrected timing logic.
            """
            # 1. Create environment
            try:
                env = LogicNetworkEnv(
                    tts, num_inputs, init_care_set_tt=care_set_tts, ffw=ffw,
                    input_tt=input_tt, max_length=self.max_seq_length,
                    max_tree_depth=self.max_tree_depth, max_inference_tree_depth=self.max_tree_depth,
                    use_controllability_dont_cares=use_controllability_dont_care,
                    tts_compressed=tts_compressed, w_gate=self.w_gate, w_delay=self.w_delay,
                    and_always_available=True,
                    use_monte_carlo_tt=num_inputs > 12, mc_tt_threshold=12,
                    mc_tt_n_samples=self.tt_num_samples,
                    mc_tt_seed=self.tt_seed,  # Use same seed for consistent Monte Carlo sampling
                    use_memory_dfs=self.use_memory_dfs, verbose=self.verbose, error_rate_threshold=self.error_rate_threshold
                )
            except Exception as e:
                if self.verbose >= 1: print(f"[ERROR] Env creation failed: {e}")
                return None

            action_masks = []
            valid_mask = self.encoder.get_valid_token_mask(num_inputs)
            
            # 2. Special token ids (last slots of max_vocab_size); keep in sync with LogicNetworkEnv.gen_action_mask
            ref_token_id = self.max_vocab_size - 1      # 518
            constant_1_id = self.max_vocab_size - 2     # 517
            constant_0_id = self.max_vocab_size - 3     # 516
            nand_token_id = self.max_vocab_size - 4     # 515
            and_token_id = self.max_vocab_size - 5      # 514
            
            # 3. Counters
            actual_node_counter = 0 
            if self.use_memory_dfs:
                env.current_seq_position = 0 

            # Helper: pad env mask to model vocab and optionally intersect valid_mask
            def _process_and_append_mask(env_mask, apply_valid_mask=True):
                padded_mask = np.zeros(self.max_vocab_size, dtype=bool)
                curr_len = len(env_mask)
                limit = min(curr_len, self.max_vocab_size)
                padded_mask[:limit] = env_mask[:limit]
                
                # Apply valid_mask only when requested
                if apply_valid_mask:
                    padded_mask = np.logical_and(padded_mask, valid_mask)
                
                action_masks.append(padded_mask)

            idx = 0
            try:
                while idx < len(seq_enc[:self.max_seq_length]):
                    # ...
                    token = seq_enc[idx]

                    _process_and_append_mask(env.action_masks[-1], apply_valid_mask=True)

                    if self.use_memory_dfs and token == ref_token_id:
                        if idx + 1 >= len(seq_enc): return None 

                        # 1. Env state for REF pair
                        env._current_actual_node_pos = actual_node_counter
                        # REF pair is atomic in seq-position semantics: REF + position consume 2 slots.
                        actual_node_counter += 2
                        
                        # 2. Step REF_TOKEN
                        env.step(token) 

                        # 3. Mask for ref_position (position index; do not apply valid_mask)
                        _process_and_append_mask(env.action_masks[-1], apply_valid_mask=False)

                        # 4. Step ref_position token
                        ref_pos_token = seq_enc[idx + 1]
                        if env._expecting_ref_position is False:
                            # Contract check: after stepping REF_TOKEN, env must expect a position token.
                            return None
                        env._current_actual_node_pos = -1  # position index is not a node
                        env.current_seq_position = actual_node_counter
                        env.step(ref_pos_token)

                        # 5. Keep env counters in sync

                        idx += 2  # skip [REF, POS]
                        continue


                    input_end = 2 + num_inputs * 2
                    is_input = (2 <= token < input_end)
                    is_gate = (token in [and_token_id, nand_token_id])
                    is_const = (token in [constant_0_id, constant_1_id])

                    if self.use_memory_dfs and (is_input or is_gate or is_const):
                        env._current_actual_node_pos = actual_node_counter
                        actual_node_counter += 1
                        env.current_seq_position = actual_node_counter
                    else:
                        env._current_actual_node_pos = -1

                    env.step(token)
                    
                    idx += 1

            except ActionMaskTimeoutError:
                return None
            except Exception as e:
                if self.verbose >= 1: print(f"[ERROR] generate_action_masks exception: {e}")
                import traceback
                traceback.print_exc()
                return None

            # 4. Pad action masks to max_seq_length
            action_masks = np.stack(action_masks)
            if len(action_masks) < self.max_seq_length:
                padding = np.zeros((self.max_seq_length - len(action_masks), self.max_vocab_size), dtype=bool)
                padding[:, 0] = True  # allow PAD in padded tail
                action_masks = np.concatenate([action_masks, padding], axis=0)
            return action_masks

    def load(self, ckpt_path: str):
        """Load model weights from checkpoint."""
        status = self._transformer.load_weights(ckpt_path)
        status.expect_partial()
        self.ckpt_path = ckpt_path
        if self.verbose > 0:
            print(f"Loaded checkpoint from {ckpt_path}")

    def save(self, ckpt_path: str):
        """Save model weights to checkpoint."""
        self._transformer.save_weights(ckpt_path)
        if self.verbose > 0:
            print(f"Saved checkpoint to {ckpt_path}")

    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        summary_lines = [
            "=" * 70,
            "Scalable Circuit Transformer Summary",
            "=" * 70,
            f"Max Inputs:          {self.max_inputs}",
            f"Max Outputs:         {self.max_outputs}",
            f"Max Vocab Size:      {self.max_vocab_size}",
            f"Embedding Width:     {self.embedding_width}",
            f"Num Layers:          {self.num_layers}",
            f"Attention Heads:     {self.num_attention_heads}",
            f"Intermediate Size:   {self.intermediate_size}",
            f"Max Seq Length:      {self.max_seq_length}",
            f"Max Tree Depth:      {self.max_tree_depth}",
            "=" * 70
        ]
        return "\n".join(summary_lines)

    def infer_circuit_size(self, roots: list[NodeWithInv]) -> dict:
        """
        Infer circuit size parameters from roots.

        Args:
            roots: List of output nodes

        Returns:
            Dictionary with circuit metadata
        """
        return self.encoder.get_circuit_metadata(roots)

    def _copy_env(self, env: LogicNetworkEnv | list[LogicNetworkEnv]):
        """Copy environment(s) for MCTS simulations."""
        if isinstance(env, list):
            return [self._copy_env(e) for e in env]
        else:
            import copy
            context_hash, tts_bitarray, input_tt_bitarray, input_tt_bitarray_compressed, ffw = \
                env.context_hash, env.tts_bitarray, env.input_tt_bitarray, env.input_tt_bitarray_compressed, env.ffw
            env.context_hash, env.tts, env.input_tt, env.input_tt_bitarray_compressed, env.ffw = None, None, None, None, None
            res = copy.deepcopy(env)
            env.context_hash, env.tts_bitarray, env.input_tt_bitarray, env.input_tt_bitarray_compressed, env.ffw = \
                context_hash, tts_bitarray, input_tt_bitarray, input_tt_bitarray_compressed, ffw
            res.context_hash, res.tts_bitarray, res.input_tt_bitarray, res.input_tt_bitarray_compressed, res.ffw = \
                context_hash, tts_bitarray, input_tt_bitarray, input_tt_bitarray_compressed, ffw
            return res

    @property
    def transformer(self):
        """Access to underlying transformer model."""
        return self._transformer

    def load_and_encode(self, filename):
        """
        Load and encode circuit data from JSON file.

        Args:
            filename: Path to JSON file containing circuit data

        Returns:
            Tuple of (seq_enc, pos_enc, opt_seq_enc, opt_pos_enc, enc_action_mask, dec_action_mask)
        """
        import json

        # Check if file exists before opening (may have been deleted by another process)
        if not os.path.exists(filename):
            if self.verbose > 0:
                print(f"[Skip] File {filename} does not exist (may have been deleted by another process).")
            return None, None, None, None, None, None

        try:
            with open(filename, 'r') as f:
                roots_aiger, num_ands, opt_roots_aiger, opt_num_ands = json.load(f)
        except FileNotFoundError:
            # File was deleted between existence check and opening (race condition in multiprocessing)
            if self.verbose > 2:
                print(f"[Skip] File {filename} was deleted during processing.")
            return None, None, None, None, None, None

        roots, info = read_aiger(aiger_str=roots_aiger)
        opt_roots, _ = read_aiger(aiger_str=opt_roots_aiger)
        
        num_inputs, num_outputs = info[1], info[3]

        # print(f"num_inputs: {num_inputs}")
        # print(f"num_outputs: {num_outputs}")

        # Note: No need to check input count limit anymore
        # Monte Carlo method uses sampling (fixed memory O(n_samples)) instead of 
        # full truth table (exponential memory O(2^num_inputs))
        # So we can handle circuits with any number of inputs

        # Ensure roots is always a list
        if not isinstance(roots, list):
            roots = [roots]
        if not isinstance(opt_roots, list):
            opt_roots = [opt_roots]

        # Data augmentation with NPN transformation
        phase = np.random.rand(num_inputs) < 0.5
        perm = np.random.permutation(num_inputs)
        output_invs = np.random.rand(num_outputs) < 0.5
        roots = npn_transform(roots, phase, perm, output_invs)
        opt_roots = npn_transform(opt_roots, phase, perm, output_invs)

        # Ensure roots is still a list after NPN transform
        if not isinstance(roots, list):
            roots = [roots]
        if not isinstance(opt_roots, list):
            opt_roots = [opt_roots]

        # Encode circuits
        # First encode without postprocessing to check length
        seq_enc_raw, pos_enc_raw = self.encoder.encode_aig(roots, num_inputs)
        # Check length before postprocessing to avoid warning
        if len(seq_enc_raw) + 1 > self.max_seq_length:  # +1 for EOS token
            # if self.verbose > 0:
            # print(f"[Skip] seq_enc length {len(seq_enc_raw)} > max_seq_length ({self.max_seq_length}), deleting file {filename}.")
            # Delete file if sequence exceeds max_seq_length
            # try:
            #     os.remove(filename)
            #     if self.verbose > 0:
            #         print(f"[Deleted] File {filename} has been deleted.")
            # except Exception as e:
            #     if self.verbose > 0:
            #         print(f"[Warning] Failed to delete file {filename}: {e}")
            return None, None, None, None, None, None
        
        # Now encode with postprocessing (skip_warning=True since we already checked)
        seq_enc, pos_enc = self._encode_postprocess(seq_enc_raw, pos_enc_raw, num_inputs)
        
        opt_seq_enc_raw, opt_pos_enc_raw = self.encoder.encode_aig(opt_roots, num_inputs)
        # Check optimized sequence length too
        if len(opt_seq_enc_raw) + 1 > self.max_seq_length:  # +1 for EOS token
            # if self.verbose > 0:
            # print(f"[Skip] opt_seq_enc length {len(opt_seq_enc_raw)} > max_seq_length ({self.max_seq_length}), deleting file {filename}.")
            # Delete file if optimized sequence exceeds max_seq_length
            # try:
            #     os.remove(filename)
            #     if self.verbose > 0:
            #         print(f"[Deleted] File {filename} has been deleted.")
            # except Exception as e:
            #     if self.verbose > 0:
            #         print(f"[Warning] Failed to delete file {filename}: {e}")
            return None, None, None, None, None, None

        # Get input truth table for this circuit size
        # For Monte Carlo, use same seed to ensure input_tt and tts use same samples
        input_tt = self.get_input_tt(num_inputs, seed=self.tt_seed)
        # print(f"input_tt: {input_tt}")
        # print(f"len(input_tt): {len(input_tt)}")

        # Compute truth tables for original circuit (adaptive: exact for small, approximate for large)
        tts = compute_tts_adaptive(
            roots, num_inputs, input_tt=input_tt,
            n_samples=self.tt_num_samples,
            threshold=12,
            seed=self.tt_seed
        )
        # print(f"tts: {tts}")
        # print(f"len(tts): {len(tts)}")

        # Compute truth tables for optimized circuit
        # Note: opt_roots should implement the same function, but may have different structure
        # We need to compute opt_tts separately to ensure correct node mapping for opt_seq_enc_raw
        opt_tts = compute_tts_adaptive(
            opt_roots, num_inputs, input_tt=input_tt,
            n_samples=self.tt_num_samples,
            threshold=12,
            seed=self.tt_seed
        )
        # print(f"opt_tts: {opt_tts}")
        # print(f"len(opt_tts): {len(opt_tts)}")

        length = len(tts[0])

        # diff_map[i]==1 iff column i differs between tts and opt_tts
        diff_map = bitarray.bitarray(length)
        diff_map.setall(0)

        # OR-accumulate per-output XOR across rows
        for t, o in zip(tts, opt_tts):
            diff_map |= (t ^ o)

        num_diff_tt = diff_map.count(1)
        error_rate = num_diff_tt / length
        # print(f"Total columns (tt): {length}")
        # print(f"Different columns: {num_diff_tt}")
        # print(f"Error Rate: {error_rate:.4f} ({error_rate * 100:.2f}%)")

        # Generate action masks
        seq_for_masks = seq_enc_raw + [self.eos_id]
        if self.verbose >= 1:
            print(f"[DEBUG load_and_encode] Generating enc_action_masks for seq_enc_raw (len={len(seq_for_masks)})")
        enc_action_masks = self.generate_action_masks(
            tts, input_tt, None, seq_for_masks, True, num_inputs, tts_compressed=tts
        )
        if enc_action_masks is None:
            if self.verbose >= 1:
                print(f"[ERROR load_and_encode] Failed to generate enc_action_masks for {filename}")
            return seq_enc, pos_enc, opt_seq_enc_raw, opt_pos_enc_raw, None, None

        opt_seq_for_masks = opt_seq_enc_raw + [self.eos_id]
        if self.verbose >= 1:
            print(f"[DEBUG load_and_encode] Generating dec_action_masks for opt_seq_enc_raw (len={len(opt_seq_for_masks)})")
        # Use opt_tts for optimized sequence to ensure correct node mapping
        dec_action_masks = self.generate_action_masks(
            opt_tts, input_tt, None, opt_seq_for_masks, True, num_inputs, tts_compressed=opt_tts
        )
        if dec_action_masks is None:
            if self.verbose >= 1:
                print(f"[ERROR load_and_encode] Failed to generate dec_action_masks for {filename}")
            return seq_enc, pos_enc, opt_seq_enc_raw, opt_pos_enc_raw, None, None

        # Post-process optimized sequence (skip_warning=True since we already checked)
        opt_seq_enc, opt_pos_enc = self._encode_postprocess(
            opt_seq_enc_raw, opt_pos_enc_raw, num_inputs
        )

        return seq_enc, pos_enc, opt_seq_enc, opt_pos_enc, enc_action_masks, dec_action_masks

    def load_and_encode_formatted(self, filename):
        """
        Load and encode circuit data, returning formatted dict for training.

        Args:
            filename: Path to JSON file containing circuit data

        Returns:
            Tuple of (inputs_dict, labels) or None if conflicts detected
        """
        seq_enc, pos_enc, opt_seq_enc, opt_pos_enc, enc_action_mask, dec_action_mask = \
            self.load_and_encode(filename)

        if enc_action_mask is None or dec_action_mask is None:
            return None

        inputs = {
            'inputs': seq_enc,
            'enc_pos_encoding': pos_enc,
            'targets': opt_seq_enc,
            'dec_pos_encoding': opt_pos_enc,
            'enc_action_mask': enc_action_mask,
            'dec_action_mask': dec_action_mask
        }
        return inputs, opt_seq_enc

    def _train_with_debug_inf(self, transformer, train_dataset, validation_dataset,
                              validation_steps, optimizer, batch_size, epochs, initial_epoch,
                              ckpt_save_path, latest_ckpt_only, num_train_samples, callbacks):
        """Custom training loop that prints detailed debug info when loss=inf."""
        from scalable_circuit_transformer_refdfs.tensorflow_transformer import masked_loss, masked_accuracy

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        global_step = 0
        steps_per_epoch = max(1, num_train_samples // batch_size)

        for epoch in range(initial_epoch, epochs):
            epoch_loss_sum = 0.0
            epoch_acc_sum = 0.0
            epoch_batches = 0
            print(f"\nEpoch {epoch + 1}/{epochs}")

            for batch_idx, batch in enumerate(train_dataset.take(steps_per_epoch)):
                inputs_dict, labels = batch
                targets = inputs_dict['targets']
                dec_action_mask = inputs_dict['dec_action_mask']

                with tf.GradientTape() as tape:
                    logits = transformer(inputs_dict, training=True)
                    loss_per_pos = loss_fn(labels, logits)
                    mask = tf.cast(labels != 0, loss_per_pos.dtype)
                    loss_sum = tf.reduce_sum(loss_per_pos * mask)
                    mask_sum = tf.reduce_sum(mask)
                    loss = tf.cond(
                        mask_sum > 0,
                        lambda: loss_sum / mask_sum,
                        lambda: tf.constant(0.0, dtype=loss_per_pos.dtype)
                    )

                loss_val = float(loss.numpy())
                if not np.isfinite(loss_val):
                    # === DEBUG: Print which sample/position caused inf ===
                    print(f"\n[DEBUG INF] Epoch {epoch+1}, batch {batch_idx}, global_step {global_step}")
                    print(f"  Total loss = {loss_val}")
                    for b in range(logits.shape[0]):
                        for t in range(logits.shape[1]):
                            if int(labels[b, t].numpy()) == 0:
                                continue
                            target = int(labels[b, t].numpy())
                            lp = float(loss_per_pos[b, t].numpy())
                            mask_ok = bool(dec_action_mask[b, t, target].numpy())
                            logit_t = float(logits[b, t, target].numpy())
                            if not np.isfinite(lp):
                                print(f"  [INF] sample={b}, pos={t}: target={target}, "
                                      f"dec_action_mask[target]={mask_ok}, logit[target]={logit_t:.2e}")
                                allowed = np.where(dec_action_mask[b, t].numpy())[0]
                                print(f"        allowed tokens (first 30): {allowed[:30].tolist()}")
                    print()
                    # Skip gradient update to avoid corrupting model
                    continue

                grads = tape.gradient(loss, transformer.trainable_variables)
                optimizer.apply_gradients(zip(grads, transformer.trainable_variables))

                # Accuracy
                pred = tf.argmax(logits, axis=2)
                match = (tf.cast(labels, pred.dtype) == pred) & (labels != 0)
                acc = tf.reduce_sum(tf.cast(match, tf.float32)) / tf.maximum(mask_sum, 1.0)

                epoch_loss_sum += loss_val
                epoch_acc_sum += float(acc.numpy())
                epoch_batches += 1
                global_step += 1

                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    print(f"  {batch_idx + 1}/{steps_per_epoch} - loss: {loss_val:.4f} - "
                          f"masked_accuracy: {float(acc.numpy()):.4f}")

            if epoch_batches > 0:
                avg_loss = epoch_loss_sum / epoch_batches
                avg_acc = epoch_acc_sum / epoch_batches
                print(f"  Epoch {epoch + 1} - avg loss: {avg_loss:.4f} - avg accuracy: {avg_acc:.4f}")

            # Validation
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_batches = 0
            for v_batch in validation_dataset.take(validation_steps):
                v_inputs_dict, v_labels = v_batch
                v_logits = transformer(v_inputs_dict, training=False)
                v_loss = masked_loss(v_labels, v_logits)
                v_acc = masked_accuracy(v_labels, v_logits)
                if np.isfinite(float(v_loss.numpy())):
                    val_loss_sum += float(v_loss.numpy())
                    val_acc_sum += float(v_acc.numpy())
                    val_batches += 1
            if val_batches > 0:
                print(f"  val_loss: {val_loss_sum/val_batches:.4f} - val_accuracy: {val_acc_sum/val_batches:.4f}")

            # Checkpoint
            if ckpt_save_path is not None:
                path = ckpt_save_path + f"model-{epoch + 1:04d}"
                transformer.save_weights(path)
                print(f"  Saved {path}")

        # Run callbacks (e.g. LogCallback) - simplified, main logic done
        for cb in callbacks:
            if hasattr(cb, 'on_train_end'):
                cb.on_train_end(logs={})

    def train(self,
              train_data_dir,
              ckpt_save_path=None,
              validation_split=0.1,
              epochs=1,
              initial_epoch=0,
              batch_size=4,
              profile=True,
              distributed=False,
              latest_ckpt_only=False,
              log_dir='tensorboard',
              excluded_files: list = None,
              freeze_layers=False,
              learning_rate=0.001,
              debug_inf=False,
              strict_ss_finetune=False,
              strict_ss_epochs=1,
              strict_ss_batch_size=4,
              strict_ss_illegal_ratio_threshold=0.3,
              strict_ss_temperature=0.0,
              strict_ss_top_k=0,
              strict_ss_lr_scale=0.2):
        """
        Train the model on circuit optimization data.

        Args:
            train_data_dir: Directory containing training JSON files
            ckpt_save_path: Path to save checkpoints
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            initial_epoch: Starting epoch (for resuming training)
            batch_size: Training batch size
            profile: Enable TensorBoard profiling
            distributed: Use distributed training
            latest_ckpt_only: Only save latest checkpoint
            log_dir: TensorBoard log directory
            excluded_files: List of files to exclude from training
            freeze_layers: Freeze encoder layers
            learning_rate: Learning rate for optimizer
            debug_inf: When True, use custom training loop that prints detailed debug info
                       when loss=inf (which sample/position, target, dec_action_mask, etc.)
            strict_ss_finetune: Run short strict state-consistent SS fine-tuning after normal training.
            strict_ss_epochs: Number of strict SS fine-tune epochs.
            strict_ss_batch_size: Batch size for strict SS fine-tune generation/training.
            strict_ss_illegal_ratio_threshold: Drop strict SS update when illegal supervision ratio is above threshold.
            strict_ss_temperature: Sampling temperature for strict rollout (0 means greedy).
            strict_ss_top_k: Top-k sampling for strict rollout (0 means disabled).
            strict_ss_lr_scale: Fine-tune learning-rate scale relative to learning_rate.
        """
        import copy
        import tracemalloc

        train_data_dir = train_data_dir + ("/" if train_data_dir[-1] != "/" else "")

        if ckpt_save_path is None:
            print("WARNING: ckpt_save_path is not specified, the trained model will not be saved during training!")
        else:
            ckpt_save_path = ckpt_save_path + ("/" if ckpt_save_path[-1] != "/" else "")
            if not os.path.exists(ckpt_save_path):
                os.makedirs(ckpt_save_path)

        # Load training files
        train_files = os.listdir(train_data_dir)
        print(f"{len(train_files)} training files listed")
        train_files.sort()
        np.random.seed(0)
        np.random.shuffle(train_files)
        print("Training files shuffled")

        # Disable cache during training
        self._transformer.return_cache = False

        # Filter excluded files
        if excluded_files is not None:
            excluded_files = set(excluded_files)
            train_files = [f for f in train_files if f not in excluded_files]
            print(f"Training files filtered: {len(train_files)} files")

        train_files = [(train_data_dir + file) for file in train_files]

        # Create copy for multiprocessing
        # Remove non-serializable objects (TensorFlow model and bound methods)
        self_copied = copy.copy(self)
        self_copied._transformer = None
        self_copied._transformer_inference = None
        # Remove inference methods bound via types.MethodType (cannot be pickled)
        self_copied._batch_estimate_policy = None
        self_copied._batch_estimate_v_value_via_simulation_kvcache = None
        self_copied.optimize = None
        self_copied.optimize_batch = None

        # Create multiprocessing dataset
        mp_dataset = MPDataset(
            train_files,
            self_copied.load_and_encode_formatted,
            validation_split=validation_split,
            num_processes=16
        )

        # Define output signature
        output_signature = (
            {
                'inputs': tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
                'enc_pos_encoding': tf.TensorSpec(
                    shape=(self.max_seq_length, self.max_tree_depth * 2),
                    dtype=tf.float32
                ),
                'targets': tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
                'dec_pos_encoding': tf.TensorSpec(
                    shape=(self.max_seq_length, self.max_tree_depth * 2),
                    dtype=tf.float32
                ),
                'enc_action_mask': tf.TensorSpec(
                    shape=(self.max_seq_length, self.max_vocab_size),
                    dtype=tf.bool
                ),
                'dec_action_mask': tf.TensorSpec(
                    shape=(self.max_seq_length, self.max_vocab_size),
                    dtype=tf.bool
                )
            },
            tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32)
        )

        print("Creating TensorFlow datasets...")
        train_dataset = tf.data.Dataset.from_generator(
            mp_dataset.train_generator,
            output_signature=output_signature
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        validation_dataset = tf.data.Dataset.from_generator(
            mp_dataset.validation_generator,
            output_signature=output_signature
        ).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Calculate validation steps for infinite dataset
        # Use len(validation_data) as approximation (some may be filtered during mapping)
        validation_steps = max(1, int(len(mp_dataset.validation_data) / batch_size))
        print(f"Datasets created. Validation steps per epoch: {validation_steps}")

        # Setup profiling
        if profile:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            summary_writer = tf.summary.create_file_writer(log_dir)

        # Import loss and accuracy functions
        from scalable_circuit_transformer_refdfs.tensorflow_transformer import masked_loss, masked_accuracy

        # Compile model
        if distributed:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                transformer = self._transformer
                if self.ckpt_path is not None:
                    transformer.load_weights(self.ckpt_path)
                optimizer = keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    beta_1=0.9,
                    beta_2=0.98,
                    epsilon=1e-9
                )
                transformer.compile(
                    optimizer=optimizer,
                    loss=masked_loss,
                    metrics=[masked_accuracy]
                )
        else:
            transformer = self._transformer
            optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-9
            )
            transformer.compile(
                optimizer=optimizer,
                loss=masked_loss,
                metrics=[masked_accuracy]
            )

        # Setup callbacks
        class LogCallback(keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                if profile and logs is not None:
                    with summary_writer.as_default():
                        tf.summary.scalar('loss', logs['loss'], step=batch)
                        tf.summary.scalar('masked_accuracy', logs['masked_accuracy'], step=batch)

        callbacks = []
        if profile:
            callbacks.append(LogCallback())

        if ckpt_save_path is not None:
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=ckpt_save_path + 'model-{epoch:04d}',
                save_weights_only=True,
                save_freq='epoch' if not latest_ckpt_only else (
                    len(mp_dataset) * (epochs - initial_epoch) // batch_size
                )
            )
            callbacks.append(checkpoint)

        # Train model
        print("Starting training...")
        if debug_inf:
            self._train_with_debug_inf(
                transformer, train_dataset, validation_dataset,
                validation_steps, optimizer, batch_size, epochs, initial_epoch,
                ckpt_save_path, latest_ckpt_only, len(mp_dataset), callbacks
            )
        else:
            transformer.fit(
                train_dataset,
                initial_epoch=initial_epoch,
                epochs=epochs,
                validation_data=validation_dataset,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )

        mp_dataset.process.terminate()
        print("Training finished")

        if profile:
            print("Profiling completed. Check log directory for TensorBoard logs.")

        # Optional bridge: short strict state-consistent SS fine-tuning
        if strict_ss_finetune and strict_ss_epochs > 0:
            print(
                f"Starting strict SS fine-tune: epochs={strict_ss_epochs}, "
                f"batch_size={strict_ss_batch_size}, lr_scale={strict_ss_lr_scale}, "
                f"illegal_threshold={strict_ss_illegal_ratio_threshold}"
            )
            try:
                prepare_batch_with_generation = __import__(
                    "scalable_circuit_transformer_refdfs.prepare_self_correction_data",
                    fromlist=["prepare_batch_with_generation"],
                ).prepare_batch_with_generation
            except ImportError:
                print("strict_ss_finetune skipped: prepare_self_correction_data is not available in minimal package.")
                return

            ss_optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate * strict_ss_lr_scale,
                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-9
            )
            np.random.seed(42)

            def _compute_supervision_mask(y_true, dec_action_mask):
                non_pad = tf.not_equal(y_true, 0)
                label_idx = tf.cast(y_true, tf.int32)
                gt_valid = tf.gather(dec_action_mask, label_idx, batch_dims=2, axis=2)
                gt_valid = tf.cast(gt_valid, tf.bool)
                return tf.logical_and(non_pad, gt_valid)

            for ss_epoch in range(strict_ss_epochs):
                np.random.shuffle(train_files)
                ss_losses = []
                ss_accs = []
                ss_illegal = []
                dropped_updates = 0
                step = 0

                for start in range(0, len(train_files), strict_ss_batch_size):
                    batch_paths = train_files[start:start + strict_ss_batch_size]
                    batch_samples = prepare_batch_with_generation(
                        self,
                        batch_paths,
                        temperature=strict_ss_temperature,
                        top_k=strict_ss_top_k,
                        seed_base=100000 + ss_epoch * 100000 + start
                    )
                    if not batch_samples:
                        continue

                    inputs_np = {
                        'inputs': np.stack([b['inputs'] for b in batch_samples], axis=0),
                        'enc_pos_encoding': np.stack([b['enc_pos_encoding'] for b in batch_samples], axis=0),
                        'targets': np.stack([b['decoder_input'] for b in batch_samples], axis=0),
                        'dec_pos_encoding': np.stack([b['dec_pos_encoding'] for b in batch_samples], axis=0),
                        'enc_action_mask': np.stack([b['enc_action_mask'] for b in batch_samples], axis=0),
                        'dec_action_mask': np.stack([b['dec_action_mask'] for b in batch_samples], axis=0),
                    }
                    labels_np = np.stack([b['targets'] for b in batch_samples], axis=0)
                    inputs_tf = {k: tf.convert_to_tensor(v) for k, v in inputs_np.items()}
                    labels_tf = tf.convert_to_tensor(labels_np)

                    with tf.GradientTape() as tape:
                        logits = transformer(inputs_tf, training=True)
                        supervision_mask = _compute_supervision_mask(labels_tf, inputs_tf['dec_action_mask'])
                        non_pad = tf.not_equal(labels_tf, 0)
                        illegal_mask = tf.logical_and(non_pad, tf.logical_not(supervision_mask))
                        illegal_ratio = float(
                            tf.reduce_sum(tf.cast(illegal_mask, tf.float32)).numpy() /
                            (tf.reduce_sum(tf.cast(non_pad, tf.float32)).numpy() + 1e-8)
                        )
                        if illegal_ratio > strict_ss_illegal_ratio_threshold:
                            dropped_updates += 1
                            continue

                        loss_object = keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True, reduction='none')
                        loss_per_pos = loss_object(labels_tf, logits)
                        mask_f = tf.cast(supervision_mask, dtype=loss_per_pos.dtype)
                        loss_per_pos *= mask_f
                        loss_per_pos = tf.where(tf.math.is_finite(loss_per_pos), loss_per_pos, tf.zeros_like(loss_per_pos))
                        loss = tf.reduce_sum(loss_per_pos) / (tf.reduce_sum(mask_f) + 1e-8)

                    grads = tape.gradient(loss, transformer.trainable_variables)
                    ss_optimizer.apply_gradients(zip(grads, transformer.trainable_variables))

                    acc = masked_accuracy(labels_tf, logits)
                    ss_losses.append(float(loss))
                    ss_accs.append(float(acc))
                    ss_illegal.append(illegal_ratio)
                    step += 1
                    if step % 20 == 0 or step == 1:
                        print(
                            f"  strict-ss epoch {ss_epoch + 1}/{strict_ss_epochs} step {step} "
                            f"- loss: {float(loss):.4f} - acc: {float(acc):.4f} - illegal_ratio: {illegal_ratio:.4f}"
                        )

                # quick validation snapshot
                val_loss_sum = 0.0
                val_acc_sum = 0.0
                val_batches = 0
                for v_batch in validation_dataset.take(validation_steps):
                    v_inputs_dict, v_labels = v_batch
                    v_logits = transformer(v_inputs_dict, training=False)
                    v_loss = masked_loss(v_labels, v_logits)
                    v_acc = masked_accuracy(v_labels, v_logits)
                    if np.isfinite(float(v_loss.numpy())):
                        val_loss_sum += float(v_loss.numpy())
                        val_acc_sum += float(v_acc.numpy())
                        val_batches += 1

                mean_loss = np.mean(ss_losses) if ss_losses else 0.0
                mean_acc = np.mean(ss_accs) if ss_accs else 0.0
                mean_illegal = np.mean(ss_illegal) if ss_illegal else 0.0
                if val_batches > 0:
                    print(
                        f"strict-ss epoch {ss_epoch + 1}/{strict_ss_epochs} "
                        f"- loss: {mean_loss:.4f} - acc: {mean_acc:.4f} - illegal_ratio: {mean_illegal:.4f} "
                        f"- dropped_updates: {dropped_updates} "
                        f"- val_loss: {val_loss_sum / val_batches:.4f} - val_acc: {val_acc_sum / val_batches:.4f}"
                    )
                else:
                    print(
                        f"strict-ss epoch {ss_epoch + 1}/{strict_ss_epochs} "
                        f"- loss: {mean_loss:.4f} - acc: {mean_acc:.4f} - illegal_ratio: {mean_illegal:.4f} "
                        f"- dropped_updates: {dropped_updates}"
                    )

        # Re-enable cache
        self._transformer.return_cache = True
