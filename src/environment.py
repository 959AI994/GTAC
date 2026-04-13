from __future__ import annotations

import copy
import time
import numpy as np
import npn
import bitarray
import bitarray.util
from scalable_circuit_transformer_refdfs.utils import (Node, NodeWithInv, compute_input_tt, 
compute_tt, check_conflict, get_inputs_rec, detect_circle, compute_critical_path, base_2_log,
check_integrity
)
from scalable_circuit_transformer_refdfs.encoding import (
    stack_to_encoding, int_to_node, deref_node,
    get_pos_encoding_n_vars, _int_to_binary_lsb, _NPN_INT_TO_TT_MAX_VARS,
)
from scalable_circuit_transformer_refdfs.dynamic_encoding import DynamicEncoder
from scalable_circuit_transformer_refdfs.monte_carlo_tt import (
    compute_tt_adaptive, compute_tts_adaptive,
    compute_input_tt_approximate, generate_input_samples,
    MonteCarloTT
)


class ActionMaskTimeoutError(Exception):
    """Exception raised when action mask generation times out."""
    pass

class LogicNetworkEnv:
    def __init__(self,
                 tts,
                 num_inputs,
                 context_num_inputs=None,
                 input_tt=None,
                 init_care_set_tt=None,                 # for the first output (which can be computed in advance) or for all the outputs (list[num_outputs])
                 max_tree_depth=128,
                 max_inference_tree_depth=16,
                 max_inference_reward=None,
                 max_length=None,
                 eos_id=1,
                 pad_id=0,
                 context_hash: set = None,
                 ffw = None,
                 and_always_available=False,            # for training
                 use_controllability_dont_cares=True,   # Patterns that cannot happen at inputs to a network node.
                 tts_compressed=None,                   # must specify when `use_controllability_dont_cares` = False, the truth table of 2^num_inputs corresponding to the "local" aig
                 verbose=0,
                 error_rate_threshold = 0.1,           # for approximate 
                 w_gate = 1,
                 w_delay = 1,
                 w_error = 1,
                 max_inputs=256,
                 max_outputs=256,
                 max_seq_length=2000,
                 use_monte_carlo_tt=False,  # Use Monte Carlo approximation for large circuits
                 mc_tt_threshold=12,  # Use MC when num_inputs > threshold
                 mc_tt_n_samples=8192,  # Number of samples for MC approximation
                 mc_tt_seed=None,  # Random seed for Monte Carlo sampling (ensures consistency within same circuit environment)
                 use_memory_dfs=False  # Use memory DFS encoding (for REF_TOKEN support)
                 ):
        # assert len(tts) == 2
        self.num_outputs = len(tts)
        self.num_inputs = num_inputs
        self.context_num_inputs = context_num_inputs if context_num_inputs is not None else num_inputs
        self.tts_bitarray = tts
        self.max_outputs = max_outputs
        self.max_inputs = max_inputs
        # Initialize care_set_tt to match truth table lengths
        # For Monte Carlo: use sampling length (mc_tt_n_samples)
        # For exact: use full length (2^context_num_inputs)
        if init_care_set_tt is not None:
            self.init_care_set_tt = init_care_set_tt
        else:
            # Determine length based on whether using Monte Carlo
            if use_monte_carlo_tt and self.context_num_inputs > mc_tt_threshold:
                # Monte Carlo: use sampling length
                tt_length = mc_tt_n_samples
            else:
                # Exact: use full length
                tt_length = 2 ** self.context_num_inputs
            self.init_care_set_tt = bitarray.util.ones(tt_length)
        self.ffw = ffw
        self.roots = []
        self.tokens = []
        self.positional_encodings = []
        self.action_masks = []
        self.is_finished = False
        self.gen_eos = False
        self.tree_stack = []
        # self.tt_hash = {}
        # self.tt_cache = {}
        self.context_hash = context_hash
        self.t = 0
        self.max_length = max_length
        self.rewards = []
        self.EOS = eos_id
        self.PAD = pad_id
        self.max_tree_depth = max_tree_depth        # for positional encoding
        self.max_inference_tree_depth = max_inference_tree_depth    # for pruning failed circuits
        self.max_inference_reward = max_inference_reward
        self.and_always_available = and_always_available
        self.use_controllability_dont_cares = use_controllability_dont_cares
        self.unfinished_penalty = -50
        self.verbose = verbose
        self.error_rate_threshold = error_rate_threshold
        self.current_outputs_tt = {}        # record computed root tt
        # new added weight parameters
        self.w_gate = w_gate
        self.w_delay = w_delay
        self.w_error = w_error

        self.prev_gate_count = 0
        self.prev_delay = 0
        self.prev_error = 0

        self.use_monte_carlo_tt = use_monte_carlo_tt
        self.mc_tt_threshold = mc_tt_threshold
        self.mc_tt_n_samples = mc_tt_n_samples
        self.mc_tt_seed = mc_tt_seed  # Store seed for consistent Monte Carlo sampling

        self.encoder = DynamicEncoder(max_inputs=max_inputs, max_outputs=max_outputs, max_seq_length=max_seq_length, use_memory_dfs=use_memory_dfs)
        
        # Memory DFS encoding support
        self.use_memory_dfs = use_memory_dfs  # Flag: enables REF_TOKEN-based subtree reuse when True
        self.decoded_nodes = [] if use_memory_dfs else None  # Holds completed subtree copies that REF_TOKEN may reference
        self.decoded_nodes_tt = [] if use_memory_dfs else None  # Stores uncompressed TT per decoded node for later reuse
        self.seq_position_to_decoded_idx = {} if use_memory_dfs else None  # Maps logical sequence positions to entries inside decoded_nodes
        self.current_seq_position = -1  # Tracks next logical sequence slot for newly completed nodes (-1 means not started yet)
        self._current_encoding_seq_pos = -1  # Tracks raw encoding position (counts every emitted token) for alignment with encoder output
        self._expecting_ref_position = False  # True when a REF_TOKEN was emitted and the next token must be a reference index
        self._node_to_and_token_position = {}  # Temporary map: (id(node.parent), inverted) -> actual node index where the AND/NAND token appeared
        self._current_actual_node_pos = -1  # Stores the encoder-supplied “actual node” counter for the token currently being processed
        self._last_ref_pos = 0  # Stores the last reference position
        # Strictness toggles for debugging/consistency: avoid permissive silent fallbacks.
        self.strict_ref_consistency = True

        # Use adaptive input TT computation: exact for small, approximate for large
        if input_tt is None:
            if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                # Use Monte Carlo approximation for large circuits
                # Use same seed to ensure input_tt and all compute_tt_adaptive calls use same samples
                self.input_tt_bitarray = compute_input_tt_approximate(self.context_num_inputs, self.mc_tt_n_samples, seed=self.mc_tt_seed)
            else:
                # Use exact computation for small circuits
                self.input_tt_bitarray = compute_input_tt(self.context_num_inputs)
        else:
            self.input_tt_bitarray = input_tt
        self.tt_cache_bitarray = {Node(i // 2, None, None): v
                                  for i, v in enumerate(self.input_tt_bitarray) if i % 2 == 0}
        self.tt_hash_bitarray = {v.tobytes(): node for node, v in self.tt_cache_bitarray.items()}
        # Use fixed vocab_size: includes REF_TOKEN if using memory DFS
        self.vocab_size = self.encoder.get_vocab_size(max_inputs)  # Fixed position layout, includes REF_TOKEN if memory DFS
        self.ref_dict = {k: 1 for k in self.tt_cache_bitarray.keys()}
        self.context_nodes = set()
        self.context_records = dict()

        if self.use_controllability_dont_cares:
            self.initialize_care_set_tt()
        else:
            assert tts_compressed is not None
            assert init_care_set_tt is None
            self.compress_indices = None
            # Use the provided input_tt instead of recomputing to maintain size consistency
            # This is critical when using sampled truth tables for large circuits
            self.input_tt_bitarray_compressed = input_tt if input_tt is not None else compute_input_tt(len(self.input_tt_bitarray) // 2)
            
            # Add constant 0/1 nodes to input_tt_bitarray_compressed if not already present
            # Constants should be the last 2 elements
            tt_size = len(self.input_tt_bitarray_compressed[0]) if len(self.input_tt_bitarray_compressed) > 0 else len(tts_compressed[0])
            num_inputs = len(self.input_tt_bitarray) // 2
            expected_length = num_inputs * 2 + 2  # num_inputs * 2 + 2 (includes constants)
            if self.verbose >= 3:
                print(f"[DEBUG env init] input_tt_bitarray_compressed length={len(self.input_tt_bitarray_compressed)}, "
                      f"expected_length={expected_length} (num_inputs={num_inputs} * 2 + 2)")
            if len(self.input_tt_bitarray_compressed) < expected_length:
                # Constants are missing, add them
                constant0 = bitarray.bitarray(tt_size)
                constant0.setall(0)
                constant1 = bitarray.bitarray(tt_size)
                constant1.setall(1)
                self.input_tt_bitarray_compressed.extend([constant0, constant1])
                if self.verbose >= 3:
                    print(f"[DEBUG env init] Added constants to input_tt_bitarray_compressed, new length={len(self.input_tt_bitarray_compressed)}")
            elif self.verbose >= 3:
                print(f"[DEBUG env init] Constants already present in input_tt_bitarray_compressed")
            
            self.tts_bitarray_compressed = tts_compressed
            self.tt_cache_bitarray_compressed = {Node(i // 2, None, None): v
                                                 for i, v in enumerate(self.input_tt_bitarray_compressed) if i % 2 == 0}

            # Add constant 0/1 nodes to the compressed cache with correct size
            constant0 = bitarray.bitarray(tt_size)
            constant0.setall(0)
            constant1 = bitarray.bitarray(tt_size)
            constant1.setall(1)

            const0_node = NodeWithInv(Node(-1, None, None), inverted=False)
            const1_node = NodeWithInv(Node(-1, None, None), inverted=True)
            self.tt_cache_bitarray_compressed[const0_node] = constant0
            self.tt_cache_bitarray_compressed[const1_node] = constant1
            
            # Debug: Print truth table sizes (only when verbose >= 3)
            if self.verbose >= 3:
                if len(self.input_tt_bitarray_compressed) > 0:
                    print(f"[DEBUG env init] input_tt_bitarray_compressed: count={len(self.input_tt_bitarray_compressed)}, "
                          f"first length={len(self.input_tt_bitarray_compressed[0])}")
                if len(self.tts_bitarray_compressed) > 0:
                    print(f"[DEBUG env init] tts_bitarray_compressed: count={len(self.tts_bitarray_compressed)}, "
                          f"first length={len(self.tts_bitarray_compressed[0])}")
                print(f"[DEBUG env init] tt_size={tt_size}, vocab_size={self.vocab_size}")

        # Generate initial action mask
        if self.verbose >= 3:
            print(f"[DEBUG env init] Generating initial action mask, vocab_size={self.vocab_size}")
        try:
            initial_mask = self.gen_action_mask()
            self.action_masks.append(initial_mask)
            if self.verbose >= 3:
                print(f"[DEBUG env init] Initial action mask generated successfully, length={len(initial_mask)}")
        except Exception as e:
            if self.verbose >= 3:
                print(f"[DEBUG env init] ERROR generating initial action mask: {e}")
                import traceback
                traceback.print_exc()
            # Create a default mask using encoder's valid token mask
            valid_mask = self.encoder.get_valid_token_mask(self.num_inputs)
            default_mask = np.zeros(self.vocab_size, dtype=bool)
            default_mask[:len(valid_mask)] = valid_mask
            self.action_masks.append(default_mask)

    @property
    def cur_root_id(self):
        '''
        Return: corresponding output id
        To do: support more outputs (now only support 2-output circuit)
        '''
        return len(self.roots) - (1 if len(self.tree_stack) > 0 else 0)

    @property
    def cumulative_reward(self):
        return sum(self.rewards)

    @property
    def min_cumulative_reward(self):
        res = np.iinfo(int).max
        cumulative_reward = 0
        for r in self.rewards:
            cumulative_reward += r
            res = min(res, cumulative_reward)
        return res

    @property
    def success(self):
        return self.gen_eos

    def _compute_current_error_rate(self) -> float:
        """
        Compute output-level error rate against target truth tables.
        Uses currently completed outputs in `current_outputs_tt`.
        """
        if not self.current_outputs_tt:
            return float(self.prev_error)

        total_bits = 0
        total_diff = 0
        for root_id, pred_tt in self.current_outputs_tt.items():
            if root_id >= len(self.tts_bitarray):
                continue
            tgt_tt = self.tts_bitarray[root_id]
            if pred_tt is None or tgt_tt is None:
                continue
            # Keep robust when lengths mismatch (e.g., compressed/full mismatch in edge cases).
            L = min(len(pred_tt), len(tgt_tt))
            if L <= 0:
                continue
            diff = (pred_tt[:L] ^ tgt_tt[:L]).count(1)
            total_diff += int(diff)
            total_bits += int(L)

        if total_bits <= 0:
            return float(self.prev_error)
        return float(total_diff) / float(total_bits)

    def _compose_normalized_reward(self, gate_reward_raw: float, current_delay: float) -> float:
        """
        Compose reward from normalized gate/delay/error components so different
        metrics stay on comparable scales.
        """
        # 1) Gate component (bounded)
        gate_scale = 10.0
        gate_norm = float(np.tanh(float(gate_reward_raw) / gate_scale))

        # 2) Delay improvement component (relative delta, clipped to [-1, 1])
        delay_base = max(1.0, float(max(self.prev_delay, current_delay)))
        delay_delta = float(self.prev_delay) - float(current_delay)
        delay_norm = float(np.clip(delay_delta / delay_base, -1.0, 1.0))

        # 3) Error-rate improvement component (relative delta, clipped to [-1, 1])
        current_error = self._compute_current_error_rate()
        error_delta = float(self.prev_error) - float(current_error)  # improvement -> positive
        error_base = max(float(self.prev_error), float(current_error), float(self.error_rate_threshold), 1e-3)
        error_norm = float(np.clip(error_delta / error_base, -1.0, 1.0))

        reward = (
            float(self.w_gate) * gate_norm +
            float(self.w_delay) * delay_norm +
            float(self.w_error) * error_norm
        )

        # Update running baselines for next step.
        self.prev_delay = float(current_delay)
        self.prev_error = float(current_error)
        return reward

    def reset(self, **kwargs):
        """重置环境到初始状态"""
        # 初始化状态变量
        self.roots = []
        self.tokens = []
        self.positional_encodings = []
        self.action_masks = []
        self.tree_stack = []
        self.is_finished = False
        self.gen_eos = False
        self.t = 0
        self.rewards = []
        self.prev_delay = 0
        self.prev_error = 0
        # self.cur_root_id = 0
        
        # 重置 REF_TOKEN 相关状态
        if self.use_memory_dfs:
            self.decoded_nodes = []
            self.decoded_nodes_tt = []
            self.seq_position_to_decoded_idx = {}
            self._node_to_and_token_position = {}
            self.current_seq_position = -1
            self._current_encoding_seq_pos = -1
            self._current_actual_node_pos = -1
            self._expecting_ref_position = False
        
        # 初始化真值表缓存
        self.tt_cache_bitarray = {Node(i // 2, None, None): v
                                 for i, v in enumerate(self.input_tt_bitarray) if i % 2 == 0}
        self.tt_hash_bitarray = {v.tobytes(): node for node, v in self.tt_cache_bitarray.items()}
        self.ref_dict = {k: 1 for k in self.tt_cache_bitarray.keys()}
        self.context_nodes = set()
        self.context_records = dict()
        
        # 初始化care set（如果使用）
        if self.use_controllability_dont_cares:
            self.initialize_care_set_tt()
        
        # 生成初始动作掩码
        self.action_masks.append(self.gen_action_mask())
        
        return self._get_obs()

    def _get_obs(self):
        """获取当前观察值"""
        # 填充序列到最大长度
        tokens = np.array(self.tokens + [self.PAD] * (self.max_length - len(self.tokens)), dtype=np.int32)
        pos_enc = np.zeros((self.max_length, self.max_tree_depth * 2), dtype=np.float32)
        if self.positional_encodings:
            pos_enc[:len(self.positional_encodings)] = np.array(self.positional_encodings)
        
        # 当前动作掩码
        action_mask = self.action_masks[-1] if self.action_masks else np.zeros(self.vocab_size, dtype=bool)
        return {
            'tokens': tokens,
            'positional_encodings': pos_enc,
            'action_mask': action_mask
        }

    def _get_node_key(self, node: NodeWithInv | None):
        """
        Return a stable identifier for AND/NAND nodes that matches the encoder's node_id logic:
        (id(node.parent), node.inverted).
        """
        if node is None or node.parent is None:
            return None
        return (id(node.parent), bool(getattr(node, 'inverted', False)))

    def initialize_care_set_tt(self):       # both controllability and observability don't cares
        self._compress_indices_full_minterms = False
        if self.cur_root_id == 0:
            self.care_set_tt = self.init_care_set_tt[self.cur_root_id] if isinstance(self.init_care_set_tt, list) else self.init_care_set_tt
        else:
            if self.ffw is not None:
                new_inputs = get_inputs_rec(self.roots)
                modified_list = []
                for extracted_input, orig_node in self.ffw.input_mapping.items():
                    if extracted_input.var in new_inputs:
                        for new_input_with_inv in new_inputs[extracted_input.var]:
                            modified_list.append((new_input_with_inv, new_input_with_inv.parent))
                            new_input_with_inv.parent = orig_node
                for new_output, output in zip(self.roots, self.ffw.outputs):
                    for node_with_inv in self.ffw.parent.fanout_dict[output].keys():
                        node_with_inv.parent = new_output.parent
                        if new_output.inverted:
                            node_with_inv.inverted = not node_with_inv.inverted
                if not detect_circle(self.ffw.parent.outputs):
                    self.care_set_tt = self.ffw.parent.compute_care_set(self.ffw.outputs[self.cur_root_id])
                else:
                    self.care_set_tt = bitarray.util.ones(2 ** self.context_num_inputs)
                for new_output, output in zip(self.roots, self.ffw.outputs):
                    for node_with_inv in self.ffw.parent.fanout_dict[output].keys():
                        node_with_inv.parent = output
                        if new_output.inverted:
                            node_with_inv.inverted = not node_with_inv.inverted
                for new_input_with_inv, parent in modified_list:
                    new_input_with_inv.parent = parent
            elif isinstance(self.init_care_set_tt, list):
                self.care_set_tt = self.init_care_set_tt[self.cur_root_id]

        a = bytearray()
        len_care_set = self.care_set_tt.count()
        num_inputs = len(self.input_tt_bitarray) // 2

        # Support large number of inputs (up to 256)
        # For inputs > 8, skip single-byte unpack (unusable), but still apply care-set row sampling
        # so compressed TT length matches external / ABC pattern-level error rate (check_conflict denominator).
        if num_inputs > 8:
            one_pat = bitarray.bitarray("1")
            care_indices = list(self.care_set_tt.search(one_pat))
            if len(care_indices) == 0:
                # No care bits: fall back to full truth tables (same as legacy path)
                self.compress_indices = []
                self.input_tt_bitarray_compressed = (
                    self.input_tt_bitarray.copy()
                    if isinstance(self.input_tt_bitarray, list)
                    else list(self.input_tt_bitarray)
                )
                tt_size = (
                    len(self.input_tt_bitarray_compressed[0])
                    if len(self.input_tt_bitarray_compressed) > 0
                    else len(self.tts_bitarray[0])
                    if len(self.tts_bitarray) > 0
                    else 2**num_inputs
                )
                expected_length = num_inputs * 2 + 2
                if len(self.input_tt_bitarray_compressed) < expected_length:
                    constant0 = bitarray.bitarray(tt_size)
                    constant0.setall(0)
                    constant1 = bitarray.bitarray(tt_size)
                    constant1.setall(1)
                    self.input_tt_bitarray_compressed.extend([constant0, constant1])
                    if self.verbose >= 3:
                        print(
                            f"[DEBUG env init] Added constants to input_tt_bitarray_compressed "
                            f"(num_inputs > 8, empty care), new length={len(self.input_tt_bitarray_compressed)}"
                        )
                self.tt_cache_bitarray_compressed = self.tt_cache_bitarray.copy()
                self.tts_bitarray_compressed = self.tts_bitarray
                return

            self._compress_indices_full_minterms = True
            self.compress_indices = care_indices
            k = len(care_indices)
            self.input_tt_bitarray_compressed = [
                bitarray.bitarray([tt[i] for i in care_indices]) for tt in self.input_tt_bitarray
            ]
            constant0 = bitarray.bitarray(k)
            constant0.setall(0)
            constant1 = bitarray.bitarray(k)
            constant1.setall(1)
            self.input_tt_bitarray_compressed.extend([constant0, constant1])

            self.tts_bitarray_compressed = [
                bitarray.bitarray([tt[i] for i in care_indices]) for tt in self.tts_bitarray
            ]
            self.tt_cache_bitarray_compressed = {
                node: bitarray.bitarray([v[i] for i in care_indices])
                for node, v in self.tt_cache_bitarray.items()
            }
            const0_node = NodeWithInv(Node(-1, None, None), inverted=False)
            const1_node = NodeWithInv(Node(-1, None, None), inverted=True)
            self.tt_cache_bitarray_compressed[const0_node] = constant0
            self.tt_cache_bitarray_compressed[const1_node] = constant1
            if self.verbose >= 3:
                print(
                    f"[DEBUG env init] num_inputs>8 care sampling: k={k} compressed columns "
                    f"(full minterms={2**num_inputs})"
                )
            return

        # For num_inputs <= 8, use single-byte encoding
        bytes_per_input = 1

        for i, tt in enumerate(self.input_tt_bitarray):
            if i % 2 == 0:
                input_idx = i // 2
                # Only process inputs that fit in single byte (< 8)
                if input_idx < 8:
                    a.extend(tt[self.care_set_tt].unpack(one=(1 << input_idx).to_bytes(1, 'big')))

        if len(a) > 0:
            a_np = np.frombuffer(a, dtype=np.uint8).reshape(min(num_inputs, 8), len_care_set)
            a_np = np.sum(a_np, axis=0, dtype=np.uint8)
            a_np_unique, self.compress_indices = np.unique(a_np, return_index=True)
        else:
            # Fallback for edge cases
            self.compress_indices = np.arange(min(len_care_set, 2 ** self.context_num_inputs), dtype=np.int64)

        if self.verbose > 1:
            a = bytearray()
            for i, tt in enumerate(self.input_tt_bitarray):
                if i % 2 == 0:
                    a.extend(tt.unpack(one=(1 << (i // 2)).to_bytes(1, 'big')))
            a_np_ = np.frombuffer(a, dtype=np.uint8).reshape(len(self.input_tt_bitarray) // 2, len(self.care_set_tt))
            a_np_ = np.sum(a_np_, axis=0, dtype=np.uint8)
            a_np_unique_, self.compress_indices_no_care_set = np.unique(a_np_, return_index=True)
            if len(self.compress_indices_no_care_set) > len(self.compress_indices):
                print("care set size: %d, without care set: %d, with care set: %d" %
                      (self.care_set_tt.count(), len(self.compress_indices_no_care_set), len(self.compress_indices)))

        self.compress_indices = list(self.compress_indices)
        a_bitarray_unique = [bitarray.bitarray() for _ in a_np_unique]
        for a_bitarray_i, a_np_i in zip(a_bitarray_unique, a_np_unique):
            a_bitarray_i.frombytes(a_np_i.tobytes())

        if len(a_bitarray_unique) == 0:
            self.input_tt_bitarray_compressed = [bitarray.bitarray() for _ in self.input_tt_bitarray]
        else:
            self.input_tt_bitarray_compressed = []
            for i, a_tuple in enumerate(zip(*a_bitarray_unique)):
                if i < 8 - len(self.input_tt_bitarray) // 2:
                    continue
                a_bitarray = bitarray.bitarray(a_tuple)
                self.input_tt_bitarray_compressed.extend([~a_bitarray, a_bitarray])
            self.input_tt_bitarray_compressed.reverse()

        constant0 = bitarray.bitarray(len(self.compress_indices))
        constant0.setall(0)  # constant 0
        constant1 = bitarray.bitarray(len(self.compress_indices))
        constant1.setall(1)  # constant 1
        self.input_tt_bitarray_compressed.extend([constant0, constant1])

        # self.input_tt_bitarray_compressed_ = [bitarray.bitarray(_) for _ in zip(*a_bitarray_unique)]
        tts_care_set = [tt[self.care_set_tt] for tt in self.tts_bitarray]
        self.tts_bitarray_compressed = [bitarray.bitarray([tt[i] for i in self.compress_indices]) for tt in tts_care_set]
        self.tt_cache_bitarray_compressed = {Node(i // 2, None, None): v
                                             for i, v in enumerate(self.input_tt_bitarray_compressed) if i % 2 == 0}
        # add constant 0/1
        const0_node = NodeWithInv(Node(-1, None, None), inverted=False)
        const1_node = NodeWithInv(Node(-1, None, None), inverted=True)

        self.tt_cache_bitarray_compressed[const0_node] = constant0
        self.tt_cache_bitarray_compressed[const1_node] = constant1

    def compress(self, tt):
        if self.compress_indices is None or len(self.compress_indices) == 0:
            return tt
        if getattr(self, "_compress_indices_full_minterms", False):
            return bitarray.bitarray([tt[i] for i in self.compress_indices])
        return (tt[self.care_set_tt])[self.compress_indices]

    def step(self, token): ###########################################################################
        token = int(token)
        # print(f"token: {token}")
        # REF position token: encoder uses cur_pos_enc+1 for position encoding, NOT token value or tree path.
        # encode_aig_memory_dfs: pos_enc.append(cur_pos_enc); pos_enc.append(cur_pos_enc + 1)
        if self.use_memory_dfs and self._expecting_ref_position:
            next_pos_int = (self._last_ref_pos << 2) + 1
            n_vars = get_pos_encoding_n_vars(self.max_tree_depth, self.max_outputs if self.max_outputs else 256)
            if n_vars <= _NPN_INT_TO_TT_MAX_VARS:
                binary_list = npn.int_to_tt(next_pos_int, n_vars)
            else:
                binary_list = _int_to_binary_lsb(next_pos_int, n_vars)
            final_pos = list(reversed(binary_list))
            pos_array = np.array(final_pos, dtype=np.float32)
            target_len = self.max_tree_depth * 2
            if len(pos_array) > target_len:
                pos_enc = pos_array[:target_len]
            else:
                pos_enc = np.zeros(target_len, dtype=np.float32)
                pos_enc[:len(pos_array)] = pos_array
            self.positional_encodings.append(pos_enc)
        elif self.use_memory_dfs and token == self.encoder.REF_TOKEN:
            # REF_TOKEN uses parent path (cur_pos_enc), not child path
            child_step = None
            if len(self.tree_stack) > 0:
                child_step = 1 if self.tree_stack[-1].left is None else 2

            eff_max = self.max_outputs if self.max_outputs else 256
            k = eff_max.bit_length() + 1
            path_int = (1 << k) + self.cur_root_id + 1
            for i in range(len(self.tree_stack) - 1):
                step_val = 1 if self.tree_stack[i].left is self.tree_stack[i+1] else 2
                path_int = (path_int << 2) + step_val
            if child_step is not None and child_step in (1, 2):
                self._last_ref_pos = (path_int << 2) + child_step
            else:
                self._last_ref_pos = path_int

            self.positional_encodings.append(stack_to_encoding(
                self.tree_stack, self.cur_root_id, self.max_tree_depth,
                max_outputs=self.max_outputs, child_step=child_step
            ))
        else:
            # Record position for regular token: path to child we're adding (matches training timing)
            child_step = None
            if len(self.tree_stack) > 0:
                child_step = 1 if self.tree_stack[-1].left is None else 2
            self.positional_encodings.append(stack_to_encoding(
                self.tree_stack, self.cur_root_id, self.max_tree_depth,
                max_outputs=self.max_outputs, child_step=child_step
            ))
        # print(f"positional_encodings: {self.positional_encodings[-1]}")
        # print(f"[DEBUG] seq_position_to_decoded_idx: {self.seq_position_to_decoded_idx}")
        # print(f"[DEBUG] _node_to_and_token_position: {self._node_to_and_token_position}")
        # print(f"[DEBUG] len(seq_position_to_decoded_idx): {len(self.seq_position_to_decoded_idx)}")
        # print(f"[DEBUG] len(_node_to_and_token_position): {len(self._node_to_and_token_position)}")
        # print(f"[DEBUG] len(decoded_nodes): {len(self.decoded_nodes)}")

        # Handle REF_TOKEN for memory DFS encoding
        if self.use_memory_dfs and self._expecting_ref_position:
            # Previous token was REF_TOKEN, current token is ref_position (sequence position)
            ref_seq_position = token
            
            # Convert sequence position to decoded_nodes index
            # FALLBACK MECHANISM DISABLED
            # print("--------------------------------")
            # print(f"node_to_and_token_position: {self._node_to_and_token_position}")
            # print(f"seq_position_to_decoded_idx: {self.seq_position_to_decoded_idx}")
            # print(f"ref_seq_position: {ref_seq_position}")
            # print("--------------------------------")
            if self.seq_position_to_decoded_idx is None or ref_seq_position not in self.seq_position_to_decoded_idx:
                # Invalid ref_position: provide clear error message
                error_msg = (f"Invalid ref_seq_position {ref_seq_position}: "
                           f"not found in seq_position_to_decoded_idx. "
                           f"self.seq_position_to_decoded_idx: {self.seq_position_to_decoded_idx}"
                           f"Available positions: {sorted(self.seq_position_to_decoded_idx.keys()) if self.seq_position_to_decoded_idx else []}")
                if self.verbose >= 1:
                    print(f"[ERROR env.step] {error_msg}")
                # Reset state and return error
                self._expecting_ref_position = False
                self.is_finished = True  # Mark environment as finished to stop inference loop
                reward, done = self.unfinished_penalty, True
                self.tokens.append(token)
                self.t += 1
                self.action_masks.append(self.gen_action_mask())
                return reward, done
            
            # Get decoded_nodes index from sequence position
            decoded_idx = self.seq_position_to_decoded_idx[ref_seq_position]
                        
            # Validate decoded_idx (for both mapped and fallback cases)
            if decoded_idx < 0 or decoded_idx >= len(self.decoded_nodes):
                error_msg = (f"Invalid decoded_idx {decoded_idx} for seq_position {ref_seq_position}: "
                           f"decoded_nodes length: {len(self.decoded_nodes)}")
                if self.verbose >= 3:
                    print(f"[ERROR env.step] {error_msg}")
                self._expecting_ref_position = False
                self.is_finished = True  # Mark environment as finished to stop inference loop
                reward, done = self.unfinished_penalty, True
                self.tokens.append(token)
                self.t += 1
                self.action_masks.append(self.gen_action_mask())
                return reward, done

            # *** START FIX ***
            # 映射 *当前* actual_node_pos (对应于此 REF_TOKEN)
            # 到它所 *引用* 的节点的 decoded_idx。
            if hasattr(self, '_current_actual_node_pos') and self._current_actual_node_pos >= 0:
                if self._current_actual_node_pos not in self.seq_position_to_decoded_idx:
                    self.seq_position_to_decoded_idx[self._current_actual_node_pos] = decoded_idx
                    if self.verbose >= 1:
                        print(f"[DEBUG env.step] Mapped REF_TOKEN position {self._current_actual_node_pos} -> "
                              f"referenced decoded_idx {decoded_idx} (from ref_seq_position {ref_seq_position})")
            
            # *** END FIX ***
            # Get the referenced node (already decoded, complete subtree)
            # REF_TOKEN always references a complete subtree that was stored in decoded_nodes
            # A node is only stored in decoded_nodes when it's complete (popped from tree_stack)
            # So the referenced node is guaranteed to be complete (left and right are both set, or it's a leaf)
            referenced_node = self.decoded_nodes[decoded_idx]
            
            # Debug: Print referenced node state
            if self.verbose >= 3:
                is_leaf = referenced_node.is_leaf()
                left_set = referenced_node.left is not None
                right_set = referenced_node.right is not None
                print(f"[DEBUG env.step] REF_TOKEN references decoded_idx {decoded_idx}:")
                print(f"[DEBUG env.step]   is_leaf={is_leaf}, left_set={left_set}, right_set={right_set}")
                if not is_leaf:
                    left_is_leaf = referenced_node.left.is_leaf() if left_set else None
                    right_is_leaf = referenced_node.right.is_leaf() if right_set else None
                    print(f"[DEBUG env.step]   left_is_leaf={left_is_leaf}, right_is_leaf={right_is_leaf}")
            
            # Verify that the referenced node is complete
            if not referenced_node.is_leaf() and (referenced_node.left is None or referenced_node.right is None):
                error_msg = (f"Invalid REF_TOKEN reference: node at decoded_idx {decoded_idx} is not complete. "
                           f"left={referenced_node.left is not None}, right={referenced_node.right is not None}")
                if self.verbose >= 0:
                    print(f"[ERROR env.step] {error_msg}")
                self._expecting_ref_position = False
                self.is_finished = True  # Mark environment as finished to stop inference loop
                reward, done = self.unfinished_penalty, True
                self.tokens.append(token)
                self.t += 1
                self.action_masks.append(self.gen_action_mask())
                return reward, done
            
            # Create a copy to avoid modifying the original referenced node
            node = copy.copy(referenced_node) if not referenced_node.is_leaf() else copy.copy(referenced_node)
            self._expecting_ref_position = False
            # REF = reference existing subtree → no new ANDs; reward 0 for beam/MCTS.
            ref_gate_cost_for_reward = 0
            
            # Debug: Print node copy state
            if self.verbose >= 3:
                print(f"[DEBUG env.step] After copying node: is_leaf={node.is_leaf()}, "
                      f"left_set={node.left is not None}, right_set={node.right is not None}")
            
            # REF_TOKEN references a complete subtree, so treat it as a leaf_node for processing
            # This means we can immediately process it without needing to build it further
            if len(self.tree_stack) == 0:
                if len(self.roots) > 0:
                    # Use adaptive TT computation: exact for small, approximate for large
                    if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                        tt_bitarray = compute_tt_adaptive(
                            self.roots[self.cur_root_id-1],
                            num_inputs=self.context_num_inputs,
                            input_tt=self.input_tt_bitarray,
                            threshold=self.mc_tt_threshold,
                            n_samples=self.mc_tt_n_samples,
                            seed=self.mc_tt_seed
                        )
                    else:
                        tt_bitarray = compute_tt(
                            self.roots[self.cur_root_id-1],
                            input_tt=self.input_tt_bitarray
                        )
                    self.current_outputs_tt[self.cur_root_id-1] = tt_bitarray
                self.roots.append(node)
                # Check if all outputs are complete after adding the root
                if len(self.roots) == self.num_outputs and len(self.tree_stack) == 0:
                    # print(f"[DEBUG1 env.step] debug: num outputs = {self.num_outputs}, tree_stack length = {len(self.tree_stack)}")
                    completed_root_id = len(self.roots) - 1
                    if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                        tt_last = compute_tt_adaptive(
                            self.roots[completed_root_id],
                            num_inputs=self.context_num_inputs,
                            input_tt=self.input_tt_bitarray,
                            threshold=self.mc_tt_threshold,
                            n_samples=self.mc_tt_n_samples,
                            seed=self.mc_tt_seed,
                        )
                    else:
                        tt_last = compute_tt(
                            self.roots[completed_root_id],
                            input_tt=self.input_tt_bitarray,
                        )
                    self.current_outputs_tt[completed_root_id] = tt_last
                    self.gen_eos = True  # next token should be EOS
                    done = False
                    self.ref_dict[node.parent] = 1
                    reward = -ref_gate_cost_for_reward
                    self.tokens.append(token)
                    self.t += 1
                    self.action_masks.append(self.gen_action_mask())
                    return reward, done
            else:
                # insert node into the tree
                if self.tree_stack[-1].left is None:
                    self.tree_stack[-1].left = node
                else:
                    self.tree_stack[-1].right = node
            
            self.ref_dict[node.parent] = 1
            # calculate reward (REF = reuse, no additional gates)
            reward = -ref_gate_cost_for_reward
            done = False
            
            # REF_TOKEN references a complete subtree, so treat it as a leaf_node
            # Add it to tree_stack and process it immediately (same as leaf case)
            if self.verbose >= 3:
                print(f"[DEBUG env.step] Before appending REF_TOKEN node to tree_stack:")
                print(f"[DEBUG env.step]   tree_stack length={len(self.tree_stack)}")
                print(f"[DEBUG env.step]   node.is_leaf()={node.is_leaf()}, "
                      f"node.left is not None={node.left is not None}, "
                      f"node.right is not None={node.right is not None}")
            
            self.tree_stack.append(node)
            
            # Debug: Check while loop condition
            if self.verbose >= 3:
                top_node = self.tree_stack[-1]
                is_leaf = top_node.is_leaf()
                left_set = top_node.left is not None
                right_set = top_node.right is not None
                condition1 = is_leaf
                condition2 = left_set and right_set
                condition_met = condition1 or condition2
                print(f"[DEBUG env.step] While loop condition check:")
                print(f"[DEBUG env.step]   tree_stack length={len(self.tree_stack)}")
                print(f"[DEBUG env.step]   top_node.is_leaf()={is_leaf}")
                print(f"[DEBUG env.step]   top_node.left is not None={left_set}")
                print(f"[ env.step]   top_node.right is not None={right_set}")
                print(f"[DEBUG env.step]   Condition: (is_leaf={is_leaf}) or (left_set={left_set} and right_set={right_set}) = {condition_met}")
            
            # Process the complete subtree (same logic as leaf case)
            # Since the node is complete (left and right are both set, or it's a leaf),
            # the while loop condition will be satisfied immediately
            loop_iterations = 0
            max_iterations = 1000  # Safety limit to prevent infinite loops
            while len(self.tree_stack) > 0 and (self.tree_stack[-1].is_leaf() or (
                    self.tree_stack[-1].left is not None and self.tree_stack[-1].right is not None)):
                loop_iterations += 1
                if loop_iterations > max_iterations:
                    error_msg = f"REF_TOKEN processing loop exceeded {max_iterations} iterations. Possible infinite loop."
                    if self.verbose >= 1:
                        print(f"[ERROR env.step] {error_msg}")
                        print(f"[ERROR env.step] tree_stack length={len(self.tree_stack)}")
                        if len(self.tree_stack) > 0:
                            top = self.tree_stack[-1]
                            print(f"[ERROR env.step] top_node.is_leaf()={top.is_leaf()}, "
                                  f"left_set={top.left is not None}, right_set={top.right is not None}")
                    self._expecting_ref_position = False
                    self.is_finished = True  # Mark environment as finished to stop inference loop
                    reward, done = self.unfinished_penalty, True
                    self.tokens.append(token)
                    self.t += 1
                    self.action_masks.append(self.gen_action_mask())
                    return reward, done
                
                if self.verbose >= 3:
                    print(f"[DEBUG env.step] REF_TOKEN processing loop iteration {loop_iterations}:")
                    print(f"[DEBUG env.step]   tree_stack length={len(self.tree_stack)}")
                    if len(self.tree_stack) > 0:
                        top = self.tree_stack[-1]
                        print(f"[DEBUG env.step]   top_node.is_leaf()={top.is_leaf()}, "
                              f"left_set={top.left is not None}, right_set={top.right is not None}")
                
                # Process complete nodes (same logic as leaf case)
                old_node = copy.copy(self.tree_stack[-1])
                old_node.inverted = False
                # Use adaptive TT computation: exact for small, approximate for large
                if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                    tt_bitarray = compute_tt_adaptive(
                        old_node,
                        num_inputs=self.context_num_inputs,
                        input_tt=self.input_tt_bitarray,
                        threshold=self.mc_tt_threshold,
                        n_samples=self.mc_tt_n_samples,
                        seed=self.mc_tt_seed
                    )
                else:
                    tt_bitarray = compute_tt(old_node, input_tt=self.input_tt_bitarray, cache=self.tt_cache_bitarray)
                tt_not_bitarray = ~tt_bitarray
                tt = tt_bitarray.tobytes()
                tt_not = tt_not_bitarray.tobytes()
                self.tt_hash_bitarray[tt_bitarray.tobytes()] = self.tree_stack[-1].parent
                self.tt_cache_bitarray[self.tree_stack[-1]] = tt_not_bitarray if self.tree_stack[-1].inverted else tt_bitarray
                if self.use_controllability_dont_cares:
                    self.tt_cache_bitarray_compressed[self.tree_stack[-1]] = self.compress(self.tt_cache_bitarray[self.tree_stack[-1]])
                else:
                    if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                        tt_bitarray_compressed = compute_tt_adaptive(
                            self.tree_stack[-1],
                            num_inputs=len(self.input_tt_bitarray_compressed[0]) if len(self.input_tt_bitarray_compressed) > 0 else self.context_num_inputs,
                            input_tt=self.input_tt_bitarray_compressed,
                            threshold=self.mc_tt_threshold,
                            n_samples=self.mc_tt_n_samples,
                            seed=self.mc_tt_seed
                        )
                    else:
                        tt_bitarray_compressed = compute_tt(self.tree_stack[-1],
                                                           input_tt=self.input_tt_bitarray_compressed,
                                                           cache=self.tt_cache_bitarray_compressed)
                    self.tt_cache_bitarray_compressed[self.tree_stack[-1]] = tt_bitarray_compressed
                v1 = 0  # Initialize v1 to 0
                if self.context_hash is not None and (tt in self.context_hash or tt_not in self.context_hash):
                    v1 = deref_node(self.tree_stack[-1].parent, self.ref_dict, self.context_nodes)
                    self.context_nodes.add(self.tree_stack[-1].parent)
                    self.context_records[self.tree_stack[-1].parent] = tt
                reward += v1
                # Track decoded node for REF_TOKEN references
                if self.use_memory_dfs:
                    complete_node = self.tree_stack[-1] if len(self.tree_stack) > 0 else None
                    if complete_node is not None and not complete_node.is_leaf():
                        # Create a deep copy to avoid modifying the original
                        complete_node_copy = copy.copy(complete_node)
                        decoded_idx = len(self.decoded_nodes)
                        self.decoded_nodes.append(complete_node_copy)
                        if self.decoded_nodes_tt is not None:
                            node_tt = self._get_or_compute_tt(complete_node)
                            # print(f"node_tt: {node_tt}")
                            self.decoded_nodes_tt.append(node_tt if node_tt is not None else None)
                        
                        # Map AND token position to decoded_idx
                        node_key = self._get_node_key(complete_node)
                        mapped = False
                        if node_key is not None and node_key in self._node_to_and_token_position:
                            and_token_pos = self._node_to_and_token_position[node_key]
                            self.seq_position_to_decoded_idx[and_token_pos] = decoded_idx
                            if self.verbose >= 2:
                                print(f"[DEBUG env.step] Mapped AND token position {and_token_pos} -> decoded_idx {decoded_idx} (node_key={node_key})")
                            del self._node_to_and_token_position[node_key]
                            mapped = True
                        
                        
                        if self.current_seq_position >= 0:
                            self.current_seq_position += 1
                
                self.tree_stack.pop()
                if self.verbose >= 3:
                    print(f"[DEBUG env.step] After pop: tree_stack length={len(self.tree_stack)}")
            
            # Match non-REF path: when REF processing empties the stack, the active root is complete;
            # must record its full-domain TT for error accumulation in gen_action_mask / check_conflict.
            if len(self.tree_stack) == 0 and len(self.roots) > 0:
                completed_root_id = len(self.roots) - 1
                if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                    tt_root = compute_tt_adaptive(
                        self.roots[completed_root_id],
                        num_inputs=self.context_num_inputs,
                        input_tt=self.input_tt_bitarray,
                        threshold=self.mc_tt_threshold,
                        n_samples=self.mc_tt_n_samples,
                        seed=self.mc_tt_seed,
                    )
                else:
                    tt_root = compute_tt(
                        self.roots[completed_root_id],
                        input_tt=self.input_tt_bitarray,
                    )
                self.current_outputs_tt[completed_root_id] = tt_root
            
            if self.verbose >= 3:
                print(f"[DEBUG env.step] REF_TOKEN processing loop finished:")
                print(f"[DEBUG env.step]   loop_iterations={loop_iterations}")
                print(f"[DEBUG env.step]   tree_stack length={len(self.tree_stack)}")
                print(f"[DEBUG env.step]   roots count={len(self.roots)}, num_outputs={self.num_outputs}")
            
            # Check if all outputs are complete after processing
            if len(self.tree_stack) == 0 and len(self.roots) == self.num_outputs:
                # print(f"[DEBUG2 env.step] debug: num outputs = {self.num_outputs}, tree_stack length = {len(self.tree_stack)}")
                self.gen_eos = True  # next token should be EOS
                done = False
                current_delay = compute_critical_path(self.roots)
                reward = self._compose_normalized_reward(reward, current_delay)
                if self.is_finished:
                    if self.gen_eos:
                        reward += 50.0
                    else:
                        completed_outputs = self.cur_root_id
                        reward = (completed_outputs / self.num_outputs) * 10 - 50

                # If done, generate action mask and return early
                self.tokens.append(token)
                self.t += 1
                self.action_masks.append(self.gen_action_mask())
                return reward, done
            
            self.tokens.append(token)
            self.t += 1
            
            # Update sequence position for regular tokens (not REF_TOKEN or ref_position)
            if self.use_memory_dfs and not (token == self.encoder.REF_TOKEN or self._expecting_ref_position):
                # This is a regular token that creates a new node
                # Initialize current_seq_position if needed
                if self.current_seq_position < 0:
                    self.current_seq_position = 0
                # Note: The encoding sequence position is set in generate_action_masks
                # The mapping from encoding seq_pos to decoded_idx is done in generate_action_masks
            
            current_delay = compute_critical_path(self.roots)
            reward = self._compose_normalized_reward(reward, current_delay)
            self.rewards.append(reward)
            if len(self.tree_stack) == 0 and self.cur_root_id < self.num_outputs and self.use_controllability_dont_cares:
                self.initialize_care_set_tt()
            self.action_masks.append(self.gen_action_mask())
            return reward, done
        
        # Handle REF_TOKEN (first token)
        if self.use_memory_dfs and token == self.encoder.REF_TOKEN:
            # REF_TOKEN is followed by a position index
            self._expecting_ref_position = True
            self.tokens.append(token)
            self.t += 1
            # Don't generate action mask yet, wait for ref_position
            # But we need to return something... Actually, we should not return here
            # The caller should call step again with ref_position
            # For now, we'll just set the flag and return
            reward, done = 0, False
            self.action_masks.append(self.gen_action_mask())
            return reward, done
        
        if token == self.EOS:
            self.is_finished = True
            if self.gen_eos:
                reward, done = 0, True
            else:
                reward, done = self.unfinished_penalty, True
        elif self.is_finished:
            assert token == self.PAD
            reward, done = 0, True
        elif not self.is_finished and self.t >= self.max_length - 1:  # reached the last step but still not finished
            self.is_finished = True  # Mark environment as finished to stop inference loop
            reward, done = self.unfinished_penalty, True
        else:
            node = self.encoder.int_to_node(token, self.num_inputs)
            # print(f"token:{token}")
            
            # Track AND token position for memory DFS encoding
            # When an AND/NAND token is processed, record its position
            # This position will be used to map to decoded_idx when the node completes
            # Use _current_actual_node_pos (actual node count) instead of _current_encoding_seq_pos (raw sequence position)
            # This matches the encoding logic where ref_position uses seq_pos_counter (actual node count)
            if self.use_memory_dfs and not node.is_leaf():
                # Check if _current_actual_node_pos is set (from generate_action_masks)
                if hasattr(self, '_current_actual_node_pos') and self._current_actual_node_pos >= 0:
                    # This is an AND/NAND token, record its actual node position
                    # Use encoder-consistent key (parent id + inverted flag) to match dynamic encoding
                    node_key = self._get_node_key(node)
                    # print(f"current_actual_node_pos: {self._current_actual_node_pos}")
                    if node_key is not None:
                        self._node_to_and_token_position[node_key] = self._current_actual_node_pos
                    if self.verbose >= 3:
                        print(f"[DEBUG env.step] Recorded AND token actual_node_pos {self._current_actual_node_pos} for node {node_key}")
            
            if len(self.tree_stack) == 0:
                # print(f"len(roots)={len(self.roots)}")
                # print(f"len(tree_stack)={len(self.tree_stack)}")
                # print(f"len(self.roots)={len(self.roots)}")
                # print(f"self.cur_root_id={self.cur_root_id-1}")
                if len(self.roots) > 0:
                    # Use adaptive TT computation: exact for small, approximate for large
                    if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                        tt_bitarray = compute_tt_adaptive(
                            self.roots[self.cur_root_id-1],
                            num_inputs=self.context_num_inputs,
                            input_tt=self.input_tt_bitarray,
                            threshold=self.mc_tt_threshold,
                            n_samples=self.mc_tt_n_samples,
                            seed=self.mc_tt_seed
                        )
                    else:
                        tt_bitarray = compute_tt(
                            self.roots[self.cur_root_id-1],
                            input_tt=self.input_tt_bitarray
                        )
                    self.current_outputs_tt[self.cur_root_id-1] = tt_bitarray
                self.roots.append(node)
            else:
                # insert node into the tree
                if self.tree_stack[-1].left is None:
                    self.tree_stack[-1].left = node
                else:
                    self.tree_stack[-1].right = node
            self.ref_dict[node.parent] = 1
            # calculate reward
            reward = 0 if node.is_leaf() else -1
            done = False
            # update stack
            if node.is_leaf():
                self.tree_stack.append(node)
                while len(self.tree_stack) > 0 and (self.tree_stack[-1].is_leaf() or (
                        self.tree_stack[-1].left is not None and self.tree_stack[-1].right is not None)):
                    old_node = copy.copy(self.tree_stack[-1])
                    old_node.inverted = False
                    # Use adaptive TT computation: exact for small, approximate for large
                    if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                        tt_bitarray = compute_tt_adaptive(
                            old_node,
                            num_inputs=self.context_num_inputs,
                            input_tt=self.input_tt_bitarray,
                            threshold=self.mc_tt_threshold,
                            n_samples=self.mc_tt_n_samples,
                            seed=self.mc_tt_seed
                        )
                    else:
                        tt_bitarray = compute_tt(old_node, input_tt=self.input_tt_bitarray, cache=self.tt_cache_bitarray)
                    tt_not_bitarray = ~tt_bitarray
                    tt = tt_bitarray.tobytes()
                    tt_not = tt_not_bitarray.tobytes()
                    create_new_hash = True
                    if tt in self.tt_hash_bitarray or tt_not in self.tt_hash_bitarray:
                        inverted = tt_not in self.tt_hash_bitarray
                        new_node = self.tt_hash_bitarray[tt_not if inverted else tt]
                        # Memory DFS: REF only references non-leaf nodes; forbid replacing current AND with a leaf
                        allow_replace = not new_node.is_leaf()
                        if allow_replace and self.ref_dict[new_node] > 0:     # use existing node to replace self.tree_stack[-1]
                            create_new_hash = False
                            new_node_with_inv = NodeWithInv(new_node, (not inverted) if self.tree_stack[-1].inverted else inverted)
                            
                            # Transfer AND token position mapping if node is being replaced (preserve _actual_node_pos semantics)
                            if self.use_memory_dfs:
                                old_node_key = self._get_node_key(self.tree_stack[-1])
                                if old_node_key is not None and old_node_key in self._node_to_and_token_position:
                                    and_token_pos = self._node_to_and_token_position[old_node_key]
                                    new_node_key = self._get_node_key(new_node_with_inv)
                                    if new_node_key is not None:
                                        self._node_to_and_token_position[new_node_key] = and_token_pos
                                    del self._node_to_and_token_position[old_node_key]
                                    if self.verbose >= 3:
                                        print(f"[DEBUG env.step] Transferred AND token position {and_token_pos} from node {old_node_key} to {new_node_key}")
                            
                            # Save old node's parent before replacing tree_stack[-1]
                            old_node_parent = self.tree_stack[-1].parent
                            self.tt_cache_bitarray[new_node_with_inv] = tt_not_bitarray if self.tree_stack[-1].inverted else tt_bitarray
                            if self.use_controllability_dont_cares:
                                self.tt_cache_bitarray_compressed[new_node_with_inv] = self.compress(self.tt_cache_bitarray[new_node_with_inv])
                            # Update parent's reference before updating tree_stack[-1]
                            if len(self.tree_stack) > 1:
                                if self.tree_stack[-2].left is self.tree_stack[-1]:
                                    self.tree_stack[-2].left = new_node_with_inv
                                elif self.tree_stack[-2].right is self.tree_stack[-1]:
                                    self.tree_stack[-2].right = new_node_with_inv
                            else:
                                self.roots[self.cur_root_id] = new_node_with_inv
                            # CRITICAL FIX: Update tree_stack[-1] to new_node_with_inv so that complete_node processing uses the correct node
                            # This ensures that _get_node_key(complete_node) matches the transferred key in _node_to_and_token_position
                            self.tree_stack[-1] = new_node_with_inv
                            self.ref_dict[new_node] += 1
                            # Only deref if the parent is in ref_dict
                            if old_node_parent in self.ref_dict:
                                self.ref_dict[old_node_parent] -= 1
                                try:
                                    v1 = deref_node(old_node_parent, self.ref_dict, self.context_nodes)
                                    reward += v1
                                except KeyError:
                                    if self.verbose >= 3:
                                        print(f"[DEBUG env.step] Skipping deref_node for replaced node (parent not in ref_dict)")
                        elif allow_replace:
                            # new_node exists but ref_dict[new_node] <= 0, need to reuse it (and we allow replace: non-leaf only in Memory DFS)
                            # Create new_node_with_inv first
                            new_node_with_inv = NodeWithInv(new_node, (not inverted) if self.tree_stack[-1].inverted else inverted)
                            create_new_hash = False
                            
                            # Transfer AND token position mapping if node is being replaced (preserve _actual_node_pos semantics)
                            if self.use_memory_dfs:
                                old_node_key = self._get_node_key(self.tree_stack[-1])
                                if old_node_key is not None and old_node_key in self._node_to_and_token_position:
                                    and_token_pos = self._node_to_and_token_position[old_node_key]
                                    new_node_key = self._get_node_key(new_node_with_inv)
                                    if new_node_key is not None:
                                        self._node_to_and_token_position[new_node_key] = and_token_pos
                                    del self._node_to_and_token_position[old_node_key]
                                    if self.verbose >= 3:
                                        print(f"[DEBUG env.step] Transferred AND token position {and_token_pos} from node {old_node_key} to {new_node_key}")
                            
                            # Update uncompressed TT cache
                            self.tt_cache_bitarray[new_node_with_inv] = tt_not_bitarray if self.tree_stack[-1].inverted else tt_bitarray
                            
                            # Use adaptive TT computation for compressed TT
                            if self.use_controllability_dont_cares:
                                self.tt_cache_bitarray_compressed[new_node_with_inv] = self.compress(self.tt_cache_bitarray[new_node_with_inv])
                            else:
                                if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                                    tt_bitarray_compressed = compute_tt_adaptive(
                                        old_node,
                                        num_inputs=len(self.input_tt_bitarray_compressed[0]) if len(self.input_tt_bitarray_compressed) > 0 else self.context_num_inputs,
                                        input_tt=self.input_tt_bitarray_compressed,
                                        threshold=self.mc_tt_threshold,
                                        n_samples=self.mc_tt_n_samples,
                                        seed=self.mc_tt_seed
                                    )
                                else:
                                    tt_bitarray_compressed = compute_tt(old_node, input_tt=self.input_tt_bitarray_compressed, cache=self.tt_cache_bitarray_compressed)
                                self.tt_cache_bitarray_compressed[new_node_with_inv] = (~tt_bitarray_compressed) if self.tree_stack[-1].inverted else tt_bitarray_compressed
                            # Save old node's parent before replacing tree_stack[-1]
                            old_node_parent = self.tree_stack[-1].parent
                            if len(self.tree_stack) > 1:
                                if self.tree_stack[-2].left is self.tree_stack[-1]:
                                    self.tree_stack[-2].left = new_node_with_inv
                                else:
                                    self.tree_stack[-2].right = new_node_with_inv
                            else:
                                self.roots[self.cur_root_id] = new_node_with_inv
                            # CRITICAL FIX: Update tree_stack[-1] to new_node_with_inv so that complete_node processing uses the correct node
                            # This ensures that _get_node_key(complete_node) matches the transferred key in _node_to_and_token_position
                            self.tree_stack[-1] = new_node_with_inv
                            self.ref_dict[new_node] += 1
                            # Only deref if the parent is in ref_dict (may not be for REF_TOKEN deepcopied nodes)
                            if old_node_parent in self.ref_dict:
                                self.ref_dict[old_node_parent] -= 1
                                try:
                                    v1 = deref_node(old_node_parent, self.ref_dict, self.context_nodes)
                                    reward += v1
                                except KeyError:
                                    # REF_TOKEN deepcopied node's children may not be in ref_dict
                                    # This is expected for referenced nodes, skip deref
                                    if self.verbose >= 3:
                                        print(f"[DEBUG env.step] Skipping deref_node for REF_TOKEN node (children not in ref_dict)")
                            else:
                                if self.verbose >= 3:
                                    print(f"[DEBUG env.step] Parent {old_node_parent} not in ref_dict, skipping deref")
                    if create_new_hash:
                        self.tt_hash_bitarray[tt_bitarray.tobytes()] = self.tree_stack[-1].parent
                        self.tt_cache_bitarray[self.tree_stack[-1]] = tt_not_bitarray if self.tree_stack[-1].inverted else tt_bitarray
                        if self.use_controllability_dont_cares:
                            self.tt_cache_bitarray_compressed[self.tree_stack[-1]] = self.compress(self.tt_cache_bitarray[self.tree_stack[-1]])
                        else:
                            # Use adaptive TT computation for compressed TT
                            if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                                tt_bitarray_compressed = compute_tt_adaptive(
                                    self.tree_stack[-1],
                                    num_inputs=len(self.input_tt_bitarray_compressed[0]) if len(self.input_tt_bitarray_compressed) > 0 else self.context_num_inputs,
                                    input_tt=self.input_tt_bitarray_compressed,
                                    threshold=self.mc_tt_threshold,
                                    n_samples=self.mc_tt_n_samples,
                                    seed=self.mc_tt_seed
                                )
                            else:
                                tt_bitarray_compressed = compute_tt(self.tree_stack[-1],
                                                                   input_tt=self.input_tt_bitarray_compressed,
                                                                   cache=self.tt_cache_bitarray_compressed)
                            self.tt_cache_bitarray_compressed[self.tree_stack[-1]] = tt_bitarray_compressed
                        if self.context_hash is not None and (tt in self.context_hash or tt_not in self.context_hash):
                            v1 = deref_node(self.tree_stack[-1].parent, self.ref_dict, self.context_nodes)
                            self.context_nodes.add(self.tree_stack[-1].parent)
                            self.context_records[self.tree_stack[-1].parent] = tt
                            reward += v1
                    # Track decoded node for REF_TOKEN references (only when node is complete)
                    # A node is complete when it's popped from tree_stack (left and right are both set)
                    if self.use_memory_dfs:
                        complete_node = self.tree_stack[-1] if len(self.tree_stack) > 0 else None
                        if complete_node is not None and not complete_node.is_leaf():
                            complete_node_copy = copy.copy(complete_node)
                            decoded_idx = len(self.decoded_nodes)
                            self.decoded_nodes.append(complete_node_copy)
                            if self.decoded_nodes_tt is not None:
                                node_tt = self._get_or_compute_tt(complete_node)
                                self.decoded_nodes_tt.append(node_tt if node_tt is not None else None)
                            
                            node_key = self._get_node_key(complete_node)
                            mapped = False
                            if node_key is not None and node_key in self._node_to_and_token_position:
                                and_token_pos = self._node_to_and_token_position[node_key]
                                self.seq_position_to_decoded_idx[and_token_pos] = decoded_idx
                                if self.verbose >= 3:
                                    print(f"[DEBUG env.step] Mapped AND token position {and_token_pos} -> decoded_idx {decoded_idx} (node_key={node_key})")
                                del self._node_to_and_token_position[node_key]
                                mapped = True
                                
                            # # FALLBACK: If node_key doesn't match, try to find any unmapped position that might correspond to this node
                            # # This handles cases where node replacement changed the key but the position mapping was lost
                            # if not mapped and len(self._node_to_and_token_position) > 0:
                            #     # Try to match by checking if any unmapped position corresponds to a similar node
                            #     # This is a fallback to handle edge cases where key matching fails
                            #     for fallback_key, fallback_pos in list(self._node_to_and_token_position.items()):
                            #         # Only use fallback if there's exactly one unmapped position (to avoid false matches)
                            #         if len(self._node_to_and_token_position) == 1:
                            #             self.seq_position_to_decoded_idx[fallback_pos] = decoded_idx
                            #             if self.verbose >= 3:
                            #                 print(f"[DEBUG env.step] FALLBACK: Mapped AND token position {fallback_pos} -> decoded_idx {decoded_idx} (fallback_key={fallback_key}, node_key={node_key})")
                            #             del self._node_to_and_token_position[fallback_key]
                            #             mapped = True
                            #             break
                                                        
                            if self.current_seq_position >= 0:
                                self.current_seq_position += 1
                    
                    self.tree_stack.pop()
                if len(self.tree_stack) == 0 and len(self.roots) == self.num_outputs:
                    # print(f"[DEBUG3 env.step] debug: num outputs = {self.num_outputs}, tree_stack length = {len(self.tree_stack)}")
                    self.gen_eos = True  # next token should be EOS
                    done = False
            else:
                self.tree_stack.append(node)
        self.tokens.append(token)
        self.t += 1

        # Update sequence position for regular tokens (not REF_TOKEN or ref_position)
        if self.use_memory_dfs and not (token == self.encoder.REF_TOKEN or self._expecting_ref_position):
            # This is a regular token that creates a new node
            # The sequence position will be set when the node is complete (above)
            # But we need to track it here for the first time
            if self.current_seq_position < 0:
                self.current_seq_position = 0

        current_delay = compute_critical_path(self.roots)
        reward = self._compose_normalized_reward(reward, current_delay)

        if self.is_finished:
            if self.gen_eos:
                reward += 50.0
            else:
                completed_outputs = self.cur_root_id
                reward = (completed_outputs / self.num_outputs) * 10 - 50
                # remaining_steps = max(0, self.max_length - self.t)
                # reward -= (float(remaining_steps) * 1.0 + 10.0)
            # print(f"final reward = {np.sum(self.rewards) + reward}")
        self.rewards.append(reward)
        if len(self.tree_stack) == 0 and self.cur_root_id < self.num_outputs and self.use_controllability_dont_cares:
            self.initialize_care_set_tt()
        self.action_masks.append(self.gen_action_mask())
        # print(f"node_to_and_token_position: {self._node_to_and_token_position}")
        # print(f"len(decoded_nodes): {len(self.decoded_nodes)}")
        return reward, done

    def ppo_step(self, action):
        """PPO训练用的step方法。生成失败或异常时返回 unfinished_penalty 并 done=True，避免卡死或崩溃。"""
        # 如果环境已结束，继续返回结束状态
        if self.is_finished:
            return self._get_obs(), 0, True, {}

        try:
            # 执行动作（使用原始step方法）
            reward, done = self.step(action)

            # 检查是否达到最大长度（未完成则给惩罚并结束）
            if self.t >= self.max_length - 1 and not self.is_finished:
                done = True
                reward = self.unfinished_penalty

            return self._get_obs(), reward, done, {}
        except Exception as e:
            if self.verbose >= 1:
                print(f"[ERROR ppo_step] circuit generation failed: {e}")
            self.is_finished = True
            return self._get_obs(), self.unfinished_penalty, True, {}

    def detect_token_cycle(self, window_size=6):
        """
        检测最近的 token 序列中是否有循环模式，特别是 AND/NAND 的简单循环
        
        检测模式如：[input, gate, input, gate, input, gate] 或 [input, gate, input, gate]
        其中 gate 是 AND 或 NAND
        
        Args:
            window_size: 检测窗口大小（默认6，检测最近6个token）
        
        Returns:
            is_cycling: bool, 是否检测到循环
            cycle_tokens: set, 循环模式中的 token 集合（需要禁止的 token）
        """
        if len(self.tokens) < 4:
            return False, set()
        
        recent = self.tokens[-window_size:]
        
        # 获取 AND 和 NAND 的 token ID
        # vocab_size includes REF_TOKEN if using memory DFS, so use vocab_size - 5, -4
        and_token_id = self.vocab_size - 5
        nand_token_id = self.vocab_size - 4
        gate_tokens = {and_token_id, nand_token_id}
        
        # 输入 token 的范围：2 到 2 + num_inputs * 2
        input_token_start = 2
        input_token_end = 2 + self.num_inputs * 2
        
        # 检测2-token循环：[A, B, A, B] 或 [A, B, A, B, A, B]
        # 其中 A 是输入 token，B 是 AND/NAND token
        if len(recent) >= 4:
            # 检查最后4个token是否是 [input, gate, input, gate] 模式
            if (recent[-4] >= input_token_start and recent[-4] < input_token_end and
                recent[-3] in gate_tokens and
                recent[-2] == recent[-4] and  # 相同的输入
                recent[-1] == recent[-3]):    # 相同的门
            
                # 检查是否至少重复2次（共4个token）
                cycle_tokens = {recent[-4], recent[-3]}  # 输入和门
                
                # 如果窗口更大，检查是否继续循环
                if len(recent) >= 6:
                    if recent[-6] == recent[-4] == recent[-2]:
                        # 确认是循环模式
                        return True, cycle_tokens
                
                # 至少检测到一次 [input, gate, input, gate] 模式
                return True, cycle_tokens
        
        return False, set()

    def _safe_copy_node(self, node, max_depth=200, visited=None):
        """
        Safely copy a node to avoid infinite recursion in deepcopy.
        Uses shallow copy for leaf nodes and limited-depth deepcopy for non-leaf nodes.
        """
        if visited is None:
            visited = set()
        
        if node is None:
            return None
        
        # Check for cycles
        node_id = id(node)
        if node_id in visited:
            # Cycle detected, return shallow copy
            return copy.copy(node)
        
        # If too deep, return shallow copy
        if max_depth <= 0:
            return copy.copy(node)
        
        # If leaf node, use shallow copy (no need to track in visited)
        if node.is_leaf():
            return copy.copy(node)
        
        # For non-leaf nodes, create a new node and recursively copy children
        visited.add(node_id)
        try:
            if isinstance(node, NodeWithInv):
                # Create a new NodeWithInv with copied parent
                new_parent = Node(node.parent.var, None, None)
                new_node = NodeWithInv(new_parent, node.inverted, node.output_symbol)
                new_node.parent.left = self._safe_copy_node(node.left, max_depth - 1, visited) if node.left else None
                new_node.parent.right = self._safe_copy_node(node.right, max_depth - 1, visited) if node.right else None
            else:
                # Create a new Node
                new_node = Node(node.var, None, None)
                new_node.left = self._safe_copy_node(node.left, max_depth - 1, visited) if node.left else None
                new_node.right = self._safe_copy_node(node.right, max_depth - 1, visited) if node.right else None
                new_node.input_symbol = node.input_symbol
            return new_node
        finally:
            # Only remove if it was added (non-leaf nodes)
            visited.discard(node_id)
    
    def _safe_copy_tree_stack(self, tree_stack, max_depth=200):
        """
        Safely copy tree_stack to avoid infinite recursion in deepcopy.
        """
        if not tree_stack:
            return []
        
        visited = set()
        copied_stack = []
        for node in tree_stack:
            copied_stack.append(self._safe_copy_node(node, max_depth, visited))
        return copied_stack

    def _get_or_compute_tt(self, node, compressed=False, base_tt=None):
        """
        Retrieve (or compute) the truth table for a node.
        When compressed=True, the returned TT matches current care-set compression.
        """
        if node is None:
            return None
        if not compressed:
            cache = self.tt_cache_bitarray
            if cache is None:
                return None
            if node in cache:
                return cache[node]
            if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                tt_bitarray = compute_tt_adaptive(
                    node,
                    num_inputs=self.context_num_inputs,
                    input_tt=self.input_tt_bitarray,
                    threshold=self.mc_tt_threshold,
                    n_samples=self.mc_tt_n_samples,
                    seed=self.mc_tt_seed
                )
            else:
                tt_bitarray = compute_tt(node, input_tt=self.input_tt_bitarray, cache=cache)
            cache[node] = tt_bitarray
            return tt_bitarray
        cache = self.tt_cache_bitarray_compressed
        if cache is None:
            return None
        if node in cache:
            return cache[node]
        if self.use_controllability_dont_cares:
            if base_tt is None:
                base_tt = self._get_or_compute_tt(node, compressed=False)
            if base_tt is None:
                return None
            tt_bitarray = self.compress(base_tt)
        else:
            if self.use_monte_carlo_tt and self.context_num_inputs > self.mc_tt_threshold:
                sample_inputs = len(self.input_tt_bitarray_compressed[0]) if len(self.input_tt_bitarray_compressed) > 0 else self.context_num_inputs
                tt_bitarray = compute_tt_adaptive(
                    node,
                    num_inputs=sample_inputs,
                    input_tt=self.input_tt_bitarray_compressed,
                    threshold=self.mc_tt_threshold,
                    n_samples=self.mc_tt_n_samples,
                    seed=self.mc_tt_seed
                )
            else:
                tt_bitarray = compute_tt(node, input_tt=self.input_tt_bitarray_compressed, cache=cache)
        cache[node] = tt_bitarray
        return tt_bitarray

    def _compress_tt_for_current_context(self, tt):
        """Return TT compressed under the current care-set (or original if no compression)."""
        if tt is None:
            return None
        if not self.use_controllability_dont_cares:
            return tt
        return self.compress(tt)

    def _get_decoded_node_compressed_tt(self, idx):
        """Get the compressed TT for a decoded node index (recomputed for current care-set if needed)."""
        if self.decoded_nodes_tt is None or idx < 0 or idx >= len(self.decoded_nodes_tt):
            return None
        base_tt = self.decoded_nodes_tt[idx]
        if base_tt is None:
            return None
        return self._compress_tt_for_current_context(base_tt)

    def gen_action_mask(self): ################################################################
        try:
            action_mask_ba = bitarray.util.zeros(self.vocab_size)
                       
            cur_node = None if len(self.tree_stack) == 0 else self.tree_stack[-1]
            action_mask_ba[self.EOS] = self.gen_eos or (cur_node is None and not self.is_finished and len(self.roots) == self.num_outputs)
            action_mask_ba[self.PAD] = self.is_finished
            
            if self.gen_eos:
                if self.verbose >= 3:
                    print(f"[DEBUG gen_action_mask] gen_eos=True, allowing only EOS token")
                return np.array(action_mask_ba.tolist(), dtype=bool)
            
            if not self.is_finished and not self.gen_eos and \
                    (self.max_inference_reward is None or self.cumulative_reward >= self.max_inference_reward):

                node = NodeWithInv(parent=Node(var=-2, left=None, right=None), inverted=False)
                probe_side = None
                if cur_node is None:
                    is_root = True
                else:
                    is_root = False
                    if cur_node.left is None:
                        cur_node.left = node
                        probe_side = "left"
                    else:
                        cur_node.right = node
                        probe_side = "right"

                def _cleanup_conflict_probe(probe_side):
                    if is_root or probe_side is None or cur_node is None:
                        return
                    if probe_side == "left":
                        cur_node.left = None
                    else:
                        cur_node.right = None

                # consider error rate from completed roots
                current_tt_conflict = []
                if len(self.current_outputs_tt) > 0 and self.cur_root_id < len(self.tts_bitarray_compressed):
                    for root_id in self.current_outputs_tt.keys():
                        if root_id < len(self.tts_bitarray_compressed):
                            # Full-domain TT in current_outputs_tt must align with tts_bitarray_compressed
                            # (care-set compressed) before XOR, or conflict bits are wrong / empty.
                            left_tt = self.compress(self.current_outputs_tt[root_id])
                            right_tt = self.tts_bitarray_compressed[root_id]
                            if len(left_tt) != len(right_tt):
                                msg = (
                                    f"TT length mismatch at root_id={root_id} after compress: "
                                    f"pred={len(left_tt)} vs tts_compressed={len(right_tt)}"
                                )
                                if self.strict_ref_consistency:
                                    raise ValueError(msg)
                                if self.verbose >= 1:
                                    print(f"[WARN gen_action_mask] {msg}")
                                continue
                            conflict_bits = (left_tt ^ right_tt)
                            current_tt_conflict.append(conflict_bits)
                
                def _compute_valid_ref_positions(candidate_tts_local):
                    """
                    Compute valid ref positions (sequence positions) for REF_TOKEN position step.
                    Returns: (allowed_seq_positions: list[int], valid_indices: bitarray/bitarray-like)
                    """
                    if not self.seq_position_to_decoded_idx:
                        return [], bitarray.util.zeros(len(candidate_tts_local))
                    try:
                        has_conflict_ref, _ = check_conflict(
                            self.tree_stack,
                            self.tts_bitarray_compressed[self.cur_root_id],
                            self.input_tt_bitarray_compressed,
                            current_tt_conflict,
                            self.tt_cache_bitarray_compressed,
                            tolerance=self.error_rate_threshold,
                            verbose=self.verbose,
                            initial_tt=candidate_tts_local,
                            base_input_tt=self.input_tt_bitarray_compressed,
                        )
                        valid_indices_local = ~has_conflict_ref
                    except Exception as e:
                        if self.verbose >= 1:
                            print(f"[WARN gen_action_mask] check_conflict failed for REF candidates: {e}")
                        # Conservative fallback: when conflict check fails, disable all REF candidates
                        # to avoid training/inference drift from permissive masks.
                        valid_indices_local = bitarray.util.zeros(len(candidate_tts_local))

                    allowed_seq_positions = []
                    for seq_pos, decoded_idx in self.seq_position_to_decoded_idx.items():
                        if 0 <= decoded_idx < len(valid_indices_local) and valid_indices_local[decoded_idx]:
                            if seq_pos < self.vocab_size:
                                # REF only references non-leaf nodes (encoding only emits REF for AND/NAND)
                                if decoded_idx < len(self.decoded_nodes) and not self.decoded_nodes[decoded_idx].is_leaf():
                                    allowed_seq_positions.append(seq_pos)
                    return allowed_seq_positions, valid_indices_local

                try:
                    if self.use_memory_dfs and self._expecting_ref_position:
                        if not self.decoded_nodes:
                            if self.strict_ref_consistency:
                                raise ValueError("REF position requested but decoded_nodes is empty")
                            return np.array(action_mask_ba.tolist(), dtype=bool)


                        candidate_tts = []
                        tt_len = len(self.input_tt_bitarray_compressed[0]) if self.input_tt_bitarray_compressed else 0
                        dummy_tt = bitarray.util.zeros(tt_len) 
                        
                        for i in range(len(self.decoded_nodes)):
                            tt = self._get_decoded_node_compressed_tt(i)
                            if tt is None:
                                print(f"DEBUG: tt is None for decoded_node {i}")
                            if self.verbose > 2:
                                print(f"{i}: {tt}")
                            candidate_tts.append(tt if tt is not None else dummy_tt)


                        allowed_seq_positions, valid_indices = _compute_valid_ref_positions(candidate_tts)

                        if self.verbose >= 2:
                            keys_preview = sorted(self.seq_position_to_decoded_idx.keys())[:20] if self.seq_position_to_decoded_idx else []
                            print(f"[DEBUG gen_action_mask] REF position step:"
                                  f" decoded_nodes={len(self.decoded_nodes)}"
                                  f" mapped_positions={len(self.seq_position_to_decoded_idx)} (preview={keys_preview})"
                                  f" allowed_positions={len(allowed_seq_positions)}"
                                  f" valid_candidates={int(valid_indices.count()) if hasattr(valid_indices,'count') else 'n/a'}")

                        # Populate mask with allowed seq positions (check_conflict-validated only).
                        for seq_pos in allowed_seq_positions:
                            action_mask_ba[seq_pos] = True

                        if not action_mask_ba.any():
                            msg = (
                                "REF position mask is empty: no seq positions pass check_conflict; "
                                "not using unverified fallback"
                            )
                            if self.strict_ref_consistency:
                                raise ValueError(msg)
                            if self.verbose >= 1:
                                print(f"[WARN gen_action_mask] {msg}")
                        
                        # NOTE: We intentionally do NOT include _node_to_and_token_position here
                        # because those nodes are not yet complete and cannot be referenced by REF_TOKEN
                        # REF_TOKEN can only reference nodes that have been completed and stored in decoded_nodes
                        _cleanup_conflict_probe(probe_side)
                        # if not is_root:
                        #     if cur_node.right is None:
                        #         cur_node.left = None
                        #     else:
                        #         cur_node.right = None

                        return np.array(action_mask_ba.tolist(), dtype=bool)

                    has_conflict_ba, completeness_ba = check_conflict(self.tree_stack, self.tts_bitarray_compressed[self.cur_root_id], ####################check conflict
                                                                      self.input_tt_bitarray_compressed, current_tt_conflict, self.tt_cache_bitarray_compressed,
                                                                      tolerance=self.error_rate_threshold, verbose=self.verbose)
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"[DEBUG gen_action_mask] ERROR in check_conflict: {e}")
                        import traceback
                        traceback.print_exc()
                    # Return a default mask on error using fixed positions
                    action_mask_ba = bitarray.util.zeros(self.vocab_size)
                    action_mask_ba[2: 2 + self.num_inputs * 2] = bitarray.util.ones(self.num_inputs * 2)
                    # Fixed positions for AND, NAND (vocab_size - 5, -4)
                    and_token_id = self.vocab_size - 5
                    nand_token_id = self.vocab_size - 4
                    action_mask_ba[and_token_id] = True
                    action_mask_ba[nand_token_id] = True
                    return np.array(action_mask_ba.tolist(), dtype=bool)
                

                value_action_mask_ba = ~has_conflict_ba
                if self.verbose >= 2:
                    print(f"[DEBUG gen_action_mask] check_conflict succeeded: value_action_mask_ba length={len(value_action_mask_ba)}, "
                          f"expected length={self.num_inputs * 2}")
                    print(f"[DEBUG gen_action_mask] has_conflict_ba: {has_conflict_ba.tolist()[:10]}... (showing first 10), "
                          f"all True={all(has_conflict_ba.tolist())}, all False={all(not x for x in has_conflict_ba.tolist())}")
                    print(f"[DEBUG gen_action_mask] value_action_mask_ba: {value_action_mask_ba.tolist()[:10]}... (showing first 10), "
                          f"all True={all(value_action_mask_ba.tolist())}, all False={all(not x for x in value_action_mask_ba.tolist())}")
                    print(f"[DEBUG gen_action_mask] tree_stack length={len(self.tree_stack)}, cur_root_id={self.cur_root_id}")
                # print(f"value_action_mask_ba={value_action_mask_ba}")
                action_mask_ba[2: 2 + len(value_action_mask_ba)] = value_action_mask_ba             # PAD: 0, EOS:1
                
                # Fixed positions for AND, NAND, Constant 0, Constant 1
                and_token_id = self.vocab_size - 5
                nand_token_id = self.vocab_size - 4
                constant_0_id = self.vocab_size - 3
                constant_1_id = self.vocab_size - 2
                
                if self.and_always_available:
                    action_mask_ba[and_token_id] = True
                    action_mask_ba[nand_token_id] = True
                else:
                    and_nand_available = not len(self.tree_stack) >= self.max_inference_tree_depth - 2  # not ((value_action_mask_ba & completeness_ba).any() or len(self.tree_stack) >= self.max_inference_tree_depth - 2)
                    # and_nand_available = not ((value_action_mask_ba & completeness_ba).any() or len(self.tree_stack) >= self.max_inference_tree_depth - 2)
                    action_mask_ba[and_token_id] = and_nand_available
                    action_mask_ba[nand_token_id] = and_nand_available

                if self.use_memory_dfs and not self._expecting_ref_position:
                    # Only allow REF_TOKEN if there exists at least one valid ref position
                    # for the *next* step; otherwise the position-step mask can become empty
                    # and argmax() can degenerate to 0 -> repeated [518,0].
                    if len(self.decoded_nodes) > 0 and self.seq_position_to_decoded_idx:
                        candidate_tts = []
                        tt_len = len(self.input_tt_bitarray_compressed[0]) if self.input_tt_bitarray_compressed else 0
                        dummy_tt = bitarray.util.zeros(tt_len)
                        for i in range(len(self.decoded_nodes)):
                            tt = self._get_decoded_node_compressed_tt(i)
                            candidate_tts.append(tt if tt is not None else dummy_tt)
                        allowed_seq_positions, _ = _compute_valid_ref_positions(candidate_tts)
                        if allowed_seq_positions:
                            ref_token_id = self.encoder.REF_TOKEN
                            if ref_token_id < self.vocab_size:
                                action_mask_ba[ref_token_id] = True
                        elif self.verbose >= 2:
                            print("[DEBUG gen_action_mask] REF_TOKEN disabled: no valid ref positions under current conflict check.")
                
                action_mask_ba[constant_0_id] = value_action_mask_ba[-2] if len(value_action_mask_ba) >= 2 else False
                action_mask_ba[constant_1_id] = value_action_mask_ba[-1] if len(value_action_mask_ba) >= 2 else False
                
                _cleanup_conflict_probe(probe_side)
                # if not is_root:
                #     if cur_node.right is None:
                #         cur_node.left = None
                #     else:
                #         cur_node.right = None
        except Exception as e:
            if self.verbose >= 2:
                print(f"[DEBUG gen_action_mask] ERROR in gen_action_mask: {e}")
                import traceback
                traceback.print_exc()
            # Return a default mask on error using fixed positions
            action_mask_ba = bitarray.util.zeros(self.vocab_size)
            action_mask_ba[2: 2 + self.num_inputs * 2] = bitarray.util.ones(self.num_inputs * 2)
            # Fixed positions for AND, NAND
            and_token_id = self.vocab_size - 5
            nand_token_id = self.vocab_size - 4
            action_mask_ba[and_token_id] = True
            action_mask_ba[nand_token_id] = True
            return np.array(action_mask_ba.tolist(), dtype=bool)

        # print(f"action_mask_ba: {action_mask_ba.tolist()}")
        if not action_mask_ba.any():
            # print(f"DEBUG: no action mask available")
            action_mask_ba[self.EOS] = True
        return np.array(action_mask_ba.tolist(), dtype=bool) 