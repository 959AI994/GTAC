from __future__ import annotations

import copy
import numpy as np
import bitarray
import bitarray.util
from gtac.utils import Node, NodeWithInv, compute_input_tt, compute_tt, check_conflict, get_inputs_rec, detect_circle, compute_critical_path
from gtac.encoding import stack_to_encoding, int_to_node, deref_node


class LogicNetworkEnv:
    def __init__(self,
                 tts,
                 num_inputs,
                 context_num_inputs=None,
                 input_tt=None,
                 init_care_set_tt=None,                 # for the first output (which can be computed in advance) or for all the outputs (list[num_outputs])
                 max_tree_depth=32,
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
                 ):
        # assert len(tts) == 2
        self.num_outputs = len(tts)
        self.num_inputs = num_inputs
        self.context_num_inputs = context_num_inputs if context_num_inputs is not None else num_inputs
        self.tts_bitarray = tts
        self.init_care_set_tt = init_care_set_tt if init_care_set_tt is not None else bitarray.util.ones(2 ** self.context_num_inputs)
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
        self.unfinished_penalty = -1  # 降低未完成惩罚，避免过度惩罚
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

        if input_tt is None:
            self.input_tt_bitarray = compute_input_tt(self.context_num_inputs)
        else:
            self.input_tt_bitarray = input_tt
        self.tt_cache_bitarray = {Node(i // 2, None, None): v
                                  for i, v in enumerate(self.input_tt_bitarray) if i % 2 == 0}
        self.tt_hash_bitarray = {v.tobytes(): node for node, v in self.tt_cache_bitarray.items()}
        self.vocab_size = 2 + num_inputs * 2 + 2 + 2                    # PAD, EOS, PI, ~PI, AND, NAND, 0, 1 
        self.ref_dict = {k: 1 for k in self.tt_cache_bitarray.keys()}
        self.context_nodes = set()
        self.context_records = dict()

        if self.use_controllability_dont_cares:
            self.initialize_care_set_tt()
        else:
            assert tts_compressed is not None
            assert init_care_set_tt is None
            self.compress_indices = None
            self.input_tt_bitarray_compressed = compute_input_tt(len(self.input_tt_bitarray) // 2)
            self.tts_bitarray_compressed = tts_compressed
            self.tt_cache_bitarray_compressed = {Node(i // 2, None, None): v
                                                 for i, v in enumerate(self.input_tt_bitarray_compressed) if i % 2 == 0}

        self.action_masks.append(self.gen_action_mask())

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
        self.cur_root_id = 0
        
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

    def initialize_care_set_tt(self):       # both controllability and observability don't cares
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
        assert len(self.input_tt_bitarray) // 2 <= 8  # one byte
        for i, tt in enumerate(self.input_tt_bitarray):
            if i % 2 == 0:
                a.extend(tt[self.care_set_tt].unpack(one=(1 << (i // 2)).to_bytes(1, 'big')))
        a_np = np.frombuffer(a, dtype=np.uint8).reshape(len(self.input_tt_bitarray) // 2, len_care_set)
        a_np = np.sum(a_np, axis=0, dtype=np.uint8)
        a_np_unique, self.compress_indices = np.unique(a_np, return_index=True)

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
        return (tt[self.care_set_tt])[self.compress_indices]

    def step(self, token): ###########################################################################
        self.positional_encodings.append(stack_to_encoding(self.tree_stack, self.cur_root_id, self.max_tree_depth))
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
            reward, done = self.unfinished_penalty, True
        else:
            node = int_to_node(token, self.num_inputs)
            # print(f"token:{token}")
            if len(self.tree_stack) == 0:
                # print(f"len(roots)={len(self.roots)}")
                # print(f"len(tree_stack)={len(self.tree_stack)}")
                # print(f"len(self.roots)={len(self.roots)}")
                # print(f"self.cur_root_id={self.cur_root_id-1}")
                if len(self.roots) > 0:
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
            # print(node)
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
                    tt_bitarray = compute_tt(old_node, input_tt=self.input_tt_bitarray, cache=self.tt_cache_bitarray)
                    tt_not_bitarray = ~tt_bitarray
                    tt = tt_bitarray.tobytes()
                    tt_not = tt_not_bitarray.tobytes()
                    create_new_hash = True
                    if tt in self.tt_hash_bitarray or tt_not in self.tt_hash_bitarray:
                        inverted = tt_not in self.tt_hash_bitarray
                        new_node = self.tt_hash_bitarray[tt_not if inverted else tt]
                        if self.ref_dict[new_node] > 0:     # use existing node to replace self.tree_stack[-1]
                            create_new_hash = False
                            new_node_with_inv = NodeWithInv(new_node, (not inverted) if self.tree_stack[-1].inverted else inverted)
                            self.tt_cache_bitarray[new_node_with_inv] = tt_not_bitarray if self.tree_stack[-1].inverted else tt_bitarray
                            if self.use_controllability_dont_cares:
                                self.tt_cache_bitarray_compressed[new_node_with_inv] = self.compress(self.tt_cache_bitarray[new_node_with_inv])
                            else:
                                tt_bitarray_compressed = compute_tt(old_node, input_tt=self.input_tt_bitarray_compressed, cache=self.tt_cache_bitarray_compressed)
                                self.tt_cache_bitarray_compressed[new_node_with_inv] = (~tt_bitarray_compressed) if self.tree_stack[-1].inverted else tt_bitarray_compressed
                            if len(self.tree_stack) > 1:
                                if self.tree_stack[-2].left is self.tree_stack[-1]:
                                    self.tree_stack[-2].left = new_node_with_inv
                                else:
                                    self.tree_stack[-2].right = new_node_with_inv
                            else:
                                self.roots[self.cur_root_id] = new_node_with_inv
                            self.ref_dict[new_node] += 1
                            self.ref_dict[self.tree_stack[-1].parent] -= 1
                            v1 = deref_node(self.tree_stack[-1].parent, self.ref_dict, self.context_nodes)
                            reward += v1
                    if create_new_hash:
                        self.tt_hash_bitarray[tt_bitarray.tobytes()] = self.tree_stack[-1].parent
                        self.tt_cache_bitarray[self.tree_stack[-1]] = tt_not_bitarray if self.tree_stack[-1].inverted else tt_bitarray
                        if self.use_controllability_dont_cares:
                            self.tt_cache_bitarray_compressed[self.tree_stack[-1]] = self.compress(self.tt_cache_bitarray[self.tree_stack[-1]])
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
                    self.tree_stack.pop()
                if len(self.tree_stack) == 0 and len(self.roots) == self.num_outputs:
                    self.gen_eos = True  # next token should be EOS
                    done = True
            else:
                self.tree_stack.append(node)
        self.tokens.append(token)
        self.t += 1

        # current_delay = compute_critical_path(self.roots)
        
        # compute delta delay
        # delta_delay = self.prev_delay - current_delay
        # reward = (
        #     self.w_gate * reward +
        #     self.w_delay * delta_delay
        # )
        # self.prev_delay = current_delay
        self.rewards.append(reward)
        if len(self.tree_stack) == 0 and self.cur_root_id < self.num_outputs and self.use_controllability_dont_cares:
            self.initialize_care_set_tt()
        self.action_masks.append(self.gen_action_mask())
        # print(f"action_masks={self.gen_action_mask()}")
        return reward, done

    def ppo_step(self, action):
        """PPO训练用的step方法"""
        # 如果环境已结束，继续返回结束状态
        if self.is_finished:
            return self._get_obs(), 0, True, {}
        
        # 执行动作（使用原始step方法）
        reward, done = self.step(action)
        
        # 检查是否达到最大长度
        if self.t >= self.max_length - 1 and not self.is_finished:
            done = True
            reward = self.unfinished_penalty
        
        return self._get_obs(), reward, done, {}

    def gen_action_mask(self): ################################################################
        action_mask_ba = bitarray.util.zeros(self.vocab_size)
        cur_node = None if len(self.tree_stack) == 0 else self.tree_stack[-1]
        action_mask_ba[self.EOS] = cur_node is None and not self.is_finished and len(self.roots) == self.num_outputs
        action_mask_ba[self.PAD] = self.is_finished
        if not self.is_finished and not self.gen_eos and \
                (self.max_inference_reward is None or self.cumulative_reward >= self.max_inference_reward):
            # insert the node into the tree
            # var = -2 means not determined (check all possibilities)
            node = NodeWithInv(parent=Node(var=-2, left=None, right=None), inverted=False)
            if cur_node is None:
                is_root = True
            else:
                is_root = False
                if cur_node.left is None:
                    cur_node.left = node
                else:
                    cur_node.right = node

            # consider error rate from completed roots
            current_tt_conflict = [bitarray.util.zeros(len(self.tts_bitarray_compressed[self.cur_root_id])) for _ in range(len(self.current_outputs_tt.keys()))]
            for i, root_id in enumerate(self.current_outputs_tt.keys()):
                current_tt_conflict[i] = (self.current_outputs_tt[root_id] ^ self.tts_bitarray_compressed[i])

            # print(f"current_tt_conflict={current_tt_conflict}")
            has_conflict_ba, completeness_ba = check_conflict(self.tree_stack, self.tts_bitarray_compressed[self.cur_root_id], ####################check conflict
                                                              self.input_tt_bitarray_compressed, current_tt_conflict, self.tt_cache_bitarray_compressed, tolerance=self.error_rate_threshold)
            value_action_mask_ba = ~has_conflict_ba
            # print(f"value_action_mask_ba={value_action_mask_ba}")
            action_mask_ba[2: 2 + len(value_action_mask_ba)] = value_action_mask_ba             # PAD: 0, EOS:1
            if self.and_always_available:
                action_mask_ba[2 + self.num_inputs * 2: 4 + self.num_inputs * 2] = bitarray.bitarray('11')
            else:
                action_mask_ba[2 + self.num_inputs * 2: 4 + self.num_inputs * 2] = bitarray.bitarray('00') \
                    if (value_action_mask_ba & completeness_ba).any() or len(self.tree_stack) >= self.max_inference_tree_depth - 2 \
                    else bitarray.bitarray('11')
            action_mask_ba[4 + self.num_inputs * 2 : 6 + self.num_inputs * 2] = value_action_mask_ba[-2:]
            # remove the node from the tree
            if not is_root:
                if cur_node.right is None:
                    cur_node.left = None
                else:
                    cur_node.right = None
        if not action_mask_ba.any():
            action_mask_ba[self.EOS] = True
        return np.array(action_mask_ba.tolist(), dtype=bool)