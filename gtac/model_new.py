from __future__ import annotations

import copy
import os
import pickle
import types
from collections import deque, Counter

import bitarray
import bitarray.util
import numpy as np
import scipy.special as special
import npn
import time
import json
import sys
import tracemalloc
import tensorflow as tf

import tf_keras as keras
import keras.backend as K
import gc

# replace your path
sys.path.append('./')

from tensorflow_models import nlp
from gtac.tensorflow_transformer import Seq2SeqTransformer, CustomSchedule, masked_loss, masked_accuracy
from gtac.utils import *
from gtac.encoding import node_to_int, int_to_node, encode_aig, stack_to_encoding, deref_node
from gtac.environment import LogicNetworkEnv
from gtac.mcts import MCTSNode, ucb


class CircuitTransformer:
    def __init__(self,
                 num_inputs=8,
                 embedding_width=512,
                 num_layers=12,
                 num_attention_heads=8,
                 intermediate_size=2048,
                 max_tree_depth=32,
                 max_seq_length=200,
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
                 w_delay=0
                 ):
        self.num_inputs = num_inputs
        self.vocab_size = 2 + 2 * self.num_inputs + 2 + 2
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
        self.add_action_mask_to_input = add_action_mask_to_input
        self.policy_temperature_in_mcts = policy_temperature_in_mcts
        self.freeze_encoder = False         # Modified: Freeze encoder layers for finetuning.
        self.constant_0_id = num_inputs * 2 + 4
        self.constant_1_id = num_inputs * 2 + 5
        self.w_gate = w_gate
        self.w_delay = w_delay
        # https://www.tensorflow.org/guide/mixed_precision
        if mixed_precision:
            keras.mixed_precision.set_global_policy('mixed_float16')
        self._transformer = self._get_tf_transformer()

        if self.ckpt_path is not None:
            # self.build_model()  # 先构建模型变量
            self.load(self.ckpt_path)  # 再加载权重

        @tf.function(reduce_retracing=True)
        def _transformer_inference_graph(self, inputs, return_kv_cache=False, return_last_token=False, return_value=False):
            policy, cache = self._transformer(inputs, return_kv_cache=return_kv_cache, return_last_token=return_last_token, return_value=return_value)
            return policy, cache

        def _transformer_inference(self, inputs, return_kv_cache=False, return_last_token=False, return_value=False):
            policy, cache = _transformer_inference_graph(self, inputs, return_kv_cache=return_kv_cache, return_last_token=return_last_token, return_value=return_value)
            return policy.numpy(), cache

        self._transformer_inference = types.MethodType(_transformer_inference, self)
        self._transformer.return_cache = True
        self.use_kv_cache = True
        self.input_tt = compute_input_tt(self.num_inputs)

    def build_model(self):
        """显式构建模型变量"""
        dummy_inputs = {
            'inputs': tf.zeros((1, self.max_seq_length), dtype=tf.int32),
            'enc_pos_encoding': tf.zeros((1, self.max_seq_length, self.max_tree_depth*2)),
            'targets': tf.zeros((1, self.max_seq_length), dtype=tf.int32),
            'dec_pos_encoding': tf.zeros((1, self.max_seq_length, self.max_tree_depth*2)),
            'enc_action_mask': tf.zeros((1, self.max_seq_length, self.vocab_size), dtype=tf.bool),
            'dec_action_mask': tf.zeros((1, self.max_seq_length, self.vocab_size), dtype=tf.bool)
        }
        _ = self._transformer(dummy_inputs)  # 触发变量创建

    def freeze_layers(self, freeze_encoder=True):
        """Freeze encoder layers, only train decoder layers"""
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            # freeze all encoder layers
            self._transformer.encoder_layer.trainable = False
            
            # freeze encoder embedding layers
            self._transformer.enc_embedding_lookup.trainable = False
            
            # freeze positional encoding layers
            self._transformer.position_embedding.trainable = False
            self._transformer.tree_position_embedding.trainable = False
            
            # unfreeze decoder layers
            self._transformer.decoder_layer.trainable = True
            self._transformer.dec_embedding_lookup.trainable = True

    def _get_tf_transformer(self):
        transformer = Seq2SeqTransformer(
            enc_vocab_size=self.vocab_size,
            dec_vocab_size=self.vocab_size,
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

        if self.freeze_encoder:
            transformer.encoder_layer.trainable = False
            transformer.enc_embedding_lookup.trainable = False
            transformer.position_embedding.trainable = False
            transformer.tree_position_embedding.trainable = False
        return transformer

    def _copy_env(self, env: LogicNetworkEnv | list[LogicNetworkEnv]):
        if isinstance(env, list):
            return [self._copy_env(e) for e in env]
        else:
            context_hash, tts_bitarray, input_tt_bitarray, input_tt_bitarray_compressed, ffw = \
                env.context_hash, env.tts_bitarray, env.input_tt_bitarray, env.input_tt_bitarray_compressed, env.ffw
            env.context_hash, env.tts, env.input_tt, env.input_tt_bitarray_compressed, env.ffw = None, None, None, None, None
            res = copy.deepcopy(env)
            env.context_hash, env.tts_bitarray, env.input_tt_bitarray, env.input_tt_bitarray_compressed, env.ffw = \
                context_hash, tts_bitarray, input_tt_bitarray, input_tt_bitarray_compressed, ffw
            res.context_hash, res.tts_bitarray, res.input_tt_bitarray, res.input_tt_bitarray_compressed, res.ffw = \
                context_hash, tts_bitarray, input_tt_bitarray, input_tt_bitarray_compressed, ffw
            return res

    def _batch_estimate_policy(self, envs: list[LogicNetworkEnv], src_tokens, src_pos_enc, src_action_mask, action_masks, cache):
        start_time = time.time()
        indices = [len(env.tokens) for env in envs]
        max_token_length = np.max(indices) + 1
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
                            np.ones((max_token_length - len(env.action_masks), self.vocab_size), dtype=bool)], axis=0)
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

    def _batch_estimate_v_value_via_simulation_kvcache(self, envs: list[LogicNetworkEnv], src_tokens, src_pos_enc, src_action_mask,
                                                       max_inference_seq_length, cache=None, num_leaf_parallelization=1):
        total_time = time.time()
        envs = self._copy_env(envs)
        copy_time = time.time() - total_time
        batch_size = len(envs)
        v = np.zeros(len(envs), dtype=int)
        inputs = {'inputs': src_tokens, 'enc_pos_encoding': src_pos_enc, 'enc_action_mask': src_action_mask, 'cache': cache}
        targets = np.zeros((batch_size, 1), dtype=np.int32)
        dec_pos_encoding = np.zeros((batch_size, 1, self.max_tree_depth * 2), dtype=np.float32)
        if cache is not None and num_leaf_parallelization > 1:
            cache['encoder_outputs'] = np.concatenate([cache['encoder_outputs']] * num_leaf_parallelization, axis=0)
        transformer_time = 0.
        action_mask_time = 0.
        step_time = 0.
        for i in range(max_inference_seq_length):
            # print(i)
            inputs['targets'], inputs['dec_pos_encoding'] = targets, dec_pos_encoding

            # generate action mask
            # action_masks = np.stack([e.gen_action_mask() for e in envs], axis=0)
            start_time = time.time()
            action_masks = [e.action_masks[i] for e in envs]# [e.gen_action_mask() if i == e.t else np.ones(self.vocab_size, dtype=bool) for e in envs]
            action_masks = np.stack(action_masks, axis=0)
            inputs['dec_action_mask'] = np.expand_dims(action_masks, axis=1)
            action_mask_time += time.time() - start_time

            start_time = time.time()
            policy, cache = self._transformer_inference(inputs, return_kv_cache=True, return_last_token=True)
            inputs['cache'] = cache
            transformer_time += time.time() - start_time

            start_time = time.time()
            tokens = np.argmax(policy, axis=1)
            tokens = [token if i == e.t else e.tokens[i] for token, e in zip(tokens, envs)]

            rewards, dones = zip(*[e.step(token) if i == e.t else (0, False) for token, e in zip(tokens, envs)])
            step_time += time.time() - start_time

            dec_pos_encoding = [e.positional_encodings[i] for e in envs]

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
        if self.verbose > 0:
            print("simulation time: total %f, copy %.2f, step %.2f, transformer %.2f, action mask %.2f; #(steps) = %d" %
                  (time.time() - total_time, copy_time, step_time, transformer_time, action_mask_time, i))
        return v.tolist(), [e.success for e in envs], cache

    def _batch_MCTS_policy_with_leaf_parallelization(self, envs: list[LogicNetworkEnv], num_leaf_parallelizations=8, num_playouts=100, max_inference_seq_length=None,
                           src_tokens=None, src_pos_enc=None, src_action_mask=None,
                           roots=None, orig_aigs_size=None, puct_explore_ratio=1.):
        def update_done_node(node: MCTSNode, root_id):
            if node is not roots[root_id]:  # update done info to avoid unnecessary search
                done_node = node.parent
                while np.all([c.info['done'] and c.explored for c in done_node.children]):
                    if self.verbose > 1:
                        print("node done:", done_node, done_node.children)
                    done_node.info['done'] = True
                    done_node = done_node.parent
                    if done_node is None:
                        break

        if max_inference_seq_length is None:
            max_inference_seq_length = self.max_seq_length

        envs = self._copy_env(envs)
        for env in envs:
            env.max_length = max_inference_seq_length

        src_tokens_parallel = np.concatenate([src_tokens] * num_leaf_parallelizations, axis=0)
        src_pos_enc_parallel = np.concatenate([src_pos_enc] * num_leaf_parallelizations, axis=0)
        src_action_mask_parallel = np.concatenate([src_action_mask] * num_leaf_parallelizations, axis=0)

        num_envs = len(envs)
        t = 0
        cache = None
        virtual_loss = -50
        if roots is None:
            roots = [MCTSNode(None, t, None, info={'env': env, 'reward': None, 'done': None, 'rollout_success': None},
                              puct_explore_ratio=puct_explore_ratio)for env in envs]  # info = (state, reward, done)

        for i in range(num_playouts):
            playout_time = time.time()
            nodes = [copy.copy(roots) for _ in range(num_leaf_parallelizations)]
            sum_rewards = [[root.info['env'].cumulative_reward for root in roots] for _ in range(num_leaf_parallelizations)]

            # selection
            expansion_time = time.time()
            largest_node_list = [[] for _ in range(num_leaf_parallelizations)]
            for l in range(num_leaf_parallelizations):
                largest_node_set = set(range(num_envs))
                while len(largest_node_set) > 0:
                    for j in largest_node_set:
                        node = roots[j]
                        sum_rewards[l][j] = roots[j].info['env'].cumulative_reward
                        while node.children:
                            not_done_children = [(c_id, c) for c_id, c in enumerate(node.children) if
                                                 not c.info['done'] or not c.explored]
                            if len(not_done_children) == 0:
                                # assert node is root
                                if self.verbose > 1:
                                    print(i, nodes[l][j], "all the children are done and explored!")
                                break
                            select_id = np.argmax([c.puct for c_id, c in not_done_children])
                            node.visits += 1
                            node.total_value += virtual_loss
                            node = node.children[not_done_children[select_id][0]]
                            sum_rewards[l][j] += node.info['reward']
                        nodes[l][j] = node

                    # expansion
                    action_masks = np.stack([node.info['env'].action_masks[-1] for node in nodes[l]], axis=0)
                    policies, cache = self._batch_estimate_policy([node.info['env'] for node in nodes[l]], src_tokens, src_pos_enc, src_action_mask,
                                                                  action_masks, cache)

                    new_largest_node_set = largest_node_set.copy()
                    for j, node in enumerate(nodes[l]):
                        if j in largest_node_set:
                            if node.explored:
                                new_action_list = np.where(action_masks[j])[0]
                                for token in new_action_list:
                                    e = self._copy_env(node.info['env'])
                                    reward, done = e.step(token)
                                    child = MCTSNode(node, node.t + 1, token, prob=policies[j][token],
                                                     info={'env': e, 'reward': reward, 'done': done, 'rollout_success': None},
                                                     puct_explore_ratio=puct_explore_ratio)
                                    node.children.append(child)
                                sorted_ids = np.argsort([c.puct for c in node.children])
                                node.visits += 1
                                node.total_value += virtual_loss
                                if len(sorted_ids) > 1:
                                    select_id, largest_id = sorted_ids[-2:]
                                    largest_node = node.children[largest_id]
                                    largest_node.explored = True
                                    largest_node.visits += 1
                                    largest_node.total_value += virtual_loss
                                    largest_node.v = virtual_loss
                                    largest_node.info['rollout_success'] = True
                                    largest_node.info['raw_value'] = virtual_loss
                                    largest_node_list[l].append((largest_node, node))
                                else:
                                    select_id = sorted_ids[-1]
                                    new_largest_node_set.remove(j)
                                    node = node.children[select_id]
                                    sum_rewards[l][j] += node.info['reward']
                            else:
                                node.explored = True
                                new_largest_node_set.remove(j)

                            if j not in new_largest_node_set:
                                node.visits += 1
                                node.total_value += virtual_loss
                                node.v = virtual_loss
                                node.info['rollout_success'] = True
                                node.info['raw_value'] = virtual_loss

                                nodes[l][j] = node
                    largest_node_set = new_largest_node_set

            expansion_time = time.time() - expansion_time

            # rollout (simulation)
            simulation_time = time.time()
            vs, rollout_success, cache = self._batch_estimate_v_value_via_simulation_kvcache(
                sum([[node.info['env'] for node in nodes_i] for nodes_i in nodes], start=[]),
                src_tokens_parallel, src_pos_enc_parallel, src_action_mask_parallel,
                max_inference_seq_length, cache, num_leaf_parallelizations)

            vs = np.array(vs).reshape((num_leaf_parallelizations, num_envs))
            rollout_success = np.array(rollout_success).reshape((num_leaf_parallelizations, num_envs))
            simulation_time = time.time() - simulation_time

            paths = [[[] for _ in range(num_envs)] for _ in range(num_leaf_parallelizations)]
            for l in range(num_leaf_parallelizations):
                for j, node in enumerate(nodes[l]):
                    if orig_aigs_size is not None and (not rollout_success[l][j] or vs[l][j] < -orig_aigs_size[j] - sum_rewards[l][j]):
                        node.v = -orig_aigs_size[j] - sum_rewards[l][j]
                    else:
                        node.v = vs[l][j]
                    sum_rewards[l][j] += node.v
                    node.explored = True
                    node.sum_reward = sum_rewards[l][j]
                    node.info['rollout_success'] = rollout_success[l][j]
                    node.info['raw_value'] = vs[l][j]
                    update_done_node(node, j)

                # backpropagate
                for j, node in enumerate(nodes[l]):
                    while node is not None:
                        if node.action is not None:
                            paths[l][j].append(node.action)
                        node.total_value += sum_rewards[l][j] - virtual_loss
                        if self.verbose > 1:
                            if sum_rewards[l][j] > node.max_value and node.max_value != MCTSNode.INIT_MAX_VALUE and node is roots[j]:
                                print("root %d node max value updated, from %d to %d!" % (j, node.max_value, sum_rewards[l][j]))
                        node.max_value = max(node.max_value, sum_rewards[l][j])
                        node = node.parent

                for largest_node, node in largest_node_list[l]:
                    largest_node.explored = True  # so #(explored nodes) >= num_rollouts
                    largest_node.v = node.v - largest_node.info['reward']
                    largest_node.max_value = node.max_value
                    largest_node.sum_reward = node.sum_reward
                    largest_node.info['rollout_success'] = node.info['rollout_success']
                    largest_node.info['raw_value'] = node.info['raw_value'] - largest_node.info['reward']
                    largest_node.info['derived_from_parent'] = True
                    while largest_node is not None:
                        largest_node.total_value += node.sum_reward - virtual_loss   # node.v
                        largest_node = largest_node.parent

            if self.verbose > 0:
                for l in range(num_leaf_parallelizations):
                    print([(p, s) for p, s in zip(paths[l], sum_rewards[l])])
                print([r.max_value for r in roots])
                print("playout %d - total time %.2f, expansion time %.2f, simulation time %.2f" %
                      (i, time.time() - playout_time, expansion_time, simulation_time))

        best_action_seqs = []
        best_child_seqs = []
        for j, root in enumerate(roots):
            best_action_seq, best_child_seq = [], []
            while root.children:
                action_list = [(c, c.max_value, c.puct) for c in root.children]
                action_list.sort(key=lambda x: (x[1], x[2]), reverse=True)
                root = action_list[0][0]
                if not root.explored:
                    break
                best_action_seq.append(root.action)
                best_child_seq.append(root)
            best_action_seqs.append(best_action_seq)
            best_child_seqs.append(best_child_seq)
        return best_action_seqs, [b[0] for b in best_child_seqs]

    def _encode_postprocess(self, seq_enc: list[int], pos_enc: list[int]):
        seq_enc.append(self.eos_id)
        if self.verbose > 0 and len(seq_enc) > self.max_seq_length:
            print("Warning: seq_enc length %d > max seq length (%d)" % (len(seq_enc), self.max_seq_length))
        seq_enc, pos_enc = seq_enc[:self.max_seq_length], pos_enc[:self.max_seq_length]
        pos_enc = np.stack(
            [list(reversed(npn.int_to_tt(pos_enc_i, base_2_log(self.max_tree_depth) + 1))) for pos_enc_i in pos_enc],
            axis=0)  # 2 ^ 6 = 64 == max_tree_depth * 2
        seq_enc = np.array(seq_enc + [0] * (self.max_seq_length - len(seq_enc)), dtype=np.int32)
        pos_enc = np.concatenate([pos_enc, np.zeros((self.max_seq_length - len(pos_enc), self.max_tree_depth * 2))],
                                 axis=0,
                                 dtype=np.float32)
        return seq_enc, pos_enc

    def load(self, ckpt_path):
        status = self._transformer.load_weights(ckpt_path)
        status.expect_partial()         # ignore warnings
        self.ckpt_path = ckpt_path

    def load_from_hf(self, hf_model_name="deepsyn_reinforced"):
        from huggingface_hub import hf_hub_download
        index_path = hf_hub_download(repo_id="snowkylin/circuit-transformer", filename="%s.index" % hf_model_name)
        data_path = hf_hub_download(repo_id="snowkylin/circuit-transformer", filename="%s.data-00000-of-00001" % hf_model_name)
        ckpt_path = index_path[:-6]
        print("checkpoint downloaded to %s" % ckpt_path)
        self._transformer.load_weights(ckpt_path)
        self.ckpt_path = index_path

    def generate_action_masks(self, tts, input_tt, care_set_tts, seq_enc, use_controllability_dont_care, tts_compressed=None, ffw = None, n = None, fileName=None):
        env = LogicNetworkEnv(tts,
                              self.num_inputs,
                              init_care_set_tt=care_set_tts,
                              ffw=ffw,
                              input_tt=input_tt,
                              max_length=self.max_seq_length,
                              use_controllability_dont_cares=use_controllability_dont_care,
                              tts_compressed=tts_compressed,
                              w_gate=self.w_gate,
                              w_delay=self.w_delay, 
                              and_always_available=True)
        action_masks = []
        flag = 1
        # error_file = "./datasets/t/IWLS_FFWs_app_0.1/0xf000000000000008000000000000000800000000000000000000000000000000_0xffffffffffffc000f00000000000000000000000000000000000000000000000.json"
        # if fileName == error_file:
        #     print(seq_enc)
            
        for token in seq_enc[:self.max_seq_length]:
            # if fileName == error_file and n == '2':
            #     print(f"Valid: {token}")
            action_mask = env.action_masks[-1]          # dec_env.gen_action_mask()
            if not action_mask[token]:                  # Modified: Jump conflict circuit data.
                # if fileName == error_file and n == '2':
                # print(f"Invalid: {token}")
                # print("Cannot pass check_conflict. Jump anyway.")
                flag = 0
                # break
            action_masks.append(action_mask)
            env.step(token)
        if not flag:
            # print(fileName)
            return None
        assert action_mask[token]                   # Problem: current action mask will contadict to the approximate dataset
           
        if len(seq_enc) < self.max_seq_length:
            action_masks.append(env.action_masks[-1])
        action_masks = np.stack(action_masks)
        if len(action_masks) < self.max_seq_length:
            action_masks_padding = np.zeros((self.max_seq_length - len(action_masks), self.vocab_size),
                                            dtype=bool)
            action_masks_padding[:, 0] = True
            action_masks = np.concatenate([action_masks, action_masks_padding], axis=0)
        return action_masks

    def load_and_encode(self, filename):
        with open(filename, 'r') as f:
            roots_aiger, num_ands, opt_roots_aiger, opt_num_ands = json.load(f)
        roots, info = read_aiger(aiger_str=roots_aiger)
        opt_roots, _ = read_aiger(aiger_str=opt_roots_aiger)
        num_inputs, num_outputs = info[1], info[3]

        phase = np.random.rand(num_inputs) < 0.5
        perm = np.random.permutation(num_inputs)
        output_invs = np.random.rand(num_outputs) < 0.5
        roots = npn_transform(roots, phase, perm, output_invs)
        opt_roots = npn_transform(opt_roots, phase, perm, output_invs)

        seq_enc, pos_enc = self._encode_postprocess(*encode_aig(roots, num_inputs))
        opt_seq_enc, opt_pos_enc = encode_aig(opt_roots, num_inputs)
        tts = compute_tts(roots, input_tt=self.input_tt)
        enc_action_masks = self.generate_action_masks(tts, self.input_tt, None, seq_enc, True, tts, n="1", fileName=filename)
        if enc_action_masks is None:                # Modified: Jump conflict circuit data.
            return seq_enc, pos_enc, opt_seq_enc, opt_pos_enc, None, None
        dec_action_masks = self.generate_action_masks(tts, self.input_tt, None, opt_seq_enc, True, tts, n="2", fileName=filename)
        # if filename == "./datasets/t/IWLS_FFWs_app_0.1/0xf000000000000008000000000000000800000000000000000000000000000000_0xffffffffffffc000f00000000000000000000000000000000000000000000000.json":
        #     print(seq_enc)
        #     print(encode_aig(roots, num_inputs)[1])
        #     print(self._encode_postprocess(opt_seq_enc, opt_pos_enc)[0])
        #     print(opt_pos_enc)
        if dec_action_masks is None:                # Modified: Jump conflict circuit data.
            return seq_enc, pos_enc, opt_seq_enc, opt_pos_enc, None, None
        opt_seq_enc, opt_pos_enc = self._encode_postprocess(opt_seq_enc, opt_pos_enc)
        return seq_enc, pos_enc, opt_seq_enc, opt_pos_enc, enc_action_masks, dec_action_masks

    def load_and_encode_formatted(self, filename):
        seq_enc, pos_enc, opt_seq_enc, opt_pos_enc, enc_action_mask, dec_action_mask = self.load_and_encode(filename)
        if enc_action_mask is None or dec_action_mask is None:          # Modified: Jump conflict circuit data.
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
              freeze_layers=False
              ):
        train_data_dir = train_data_dir + ("/" if train_data_dir[-1] != "/" else "")
        
        if ckpt_save_path is None:
            print("WARNING: ckpt_save_path is not specified, the trained model will not be saved during training!")
        else:
            ckpt_save_path = ckpt_save_path + ("/" if ckpt_save_path[-1] != "/" else "")

            if not os.path.exists(ckpt_save_path):
                os.mkdir(ckpt_save_path)

        if freeze_layers:
            self.freeze_layers(freeze_encoder=True)

        train_files = os.listdir(train_data_dir)
        print("%d training files listed" % len(train_files))
        train_files.sort()
        print("training files sorted")
        np.random.seed(0)
        np.random.shuffle(train_files)
        print("training files shuffled")

        self._transformer.return_cache = False

        if excluded_files is not None:
            print("excluded files is not None, filtering training files...")
            excluded_files = set(excluded_files)
            new_train_files = []
            for file in train_files:
                if file not in excluded_files:
                    new_train_files.append(file)
            print("training files filtered, from %d to %d" % (len(train_files), len(new_train_files)))
            train_files = new_train_files

        train_files = [(train_data_dir + file) for file in train_files]
        self_copied = copy.copy(self)
        self_copied._transformer = None
        self_copied._transformer_inference = None

        mp_dataset = MPDataset(train_files, self_copied.load_and_encode_formatted, validation_split=validation_split, num_processes=8)

        output_signature = (
            {
                'inputs': tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32),
                'enc_pos_encoding': tf.TensorSpec(shape=(self.max_seq_length, self.max_tree_depth * 2),
                                                  dtype=tf.float32),
                'targets': tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32), #######################################输入了target
                'dec_pos_encoding': tf.TensorSpec(shape=(self.max_seq_length, self.max_tree_depth * 2),
                                                  dtype=tf.float32),
                'enc_action_mask': tf.TensorSpec(shape=(self.max_seq_length, self.vocab_size),
                                                 dtype=tf.bool),
                'dec_action_mask': tf.TensorSpec(shape=(self.max_seq_length, self.vocab_size),
                                                  dtype=tf.bool)
            }, tf.TensorSpec(shape=(self.max_seq_length,), dtype=tf.int32)
        )
        print("正在创建TensorFlow数据集...")
        train_dataset = tf.data.Dataset.from_generator(mp_dataset.train_generator,
                                                       output_signature=output_signature) \
            .batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset = tf.data.Dataset.from_generator(mp_dataset.validation_generator,
                                                            output_signature=output_signature) \
            .batch(batch_size).prefetch(tf.data.AUTOTUNE)
        print("数据集创建完成")
        
        if profile:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            summary_writer = tf.summary.create_file_writer(log_dir)

        if distributed:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                transformer = self._get_tf_transformer()
                if self.ckpt_path is not None:
                    transformer.load_weights(self.ckpt_path)
                # learning_rate = CustomSchedule(self.embedding_width)
                optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
                # optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
                
                transformer.compile(
                    optimizer=optimizer,
                    loss=masked_loss,
                    metrics=[masked_accuracy]
                )
        else:
            transformer = self._transformer
            # learning_rate = CustomSchedule(self.embedding_width)
            optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            # optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            transformer.compile(
                optimizer=optimizer,
                loss=masked_loss,
                metrics=[masked_accuracy],
            )

        class LogCallback(keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                print(f"Batch {batch} finished with logs: {logs}")
                # step = transformer.optimizer.iterations.numpy()
                with summary_writer.as_default():
                    # 写入损失和准确率
                    tf.summary.scalar('loss', logs['loss'], step=batch)
                    tf.summary.scalar('accuracy', logs['accuracy'], step=batch)
                pass

            def on_epoch_end(self, epoch, logs=None):
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                for stat in top_stats:
                    print(stat)

        log = LogCallback()

        callbacks = []
        if ckpt_save_path is not None:
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=ckpt_save_path + 'model-{epoch:04d}',
                save_weights_only=True,
                save_freq=(len(mp_dataset) * (epochs - initial_epoch) // batch_size) if latest_ckpt_only else 'epoch') # type: ignore
            callbacks.append(checkpoint)

        print("开始训练，准备调用 fit() 方法")
        transformer.fit(train_dataset,
                        initial_epoch=initial_epoch,
                        epochs=epochs,
                        validation_data=validation_dataset,
                        callbacks=callbacks,
                        verbose=1)
        mp_dataset.process.terminate()
        print("training finished")

        if profile:
            # 简单的profiling，不使用trace_export
            print("Profiling completed. Check log directory for TensorBoard logs.")

        self._transformer.return_cache = True

    def train_ppo(self, 
                  train_data_dir,
                  ckpt_save_path = None,
                  epochs=1000,
                  steps_per_epoch=10,
                  batch_size=64,
                  gamma=0.99,
                  clip_ratio=0.2,
                  policy_lr=1e-4,
                  value_lr=1e-3,
                  freeze_layers=False,
                  ppo_train_epoch=2,
                  target_kl=0.01):

        train_data_dir = train_data_dir + ("/" if train_data_dir[-1] != "/" else "")
        log_file = open("log/ppo_train_log.txt", "a") 
        if ckpt_save_path is None:
            print("WARNING: ckpt_save_path is not specified, the trained model will not be saved during training!")
        else:
            ckpt_save_path = ckpt_save_path + ("/" if ckpt_save_path[-1] != "/" else "")

            if not os.path.exists(ckpt_save_path):
                os.mkdir(ckpt_save_path)

        if freeze_layers:
            self.freeze_layers(freeze_encoder=True)

        """PPO训练循环"""
        # 初始化优化器
        policy_optimizer = keras.optimizers.AdamW(learning_rate=policy_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0)
        value_optimizer = keras.optimizers.Adam(learning_rate=value_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0)
        
        # 获取所有电路文件
        circuit_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)]
        if not circuit_files:
            raise ValueError(f"No circuit files found in directory: {train_data_dir}")
        
        # 创建电路迭代器
        circuit_iterator = self._circuit_iterator(circuit_files)
        
        # 训练循环
        for epoch in range(epochs):
            # 收集轨迹数据
            trajectories = []
            for step in range(steps_per_epoch):
                current_circuit = next(circuit_iterator)
                
                # 创建当前电路的环境
                env = self._create_circuit_env(current_circuit)
                state = env._get_obs()
                done = False
                episode_data = []
                
                while not done:
                    # 准备模型输入
                    inputs = self._prepare_model_input(state)
                    
                    # 使用当前策略选择动作
                    logits, value = self._transformer_inference(inputs, return_last_token=True)
                    action, log_prob = self._sample_action(logits[0], state['action_mask'])
                    
                    # 执行动作
                    next_state, reward, done = self._batch_step([env], [action])
                    next_state = next_state[0]
                    reward = reward[0]
                    done = done[0]
                    # 存储转换
                    episode_data.append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'log_prob': log_prob,
                        'value': value[0][0],
                        'done': done
                    })
                    
                    state = next_state
                
                # 添加到轨迹数据
                trajectories.append(episode_data)
            
            # 处理轨迹数据
            states, actions, old_log_probs, returns, advantages = self._process_trajectories(
                trajectories, gamma
            )
            print(f"trajectories length = {len(trajectories)}")
            start_time = time.time()
            # 更新策略
            policy_loss = self._update_policy(
                states, actions, old_log_probs, advantages, 
                policy_optimizer, clip_ratio, target_kl, batch_size, ppo_train_epoch
            )
            step1_time = time.time()
            print(f"Policy update total time: {step1_time - start_time:.4f} seconds")
            # 更新价值函数
            value_loss = self._update_value_function(
                states, returns, value_optimizer, batch_size, ppo_train_epoch
            )
            step2_time = time.time()
            print(f"Value update total time: {step2_time - step1_time:.4f} seconds")

            # 打印进度
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Policy Loss: {policy_loss:.4f} | "
                f"Value Loss: {value_loss:.4f} | "
                f"Avg Return: {np.mean(returns):.2f}")
            log_msg = (f"Epoch {epoch+1}/{epochs} | "
                        f"Policy Loss: {policy_loss:.4f} | "
                        f"Value Loss: {value_loss:.4f} | "
                        f"Avg Return: {np.mean(returns):.2f}")
            print(log_msg)
            log_file.write(log_msg + "\n")
            log_file.flush()  # 保证及时写入磁盘

            # 定期保存模型
            if (epoch + 1) % 5 == 0 and ckpt_save_path is not None:
                save_path = os.path.join(ckpt_save_path, f"model-{epoch+1:04d}")
                self._transformer.save_weights(save_path)
                print(f"Model weights saved at {save_path}")

    def _circuit_iterator(self, circuit_files):
        """创建电路文件的无限迭代器"""
        while True:
            # 随机打乱电路文件顺序
            np.random.shuffle(circuit_files)
            for circuit_file in circuit_files:
                yield circuit_file
    
    def _create_circuit_env(self, circuit_file):
        """为给定电路文件创建环境"""
        # 读取电路
        with open(circuit_file, 'r') as f:
            roots_aiger, num_ands, opt_roots_aiger, opt_num_ands = json.load(f)
        
        roots, info = read_aiger(aiger_str=roots_aiger)
        num_inputs, num_outputs = info[1], info[3]
        
        # 计算真值表
        tts = compute_tts(roots, num_inputs=num_inputs)
        
        # 创建环境
        return LogicNetworkEnv(
            tts=tts,
            num_inputs=num_inputs,
            max_length=self.max_seq_length,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            max_tree_depth=self.max_tree_depth,
            max_inference_tree_depth=16,
            use_controllability_dont_cares=True,
            verbose=0
        )
    
    def _prepare_model_input(self, state):
        """准备Transformer的输入格式, 确保正确的形状"""
        # 确保 inputs 是二维的 [1, sequence_length]
        tokens = state['tokens']
        if len(tokens.shape) == 1:
            inputs = tf.expand_dims(tokens, axis=0)  # [1, seq_len]
        else:
            inputs = tokens  # 已经是正确的形状
        
        # 确保 enc_pos_encoding 是三维的 [batch, sequence_length, features]
        pos_enc = state['positional_encodings']
        if len(pos_enc.shape) == 2:  # [seq_len, features]
            enc_pos_encoding = tf.expand_dims(pos_enc, axis=0)  # [1, seq_len, features]
        elif len(pos_enc.shape) == 3:  # [1, seq_len, features]
            enc_pos_encoding = pos_enc  # 已经是正确的形状
        elif len(pos_enc.shape) == 4:  # [1, seq_len, 1, features] - 需要压缩
            enc_pos_encoding = tf.squeeze(pos_enc, axis=2)  # [1, seq_len, features]
        else:
            enc_pos_encoding = pos_enc  # 已经是正确的形状
        
        # 确保 enc_action_mask 是四维的 [batch, sequence_length, 1, vocab_size]
        action_mask = state['action_mask']
        if len(action_mask.shape) == 1:  # [vocab_size]
            # 创建与序列长度匹配的动作掩码
            seq_len = tf.shape(inputs)[1]
            enc_action_mask = tf.tile(
                tf.expand_dims(tf.expand_dims(tf.expand_dims(action_mask, axis=0), axis=0), axis=0),
                [1, seq_len, 1, 1]
            )  # [1, seq_len, 1, vocab_size]
        elif len(action_mask.shape) == 2:  # [seq_len, vocab_size]
            enc_action_mask = tf.expand_dims(tf.expand_dims(action_mask, axis=0), axis=2)  # [1, seq_len, 1, vocab_size]
        elif len(action_mask.shape) == 3:  # [1, seq_len, vocab_size]
            enc_action_mask = tf.expand_dims(action_mask, axis=2)  # [1, seq_len, 1, vocab_size]
        else:
            enc_action_mask = action_mask  # 已经是正确的形状
        
        # 处理 targets - 使用最后一个token，确保形状为 [batch, 1]
        if tf.shape(inputs)[1] > 0:
            last_token = inputs[0, -1]  # 获取最后一个token
            targets = tf.expand_dims(tf.expand_dims(last_token, axis=0), axis=0)  # [1, 1]
        else:
            targets = tf.zeros((1, 1), dtype=tf.int32)
        
        # 处理 dec_pos_encoding - 使用最后一个位置编码，确保形状为 [batch, 1, features]
        if tf.shape(enc_pos_encoding)[1] > 0:
            last_pos_enc = enc_pos_encoding[0, -1, :]  # 获取最后一个位置编码
            dec_pos_encoding = tf.expand_dims(tf.expand_dims(last_pos_enc, axis=0), axis=0)  # [1, 1, features]
        else:
            dec_pos_encoding = tf.zeros((1, 1, self.max_tree_depth * 2), dtype=tf.float32)
        
        # 处理 dec_action_mask - 使用当前动作掩码，确保形状为 [batch, 1, 1, vocab_size]
        if len(state['action_mask'].shape) == 1:  # [vocab_size]
            dec_action_mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(state['action_mask'], axis=0), axis=0), axis=0)  # [1, 1, 1, vocab_size]
        else:
            # 如果已经有更高维度，取最后一个时间步
            dec_action_mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(action_mask[-1], axis=0), axis=0), axis=0)  # [1, 1, 1, vocab_size]
        
        return {
            'inputs': inputs,
            'enc_pos_encoding': enc_pos_encoding,
            'enc_action_mask': enc_action_mask,
            'targets': targets,
            'dec_pos_encoding': dec_pos_encoding,
            'dec_action_mask': dec_action_mask
        }

    def _prepare_batch_input(self, states):
        MAX_SEQ_LEN = self.max_seq_length  # 使用你模型中设定的最大序列长度

        inputs = []
        enc_pos_encoding = []
        enc_action_mask = []
        targets = []
        dec_pos_encoding = []
        dec_action_mask = []

        for s in states:
            seq_len = s['tokens'].shape[0]

            # tokens padding
            padded_tokens = np.pad(
                s['tokens'],
                (0, MAX_SEQ_LEN - seq_len),
                constant_values=self.pad_id
            )
            inputs.append(padded_tokens)

            # pos_enc padding
            pos_enc = s['positional_encodings']
            padded_pos_enc = np.pad(
                pos_enc,
                ((0, MAX_SEQ_LEN - seq_len), (0, 0)),
                constant_values=0.0
            )
            enc_pos_encoding.append(padded_pos_enc)

            # action mask padding
            mask = s['action_mask']
            if mask.ndim == 2:  # [seq_len, vocab_size]
                padded_mask = np.pad(
                    mask,
                    ((0, MAX_SEQ_LEN - seq_len), (0, 0)),
                    constant_values=False
                )
            else:  # [vocab_size]
                padded_mask = np.tile(mask, (MAX_SEQ_LEN, 1))
            enc_action_mask.append(np.expand_dims(padded_mask, axis=1))  # [seq_len, 1, vocab]

            # targets - 最后一个 token
            targets.append([s['tokens'][seq_len - 1]])

            # dec_pos_encoding - 最后一个位置编码
            dec_pos_encoding.append([s['positional_encodings'][seq_len - 1]])

            # dec_action_mask - 当前 action mask
            if mask.ndim == 1:  # [vocab]
                dec_action_mask.append([[mask]])
            else:
                dec_action_mask.append([[mask[-1]]])

        # Convert to tensor
        return {
            'inputs': tf.convert_to_tensor(inputs, dtype=tf.int32),  # [B, MAX_SEQ_LEN]
            'enc_pos_encoding': tf.convert_to_tensor(enc_pos_encoding, dtype=tf.float32),  # [B, MAX_SEQ_LEN, D]
            'enc_action_mask': tf.convert_to_tensor(enc_action_mask, dtype=tf.bool),  # [B, MAX_SEQ_LEN, 1, V]
            'targets': tf.convert_to_tensor(targets, dtype=tf.int32),  # [B, 1]
            'dec_pos_encoding': tf.convert_to_tensor(dec_pos_encoding, dtype=tf.float32),  # [B, 1, D]
            'dec_action_mask': tf.convert_to_tensor(dec_action_mask, dtype=tf.bool),  # [B, 1, 1, V]
        }

    
    def _sample_action(self, logits, mask):
        """根据logits和mask采样动作"""
        # 将mask应用于logits（将不可行动作对应的logits设为极小值）
        masked_logits = np.where(mask, logits, np.finfo(np.float32).min)
        
        # 计算softmax
        probs = np.exp(masked_logits - np.max(masked_logits))
        probs /= np.sum(probs)
        probs = np.squeeze(probs)
        # 采样动作
        action = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action])
        
        return action, log_prob
    
    def _process_trajectories(self, trajectories, gamma, lam=0.95):
        """处理轨迹数据，计算回报和优势函数"""
        states = []
        actions = []
        old_log_probs = []
        returns = []
        advantages = []
        
        for episode in trajectories:
            # 提取轨迹数据
            episode_states = [step['state'] for step in episode]
            episode_actions = [step['action'] for step in episode]
            episode_rewards = [step['reward'] for step in episode]
            episode_values = [step['value'] for step in episode]
            episode_log_probs = [step['log_prob'] for step in episode]
            episode_dones = [step['done'] for step in episode]
            
            # 确保数值类型一致
            episode_rewards = [float(r) for r in episode_rewards]
            episode_values = [float(v) for v in episode_values]
            episode_log_probs = [float(lp) for lp in episode_log_probs]
            
            # 计算蒙特卡洛回报
            R = 0.0
            discounted_returns = []
            for r in reversed(episode_rewards):
                R = r + gamma * R
                discounted_returns.insert(0, R)
            
            # 计算广义优势估计 (GAE)
            advantages_ep = []
            last_gae = 0.0
            next_value = 0.0
            next_done = True  # 假设episode结束时done=True
            
            for t in reversed(range(len(episode))):
                if t == len(episode) - 1:
                    next_non_terminal = 1.0 - float(next_done)
                    next_value = next_value
                else:
                    next_non_terminal = 1.0 - float(episode_dones[t+1])
                    next_value = episode_values[t+1]
                
                delta = episode_rewards[t] + gamma * next_value * next_non_terminal - episode_values[t]
                gae = delta + gamma * lam * next_non_terminal * last_gae
                last_gae = gae
                advantages_ep.insert(0, gae)
            
            # 标准化优势函数
            advantages_ep = np.array(advantages_ep, dtype=np.float32)
            if advantages_ep.std() > 0:
                advantages_ep = (advantages_ep - advantages_ep.mean()) / (advantages_ep.std() + 1e-8)
            
            # 添加到结果列表
            states.extend(episode_states)
            actions.extend(episode_actions)
            old_log_probs.extend(episode_log_probs)
            returns.extend(discounted_returns)
            advantages.extend(advantages_ep)
        
        return states, actions, old_log_probs, returns, advantages



    def _update_policy(self, states, actions, old_log_probs, advantages, optimizer, clip_ratio, target_kl, batch_size, ppo_train_epoch):
        """更新策略网络"""
        @tf.function(reduce_retracing=True)
        def train_step(batch_inputs, batch_actions, batch_old_log_probs, batch_advantages, policy_vars):
            with tf.GradientTape() as tape:
                policy_loss, kl, _ = self.policy_forward_pass(
                    batch_inputs, 
                    batch_actions, 
                    batch_old_log_probs, 
                    batch_advantages,
                    clip_ratio
                )
            grads = tape.gradient(policy_loss, policy_vars)
            return grads, policy_loss, kl
        # 创建数据集（仅第一次）
        if not hasattr(self, 'policy_dataset') or self.policy_dataset is None:
            dataset = tf.data.Dataset.from_tensor_slices({
                'tokens': tf.stack([s['tokens'] for s in states]),
                'pos_enc': tf.stack([s['positional_encodings'] for s in states]),
                'action_mask': tf.stack([s['action_mask'] for s in states]),
                'actions': tf.convert_to_tensor(actions, dtype=tf.int32),
                'old_log_probs': tf.convert_to_tensor(old_log_probs, dtype=tf.float32),
                'advantages': tf.convert_to_tensor(advantages, dtype=tf.float32)
            }).shuffle(len(states)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            self.policy_dataset = dataset
        else:
            dataset = self.policy_dataset
        
        # 获取策略网络变量
        policy_vars = [var for var in self._transformer.trainable_variables if 'value_head' not in var.name]
        
        total_policy_loss = 0
        total_kl = 0
        num_batches = 0
        
        for ppo_epoch in range(ppo_train_epoch):
            for batch in dataset:
                # 准备输入
                batch_states = [{
                    'tokens': tokens,
                    'positional_encodings': pos_enc,
                    'action_mask': action_mask
                } for tokens, pos_enc, action_mask in zip(
                    batch['tokens'], batch['pos_enc'], batch['action_mask']
                )]
                inputs = self._prepare_batch_input(batch_states)
                
                # 使用预定义计算图执行训练步骤
                grads, policy_loss, kl = train_step(
                    inputs,
                    batch['actions'],
                    batch['old_log_probs'],
                    batch['advantages'],
                    policy_vars
                )
                
                # 应用梯度
                if grads is not None:
                    grads, _ = tf.clip_by_global_norm(grads, 1.0)
                    optimizer.apply_gradients(zip(grads, policy_vars))
                
                total_policy_loss += policy_loss
                total_kl += kl
                num_batches += 1
                
                # 检查KL散度
                if total_kl / num_batches > 1.5 * target_kl:
                    print(f"Early stopping at KL divergence {total_kl/num_batches:.4f} > {1.5*target_kl:.4f}")
                    break
        
        # 清理资源
        K.clear_session()
        gc.collect()
        
        return total_policy_loss / num_batches

    def _update_value_function(self, states, returns, optimizer, batch_size, ppo_train_epoch):
        """更新价值函数"""
        # 创建数据集 - 使用 TensorFlow 操作
        dataset = tf.data.Dataset.from_tensor_slices({
            'tokens': tf.stack([s['tokens'] for s in states]),
            'pos_enc': tf.stack([s['positional_encodings'] for s in states]),
            'action_mask': tf.stack([s['action_mask'] for s in states]),
            'returns': tf.convert_to_tensor(returns, dtype=tf.float32)
        })
        
        # 应用批处理和预取
        dataset = dataset.shuffle(len(states)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        total_value_loss = 0
        num_batches = 0
        
        # 获取价值网络变量
        value_vars = self._transformer.value_head.trainable_variables
        for epoch in range(ppo_train_epoch):
            for batch in dataset:
                # 使用修改后的 _prepare_batch_input 准备输入
                batch_states = [{
                    'tokens': tokens,
                    'positional_encodings': pos_enc,
                    'action_mask': action_mask
                } for tokens, pos_enc, action_mask in zip(
                    batch['tokens'], 
                    batch['pos_enc'], 
                    batch['action_mask']
                )]
                inputs = self._prepare_batch_input(batch_states)
                
                # 准备参数
                returns_tensor = batch['returns']
                
                with tf.GradientTape() as tape:
                    # 使用图模式计算价值损失
                    value_loss = self.value_forward_pass(inputs, returns_tensor)
                
                # 计算梯度并更新（只更新价值函数相关的权重）
                grads = tape.gradient(value_loss, value_vars)
                if grads is not None:  # 确保梯度存在
                    optimizer.apply_gradients(zip(grads, value_vars))
                
                total_value_loss += value_loss
                num_batches += 1
        
        return total_value_loss / (num_batches * ppo_train_epoch)
    
    def _log_prob(self, logits, actions):
        """计算给定动作的对数概率"""
        # 确保logits是float32类型
        logits = tf.cast(logits, tf.float32)
        
        # 将logits转换为概率分布
        probs = tf.nn.softmax(logits)
        
        # 创建动作的one-hot编码
        actions_one_hot = tf.one_hot(actions, depth=self.vocab_size, dtype=tf.float32)
        
        # 计算对数概率
        return tf.math.log(tf.reduce_sum(probs * actions_one_hot, axis=-1) + 1e-10)

    def _batch_step(self, envs, actions):
        """批量执行环境步骤（向量化操作）"""
        next_states = []
        rewards = []
        dones = []
        
        for env, action in zip(envs, actions):
            next_state, reward, done, _ = env.ppo_step(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

        return next_states, np.array(rewards), np.array(dones)



    @tf.function(reduce_retracing=True)
    def value_forward_pass(self, inputs, returns):
        """价值网络前向传播（图模式）"""
        # 预测价值
        _, values = self._transformer(inputs, training=True)
        
        # 确保数据类型一致 - 将values转换为float32
        values = tf.cast(values, tf.float32)
        returns = tf.cast(returns, tf.float32)
        
        # 计算价值损失
        return tf.reduce_mean(tf.square(returns - values))

    @tf.function(reduce_retracing=True)
    def policy_forward_pass(self, inputs, actions, old_log_probs, advantages, clip_ratio):
        """策略网络前向传播（图模式）"""
        # 获取策略网络输出
        logits, _ = self._transformer(inputs, training=True)
        
        # 确保logits是float32类型
        logits = tf.cast(logits, tf.float32)
        
        # 计算新策略的对数概率
        new_log_probs = self._log_prob(logits, actions)
        
        # 确保所有张量都是float32类型
        new_log_probs = tf.cast(new_log_probs, tf.float32)
        old_log_probs = tf.cast(old_log_probs, tf.float32)
        advantages = tf.cast(advantages, tf.float32)
        
        # 计算PPO损失
        ratio = tf.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        
        # 计算KL散度
        kl = tf.reduce_mean(old_log_probs - new_log_probs)
        
        return policy_loss, kl, new_log_probs

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
                 w_delay=0
                 ):
        if self.ckpt_path is None:
            print("no checkpoint loaded, downloading from https://huggingface.co/snowkylin/circuit-transformer ...")
            self.load_from_hf()

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
                                                  w_delay=w_delay
                                                  )
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
                       w_delay=0
                       ):
        total_time = time.time()
        start_time = time.time()
        encoded_aigs = []
        aigs = aigs.copy()
        if max_mcts_inference_seq_length is None:
            max_mcts_inference_seq_length = max_inference_seq_length
        tts_list = []
        enc_action_masks = []
        orig_aig_size = []
        for i, aig in enumerate(aigs):
            if isinstance(aig, str):
                aigs[i], info = read_aiger(aig)
            if len(aigs[i]) > 2:
                raise OverflowError("the number of outputs for input aig network should be <= 2 "
                                    "as the default model is trained on 8-input, 2-output networks")
            orig_aig_size.append(count_num_ands(aigs[i]))
            seq_enc, pos_enc = encode_aig(aigs[i], self.num_inputs)
            input_tt = self.input_tt if input_tts is None else input_tts[i]
            tts = [compute_tt(root, input_tt=input_tt) for root in aig]
            enc_action_masks.append(self.generate_action_masks(tts,
                                                               input_tt,
                                                               None if care_set_tts is None else care_set_tts[i],
                                                               seq_enc,
                                                               use_controllability_dont_care=use_controllability_dont_cares,
                                                               tts_compressed=None if tts_compressed is None else tts_compressed[i]))
            encoded_aigs.append(self._encode_postprocess(seq_enc, pos_enc))
            tts_list.append(tts)
        enc_action_masks = np.stack(enc_action_masks)
        seq_enc, pos_enc = tuple(map(lambda x: np.stack(x, axis=0), zip(*encoded_aigs)))
        batch_size = len(aigs)

        inputs = {'inputs': seq_enc, 'enc_pos_encoding': pos_enc, 'enc_action_mask': enc_action_masks}
        targets = np.zeros((batch_size, 1), dtype=np.int32)
        dec_pos_encoding = np.zeros((batch_size, 1, self.max_tree_depth * 2), dtype=np.float32)
        envs = [LogicNetworkEnv(
            tts=tts_list[i],
            num_inputs=self.num_inputs,
            context_num_inputs=context_num_inputs,
            input_tt=self.input_tt if input_tts is None else input_tts[i],
            init_care_set_tt=None if care_set_tts is None else care_set_tts[i],
            ffw=None if ffws is None else ffws[i],
            context_hash=None if context_hash_list is None else context_hash_list[i],
            max_tree_depth=self.max_tree_depth,
            max_length=max_inference_seq_length,
            max_inference_reward=None if max_inference_reward_list is None else max_inference_reward_list[i],
            use_controllability_dont_cares=use_controllability_dont_cares,
            tts_compressed=None if tts_compressed is None else tts_compressed[i],
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            w_gate = w_gate,
            w_delay = w_delay,
            verbose=self.verbose)
            for i, aig in enumerate(aigs)]
        init_mcts_roots = [MCTSNode(None, 0, None, info={'env': env, 'reward': None, 'done': None, 'rollout_success': None}, puct_explore_ratio=puct_explore_ratio) for env in self._copy_env(envs)]
        transformer_time = 0.
        action_mask_time = 0.
        step_time = 0.
        init_time = time.time() - start_time
        if self.verbose > 0:
            print("optimization initialized, time cost %.2f" % init_time)
        for i in range(max_inference_seq_length):
            if all([e.is_finished for e in envs]):
                break
            inputs['targets'], inputs['dec_pos_encoding'] = targets, dec_pos_encoding

            # generate action mask
            start_time = time.time()
            action_masks = np.stack([e.action_masks[i] for e in envs], axis=0)
            inputs['dec_action_mask'] = np.expand_dims(action_masks, axis=1)
            action_mask_time += time.time() - start_time

            start_time = time.time()
            policy, cache = self._transformer_inference(inputs, return_kv_cache=True, return_last_token=True)
            transformer_time += time.time() - start_time
            inputs['cache'] = cache

            start_time = time.time()
            tokens = np.argmax(policy, axis=1)

            if num_mcts_steps > 0:
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
                    for j, b in enumerate(best_token_seqs):
                        if len(b) >= i - num_mcts_steps + 2:
                            tokens[j] = b[i - num_mcts_steps + 1]

            rewards, dones = zip(*[e.step(token) for token, e in zip(tokens, envs)])
            pos_encodings = [e.positional_encodings[-1] for e in envs]

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
                print(i, tokens)
        if return_envs and self.verbose == 0:
            return envs
        optimized_aigs = []
        num_succeed_aigs = 0
        total_gain, seq_total_gain_for_succeeded_aig, seq_total_gain = 0, 0, 0
        for i, (aig, env) in enumerate(zip(aigs, envs)):
            orig_num_ands = count_num_ands(aig)
            if self.verbose > 1:
                seq_roots, info = sequential_synthesis(aig)
                seq_total_gain += orig_num_ands - info[4]
            assert env.success == (len(env.roots) == len(aig) and check_integrity(env.roots))
            if env.success:
                num_succeed_aigs += 1
                if not return_envs:
                    optimized_aigs.append(env.roots)
                if self.verbose > 0:
                    num_ands = count_num_ands(env.roots)
                    total_gain += max(orig_num_ands - num_ands, 0)
                    print("aig #%d successfully optimized, #(AND) from %d to %d, cumulative reward %d, gain = %d" %
                          (i, orig_num_ands, num_ands, env.cumulative_reward, orig_num_ands - num_ands),
                          end="" if self.verbose > 1 else "\n")
                    if self.verbose > 1:
                        seq_total_gain_for_succeeded_aig += orig_num_ands - info[4]
                        print(" (resyn2: %d)" % info[4])
            else:
                if self.verbose > 0:
                    print("aig #%d (#(AND) = %d) failed to be optimized%s" %
                          (i, orig_num_ands, ', use original aig instead' if overflow_option == 'origin' else ''))
                    if self.verbose > 1:
                        print(" (resyn2: %d)" % info[4])
                if not return_envs:
                    optimized_aigs.append(aig if overflow_option == 'origin' else env.roots)
        if self.verbose > 0:
            print(
                "%d out of %d aigs successfully optimized, total time %.2f, init time %.2f, transformer time %.2f, action mask time %.2f, step time %.2f" %
                (num_succeed_aigs, len(aigs), time.time() - total_time, init_time, transformer_time, action_mask_time, step_time))
            if num_succeed_aigs > 0:
                print("average gain %.3f for successfully optimized aigs" % (total_gain / num_succeed_aigs))
            print("average gain %.3f for all aigs (failed aigs correspond to zero gain)" % (total_gain / len(aigs)))
            if self.verbose > 1:
                if num_succeed_aigs > 0:
                    print("resyn2: %.3f / %.3f" % (
                        seq_total_gain_for_succeeded_aig / num_succeed_aigs, seq_total_gain / len(aigs)))
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


if __name__ == "__main__":
    # 推理测试：传入预训练模型权重
    circuit_transformer = CircuitTransformer(ckpt_path='./ckpt-origin/deepsyn_reinforced')
    aig0, info0 = read_aiger(aiger_str="""aag 33 8 0 2 25
2\n4\n6\n8\n10\n12\n14\n16\n58\n67
18 13 16\n20 19 7\n22 21 15\n24 3 9\n26 25 11
28 27 17\n30 3 6\n32 29 31\n34 29 32\n36 23 35
38 7 36\n40 10 29\n42 41 32\n44 13 15\n46 42 45
48 47 21\n50 39 49\n52 4 45\n54 25 53\n56 54 5
58 51 57\n60 45 12\n62 18 61\n64 63 19\n66 48 64
""")
    aig1, info1 = read_aiger(aiger_str="""aag 22 8 0 2 14
2\n4\n6\n8\n10\n12\n14\n16\n24\n44
18 10 12\n20 8 7\n22 21 5\n24 19 23\n26 11 3
28 6 4\n30 26 28\n32 8 5\n34 32 26\n36 35 17
38 37 7\n40 31 39\n42 41 12\n44 43 15
""")
    aigs = [aig0, aig1]

    optimized_aigs = circuit_transformer.optimize(aigs)
    print("Circuit Transformer:")
    for i, (aig, optimized_aig) in enumerate(zip(aigs, optimized_aigs)):
        print("aig %d #(AND) from %d to %d, equivalence check: %r" %
              (i, count_num_ands(aig), count_num_ands(optimized_aig), cec(aig, optimized_aig)))

    optimized_aigs_with_mcts = circuit_transformer.optimize(
        aigs=aigs,
        num_mcts_steps=1,
        num_mcts_playouts_per_step=10
    )
    print("Circuit Transformer + Monte-Carlo Tree Search:")
    for i, (aig, optimized_aig) in enumerate(zip(aigs, optimized_aigs_with_mcts)):
        print("aig %d #(AND) from %d to %d, equivalence check: %r" %
              (i, count_num_ands(aig), count_num_ands(optimized_aig), cec(aig, optimized_aig)))
              