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
from circuit_transformer.tensorflow_transformer import Seq2SeqTransformer, CustomSchedule, masked_loss, masked_accuracy
from circuit_transformer.utils import *
from circuit_transformer.encoding import node_to_int, int_to_node, encode_aig, stack_to_encoding, deref_node
from circuit_transformer.environment import LogicNetworkEnv
from circuit_transformer.mcts import MCTSNode, ucb


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
                 w_delay=0,
                 return_cache=True
                 ):
        self.num_inputs = num_inputs
        self.vocab_size = 2 + 2 * self.num_inputs + 2 + 2
        # self.vocab_size = 2 + 2 * self.num_inputs + 2
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
            self.build_model()  
            self.load(self.ckpt_path) 

        def _transformer_inference_graph(self, inputs, return_kv_cache=False, return_last_token=False, return_value=False):
            policy, cache = self._transformer(inputs, training=False, return_kv_cache=return_kv_cache, return_last_token=return_last_token, return_value=return_value)
            return policy, cache

        def _transformer_inference(self, inputs, return_kv_cache=False, return_last_token=False, return_value=False):
            policy, cache = _transformer_inference_graph(self, inputs, return_kv_cache=return_kv_cache, return_last_token=return_last_token, return_value=return_value)
            # In graph mode, return tensor; in eager mode, return numpy array
            try:
                return policy.numpy(), cache
            except AttributeError:
                # policy is already a tensor in graph mode
                return policy, cache

        self._transformer_inference = types.MethodType(_transformer_inference, self)
        self._transformer.return_cache = return_cache
        self.use_kv_cache = True
        self.input_tt = compute_input_tt(self.num_inputs)

    def build_model(self):
        """explicitly build model variables"""
        dummy_inputs = {
            'inputs': tf.zeros((1, self.max_seq_length), dtype=tf.int32),
            'enc_pos_encoding': tf.zeros((1, self.max_seq_length, self.max_tree_depth*2)),
            'targets': tf.zeros((1, self.max_seq_length), dtype=tf.int32),
            'dec_pos_encoding': tf.zeros((1, self.max_seq_length, self.max_tree_depth*2)),
            'enc_action_mask': tf.zeros((1, self.max_seq_length, self.vocab_size), dtype=tf.bool),
            'dec_action_mask': tf.zeros((1, self.max_seq_length, self.vocab_size), dtype=tf.bool)
        }
        # Call once without returning value, then once with value to ensure all layers are initialized
        _ = self._transformer(dummy_inputs, return_value=False)
        _ = self._transformer(dummy_inputs, return_value=True)  # Trigger value_head initialization

    def freeze_layers(self, freeze_encoder=True):
        """Freeze encoder layers, only train decoder layers"""
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            # Freeze all encoder layers
            self._transformer.encoder_layer.trainable = False
            
            # Freeze encoder embedding layers
            self._transformer.enc_embedding_lookup.trainable = False
            
            # Freeze positional encoding layers
            self._transformer.position_embedding.trainable = False
            self._transformer.tree_position_embedding.trainable = False
            
            # Freeze value head (not trained in pre-training)
            self._transformer.value_head.trainable = False
            
            # Unfreeze decoder layers
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

    # add placeholder for training and inference methods
    def train(self, *args, **kwargs):
        """training method - provided by training module"""
        from .training import train
        return train(self, *args, **kwargs)
    
    def optimize(self, *args, **kwargs):
        """optimization method - provided by inference module"""
        from .inference import optimize
        return optimize(self, *args, **kwargs)
    
    def _batch_MCTS_policy_with_leaf_parallelization(self, *args, **kwargs):
        '''MCTS method - provided by mcts_optimization module'''
        from .mcts_optimization import _batch_MCTS_policy_with_leaf_parallelization
        return _batch_MCTS_policy_with_leaf_parallelization(self, *args, **kwargs)
    
    def _batch_estimate_policy(self, *args, **kwargs):
        '''MCTS policy method - provided by inference module'''
        from .inference import _batch_estimate_policy
        return _batch_estimate_policy(self, *args, **kwargs)
    
    def _batch_estimate_v_value_via_simulation_kvcache(self, *args, **kwargs):
        '''MCTS simulation method - provided by inference module'''
        from .inference import _batch_estimate_v_value_via_simulation_kvcache
        return _batch_estimate_v_value_via_simulation_kvcache(self, *args, **kwargs)
    
    def optimize_batch(self, *args, **kwargs):
        """batch optimization method - provided by inference module"""
        from .inference import optimize_batch
        return optimize_batch(self, *args, **kwargs) 