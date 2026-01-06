import time
import numpy as np
import scipy.special as special
from circuit_transformer.utils import *
from circuit_transformer.environment import LogicNetworkEnv
from circuit_transformer.mcts import MCTSNode
from circuit_transformer.encoding import node_to_int, int_to_node, encode_aig, stack_to_encoding, deref_node

def _batch_estimate_policy(self, envs: list, src_tokens, src_pos_enc, src_action_mask, action_masks, cache):
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


def _batch_estimate_v_value_via_simulation_kvcache(self, envs: list, src_tokens, src_pos_enc, src_action_mask,
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
            error_rate_threshold = 1.0,
            w_gate=1,
            w_delay=0,
            use_greedy_decoding=False
            ):
    if self.ckpt_path is None:
        print("no checkpoint loaded")
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
                                            error_rate_threshold=error_rate_threshold,
                                            w_gate=w_gate,
                                            w_delay=w_delay,
                                            use_greedy_decoding=use_greedy_decoding
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
                error_rate_threshold = 1.0,
                w_gate=1,
                w_delay=0,
                use_greedy_decoding=False
                ):
    # print(f"error threshold: {error_rate_threshold}")
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
        error_rate_threshold = error_rate_threshold,
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
        # print(inputs['dec_action_mask'].shape)
        action_mask_time += time.time() - start_time

        start_time = time.time()
        policy, cache = self._transformer_inference(inputs, return_kv_cache=True, return_last_token=True)
        transformer_time += time.time() - start_time
        inputs['cache'] = cache

        start_time = time.time()
        # Use greedy decoding for deterministic evaluation (Algorithm 2 requirement)
        if use_greedy_decoding:
            tokens = np.argmax(policy, axis=1)
        else:
            # Default behavior (can be probabilistic if needed)
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
        # print(tokens)
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