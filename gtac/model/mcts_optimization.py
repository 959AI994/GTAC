import copy
import time
import numpy as np
from gtac.mcts import MCTSNode


def _batch_MCTS_policy_with_leaf_parallelization(self, envs: list, num_leaf_parallelizations=8, num_playouts=100, max_inference_seq_length=None,
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