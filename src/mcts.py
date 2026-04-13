from __future__ import annotations

import numpy as np

class MCTSNode:
    INIT_MAX_VALUE = -1000

    def __init__(self, parent, t, action, prob=None, info=None, puct_explore_ratio=1.):
        self.t = t
        self.parent = parent
        self.action = action
        self.explored = False
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.total_value = 0
        self.info = info
        self.v = None
        self.prob = prob
        self.max_value = self.INIT_MAX_VALUE
        self.puct_explore_ratio = puct_explore_ratio

    @property
    def value(self):  # Q
        if self.visits != 0:
            return self.total_value / self.visits
        else:
            # For unvisited nodes, use optimistic initial value based on path progress
            # This helps differentiate between nodes at different depths and actions
            initial_value = 100.0
            
            if self.info is not None and self.info.get('env') is not None:
                env = self.info['env']
                # Calculate progress: number of tokens generated so far
                path_length = len(env.tokens) if hasattr(env, 'tokens') else 0
                # Shorter paths (fewer AND gates) should have slightly higher initial value
                # Longer paths get slightly lower initial value (penalty for more gates)
                initial_value -= min(path_length * 0.2, 10.0)  # Max 10.0 penalty for very long paths
            
            # Add small variation based on action to help differentiate nodes at same depth
            # This prevents all nodes at the same depth from having identical initial values
            if self.action is not None:
                # Use action value to add small deterministic variation (0-2.0 range)
                action_variation = (self.action % 100) / 50.0  # Normalize to 0-2.0
                initial_value += action_variation
            
            return initial_value

    @property
    def puct(self):
        base_puct = self.value + self.puct_explore_ratio * self.prob * np.sqrt(self.parent.visits) / (1 + self.visits)
        
        # Penalize unfinished sequences and reward completed ones
        if self.info is not None:
            done = self.info.get('done', False)
            rollout_success = self.info.get('rollout_success', None)
            
            # Penalty for unfinished sequences (done=False)
            if not done:
                # Apply penalty: reduce PUCT value to discourage unfinished paths
                # The penalty scales with exploration ratio to maintain balance
                # This encourages MCTS to explore paths that lead to completion
                penalty = -0.3 * self.puct_explore_ratio
                base_puct += penalty
            
            # Additional penalty for failed rollouts (done=True but rollout_success=False)
            elif done and rollout_success is False:
                # Strong penalty for paths that completed but failed in rollout
                penalty = -0.8 * self.puct_explore_ratio
                base_puct += penalty
            
            # Bonus for completed successful sequences (done=True and rollout_success=True)
            elif done and rollout_success is True:
                # Bonus for completed successful paths to encourage exploration
                bonus = 0.5 * self.puct_explore_ratio
                base_puct += bonus
        
        return base_puct

    def __repr__(self):     # sum reward: from the root to the end, value: future reward from (excluding) the current node to the end
        repr = "(%s%s, visits: %d, avg sum reward: %.2f, max sum reward: %d, value: %s, seq: %s)" % \
               (self.action, " (Done)" if self.info['done'] else "", self.visits, self.value, self.max_value, self.v, self.info['env'].tokens)
        if self.prob is not None:
            repr = repr[:-1] + ", prob: %.2f, puct: %.2f)" % (self.prob, self.puct)
        return repr


def ucb(node: MCTSNode):
    """UCB (Upper Confidence Bound) 函数"""
    return node.value + np.sqrt(np.log(node.parent.visits) / node.visits) 