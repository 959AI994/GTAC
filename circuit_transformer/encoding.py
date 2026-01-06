from __future__ import annotations

import numpy as np
from circuit_transformer.utils import Node, NodeWithInv

'''
for 8-input, 2-output circuits
Token(int):
0: PAD
1: EOS
2,4,6,...,16: PI1, PI2, PI3, ... , PI8
3,5,7,...,17: ~PI1, ~PI2, ~PI3, ..., ~PI8
18: AND
19: NAND

Newly add:
20: constant 0
21: constant 1
'''

def node_to_int(root: NodeWithInv, num_inputs: int):
    """convert node to integer token"""
    # zero for [PAD] that will be masked
    if root.is_leaf():
        if root.var >= 0:
            return 2 + root.var * 2 + root.inverted  # [x_i] or [x_i_NOT]
        else:
            return 21 + root.var + root.inverted    # [constant 0] or [constant 1]
    else:
        return 2 + num_inputs * 2 + root.inverted  # [AND] or [AND_NOT]


def int_to_node(token: int, num_inputs: int):
    """convert integer token to node"""
    if token == 0 or token == 1:  # [PAD] or [EOS]
        return False
    elif token < 2 + num_inputs * 2:  # [x_i] or [x_i_NOT]
        return NodeWithInv(Node((token - 2) // 2, None, None), token % 2)
    elif token < 2 + num_inputs * 2 + 2:  # [AND] or [AND_NOT]
        return NodeWithInv(Node(None, None, None), token % 2)
    elif token < 2 + num_inputs *2 + 4:  # [constant 0] or [constant 1]
        return NodeWithInv(Node(-1,None,None), token % 2)  
    else:
        raise ValueError(token)


def encode_aig(roots: list[NodeWithInv], num_inputs: int) -> (list[int], list[int]):
    """encode AIG to sequence and position encoding"""
    def encode_aig_rec(root: NodeWithInv, seq_enc: list[int], cur_pos_enc: int, pos_enc: list[int]):
        seq_enc.append(node_to_int(root, num_inputs))
        pos_enc.append(cur_pos_enc)
        if not root.is_leaf():
            encode_aig_rec(root.left, seq_enc, (cur_pos_enc << 2) + 1, pos_enc)
            encode_aig_rec(root.right, seq_enc, (cur_pos_enc << 2) + 2, pos_enc)

    seq_enc, pos_enc = [], []
    assert len(roots) <= 2
    encode_aig_rec(roots[0], seq_enc, 1, pos_enc)
    if len(roots) == 2:
        encode_aig_rec(roots[1], seq_enc, 2, pos_enc)
    return seq_enc, pos_enc


def stack_to_encoding(tree_stack: list, root_id: int, max_tree_depth: int):
    """convert tree stack to encoding"""
    assert len(tree_stack) <= max_tree_depth
    assert root_id >= 0
    encoding = np.zeros(max_tree_depth * 2, np.float32)
    for i, node in enumerate(reversed(tree_stack)):
        if i == 0:
            if node.left is None:  # the current node should be inserted on the left side
                encoding[i * 2] = 1.
            else:
                encoding[i * 2 + 1] = 1.
        else:
            if node.right is None:  # the current node is inserted on the left side
                encoding[i * 2] = 1.
            else:
                encoding[i * 2 + 1] = 1.
    encoding[len(tree_stack) * 2 + root_id] = 1.
    return encoding


def deref_node(root: Node, ref_dict: dict, context_nodes=None, verbose=0):
    """dereference node"""
    if context_nodes is not None and root in context_nodes:
        return 0
    if root.is_leaf():
        return 0
    value = 1
    for child in [root.left.parent, root.right.parent]:
        if verbose > 1:
            print("ref %s (parent %s, %s) from %d to %d" %
                  (child, root, "left" if child is root.left.parent else "right", ref_dict[child], ref_dict[child] - 1))
        ref_dict[child] -= 1
        if ref_dict[child] == 0:
            value += deref_node(child, ref_dict, context_nodes, verbose)
    return value 