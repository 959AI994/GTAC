from __future__ import annotations

import numpy as np
from scalable_circuit_transformer_refdfs.utils import Node, NodeWithInv, base_2_log
import npn

# npn.int_to_tt allocates 2^n_vars elements; overflow when n_vars > ~24
_NPN_INT_TO_TT_MAX_VARS = 24


def _int_to_binary_lsb(n: int, n_vars: int) -> list:
    """Convert int to LSB-first binary list (matches npn.int_to_tt output format)."""
    return [(n >> i) & 1 for i in range(n_vars)]


def get_pos_encoding_n_vars(max_tree_depth: int, max_outputs: int = 256) -> int:
    """n_vars for position encoding; must match stack_to_encoding."""
    k = max_outputs.bit_length() + 1
    min_vars_for_roots = k + 1
    min_vars_for_depth = base_2_log(max_tree_depth) + 1
    min_vars_for_path = k + 2 * max_tree_depth
    target_len = max_tree_depth * 2
    return min(target_len, max(min_vars_for_roots, min_vars_for_depth, min_vars_for_path))


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

Position encoding (training vs inference alignment):
- Training (encode_aig_memory_dfs): REF_TOKEN uses cur_pos_enc (parent path),
  ref_index uses (cur_pos_enc << 2) + 1 (left subtree path). stack_to_encoding
  (use_current_node_only=True) for REF_TOKEN yields parent path; env.step(ref_index)
  computes (cur_pos_enc << 2) + 1 from prev_enc. Both must match training.
'''

def node_to_int(root: NodeWithInv, num_inputs: int) -> int:
    """
    Convert a node to an integer token.

    Args:
        root: The node to convert
        num_inputs: Number of inputs in the circuit

    Returns:
        Integer token ID
    """
    if root.is_leaf():
        if root.var >= 0:
            # Input variable: 2 + var*2 + inverted
            return 2 + root.var * 2 + root.inverted
        else:
            # Constant: token_id for constant_0 + offset
            constant_0_id = 2 + num_inputs * 2 + 2
            return constant_0_id + (root.var + 1) + root.inverted
    else:
        # AND gate: base_id + inverted
        and_token_id = 2 + num_inputs * 2
        return and_token_id + root.inverted

def int_to_node(token: int, num_inputs: int) -> NodeWithInv | bool:
    """
    Convert an integer token to a node.

    Args:
        token: Token ID
        num_inputs: Number of inputs in the circuit

    Returns:
        NodeWithInv object or False for special tokens
    """    
    if token == 0 or token == 1:  # PAD or EOS
        return False
    elif token < 2 + num_inputs * 2:  # Input variable
        var_id = (token - 2) // 2
        inverted = token % 2
        return NodeWithInv(Node(var_id, None, None), inverted)
    elif token < 2 + num_inputs * 2 + 2:  # AND/NAND gate
        inverted = token % 2
        return NodeWithInv(Node(None, None, None), inverted)
    elif token < 2 + num_inputs * 2 + 4:  # Constant
        inverted = token % 2
        return NodeWithInv(Node(-1, None, None), inverted)
    else:
        raise ValueError(f"Invalid token {token} for {num_inputs} inputs")


def encode_aig(roots: list[NodeWithInv], num_inputs: int) -> (list[int], list[int]):
    """Encode AIG as token sequence and positional encodings."""
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


def stack_to_encoding(tree_stack: list, root_id: int, max_tree_depth: int, num_outputs: int = None,
                     max_outputs: int = None, child_step: int = None):
    """
    [FIXED] Convert tree stack to position encoding using Integer Bit-Shift.
    Strictly aligns with `encode_aig` + `_encode_postprocess` in training.

    Args:
        max_outputs: Must match DynamicEncoder.max_outputs for offset alignment.
            If None, falls back to num_outputs or 256 for backward compat.
        child_step: When adding a child, 1=left or 2=right. If provided and tree_stack
            is not empty, returns path to the child (parent_path << 2) + child_step,
            matching training's encode_aig_rec timing.
    """
    effective_max = max_outputs if max_outputs is not None else (num_outputs if num_outputs is not None else 256)
    k = effective_max.bit_length() + 1
    offset = (1 << k)
    
    current_pos_int = offset + root_id + 1
    
    if len(tree_stack) > 0:
        for i in range(len(tree_stack) - 1):
            parent = tree_stack[i]
            child = tree_stack[i+1]
            step = 1 if parent.left == child else 2
            current_pos_int = (current_pos_int << 2) + step
        if child_step is not None and child_step in (1, 2):
            current_pos_int = (current_pos_int << 2) + child_step

    target_len = max_tree_depth * 2

    n_vars = get_pos_encoding_n_vars(max_tree_depth, num_outputs if num_outputs else 256)

    # npn.int_to_tt allocates 2^n_vars; use manual conversion when n_vars too large
    if n_vars <= _NPN_INT_TO_TT_MAX_VARS:
        binary_list = npn.int_to_tt(current_pos_int, n_vars)
    else:
        binary_list = _int_to_binary_lsb(current_pos_int, n_vars)
    
    final_pos = list(reversed(binary_list))
    
    pos_array = np.array(final_pos, dtype=np.float32)
    target_len = max_tree_depth * 2
   
    if len(pos_array) > target_len:
        return pos_array[:target_len]
    else:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:len(pos_array)] = pos_array
        return padded

def ref_position_encoding_from_ref_token(prev_enc: np.ndarray, max_tree_depth: int, max_outputs: int = 256) -> np.ndarray:
    """
    Compute positional encoding for ref_index from the REF_TOKEN encoding (aligned with training).
    """
    from scalable_circuit_transformer_refdfs.encoding import get_pos_encoding_n_vars, _int_to_binary_lsb, _NPN_INT_TO_TT_MAX_VARS
    
    # 1. Bit width (must match training / stack_to_encoding)
    n_vars = get_pos_encoding_n_vars(max_tree_depth, max_outputs)

    # 2. Recover integer path: prev_enc is MSB-first (reversed LSB)
    prev_bits = prev_enc[:n_vars] 
    prev_bits_lsb = list(reversed(prev_bits)) 
    cur_pos_int = sum(int(b) * (2 ** i) for i, b in enumerate(prev_bits_lsb))
    
    # 3. Path step (left-child)
    next_pos_int = (cur_pos_int << 2) + 1

    # 4. Back to binary bitstream
    if n_vars <= _NPN_INT_TO_TT_MAX_VARS:
        binary_list = npn.int_to_tt(next_pos_int, n_vars)
    else:
        binary_list = _int_to_binary_lsb(next_pos_int, n_vars)
    
    final_pos = list(reversed(binary_list))
    
    # 5. Pack into fixed-length vector
    target_len = max_tree_depth * 2
    pos_array = np.zeros(target_len, dtype=np.float32)
    num_to_copy = min(len(final_pos), target_len)
    pos_array[:num_to_copy] = np.array(final_pos[:num_to_copy], dtype=np.float32)
    
    return pos_array

def deref_node(root: Node, ref_dict: dict, context_nodes=None, verbose=0):
    """Dereference / count shared nodes under root."""
    if context_nodes is not None and root in context_nodes:
        return 0
    if root.is_leaf():
        return 0
    value = 1
    for child in [root.left.parent, root.right.parent]:
        # Skip if child is not in ref_dict (may happen for REF_TOKEN deepcopied nodes)
        if child not in ref_dict:
            if verbose > 1:
                print(f"ref {child} (parent {root}) not in ref_dict, skipping")
            continue
        if verbose > 1:
            print("ref %s (parent %s, %s) from %d to %d" %
                  (child, root, "left" if child is root.left.parent else "right", ref_dict[child], ref_dict[child] - 1))
        ref_dict[child] -= 1
        if ref_dict[child] == 0:
            value += deref_node(child, ref_dict, context_nodes, verbose)
    return value 