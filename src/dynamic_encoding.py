"""
Dynamic Encoding Module for Scalable Circuit Transformer

This module implements a dynamic vocabulary and encoding system that supports
variable input/output circuits without being constrained to fixed sizes.

Key Features:
- Dynamic vocabulary size based on actual circuit inputs
- Efficient token mapping that scales with circuit complexity
- Support for arbitrary number of inputs and outputs
- Backward compatible with fixed-size models
"""

from __future__ import annotations
import numpy as np
from scalable_circuit_transformer_refdfs.utils import Node, NodeWithInv

class DynamicEncoder:
    """
    Dynamic encoder that adapts to different circuit sizes.

    Token Structure (Fixed Position Layout):
    0: PAD
    1: EOS
    2 to 2+2N-1: PI1, ~PI1, PI2, ~PI2, ..., PIN, ~PIN (for N inputs)
    2+2N to max_vocab_size-5: [Unused, masked out for circuits with < max_inputs]
    max_vocab_size-4: AND (fixed position)
    max_vocab_size-3: NAND (fixed position)
    max_vocab_size-2: Constant 0 (fixed position)
    max_vocab_size-1: Constant 1 (fixed position)

    Total vocab size = max_vocab_size = 2 + 2*max_inputs + 4 (fixed)
    For circuits with N < max_inputs inputs, positions 2+2N to max_vocab_size-5 are masked.
    """

    def __init__(self, max_inputs=64, max_outputs=64, max_seq_length=500, use_memory_dfs=False):
        """
        Initialize dynamic encoder.

        Args:
            max_inputs: Maximum number of inputs to support (default: 64)
            max_outputs: Maximum number of outputs to support (default: 64)
            max_seq_length: Maximum sequence length (default: 500)
            use_memory_dfs: Whether to use memory DFS encoding (default: False)
        """
        self.max_inputs = max_inputs
        self.max_outputs = max_outputs
        self.max_seq_length = max_seq_length
        self.max_vocab_size = 2 + 2 * max_inputs + 4  # Maximum possible vocabulary
        self.use_memory_dfs = use_memory_dfs
        # Reference token: use max_vocab_size as REF_TOKEN
        # This requires extending vocab_size by 1 when using memory DFS
        self.REF_TOKEN = self.max_vocab_size  # Token for referencing previously encoded nodes

    def get_vocab_size(self, num_inputs: int) -> int:
        """
        Get vocabulary size for a given number of inputs.
        
        Note: Returns max_vocab_size (fixed) regardless of num_inputs.
        The actual valid tokens for a circuit with N inputs are:
        - Positions 0-1: PAD, EOS
        - Positions 2 to 2+2N-1: Input variables
        - Positions max_vocab_size-4 to max_vocab_size-1: AND, NAND, 0, 1
        - Positions 2+2N to max_vocab_size-5: Masked out (invalid)
        - If use_memory_dfs: max_vocab_size is REF_TOKEN
        """
        if self.use_memory_dfs:
            return self.max_vocab_size + 1  # Add 1 for REF_TOKEN
        return self.max_vocab_size

    def node_to_int(self, root: NodeWithInv, num_inputs: int) -> int:
        """
        Convert a node to an integer token (fixed position layout).

        Args:
            root: The node to convert
            num_inputs: Number of inputs in the circuit (unused for AND/NAND/constants)

        Returns:
            Integer token ID
        """
        vocab_size = self.get_vocab_size(num_inputs)  # Includes REF_TOKEN if using memory DFS
        if root.is_leaf():
            if root.var >= 0:
                # Input variable: 2 + var*2 + inverted
                return 2 + root.var * 2 + root.inverted
            else:
                # Constant: fixed positions at end
                if root.var == -1:
                    # Constant 0: vocab_size - 3
                    # Constant 1: vocab_size - 2
                    if self.use_memory_dfs:
                        constant_0_id = vocab_size - 3
                    else:
                        constant_0_id = vocab_size - 2
                    return constant_0_id + root.inverted
                else:
                    raise ValueError(f"Invalid constant var: {root.var}")
        else:
            # AND gate: fixed positions at end
            # AND: vocab_size - 5
            # NAND: vocab_size - 4
            if self.use_memory_dfs:
                and_token_id = vocab_size - 5
            else:
                and_token_id = vocab_size - 4
            return and_token_id + root.inverted

    def int_to_node(self, token: int, num_inputs: int) -> NodeWithInv | bool:
        """
        Convert an integer token to a node (fixed position layout).

        Args:
            token: Token ID
            num_inputs: Number of inputs in the circuit

        Returns:
            NodeWithInv object or False for special tokens
        """
        vocab_size = self.get_vocab_size(num_inputs)  # Includes REF_TOKEN if using memory DFS
        if token == 0 or token == 1:  # PAD or EOS
            return False
        elif token < 2 + num_inputs * 2:  # Input variable
            var_id = (token - 2) // 2
            inverted = token % 2
            return NodeWithInv(Node(var_id, None, None), inverted)
        elif self.use_memory_dfs:
            if token == vocab_size - 5:  # AND (fixed position)
                return NodeWithInv(Node(None, None, None), inverted=False)
            elif token == vocab_size - 4:  # NAND (fixed position)
                return NodeWithInv(Node(None, None, None), inverted=True)
            elif token == vocab_size - 3:  # Constant 0 (fixed position)
                return NodeWithInv(Node(-1, None, None), inverted=False)
            elif token == vocab_size - 2:  # Constant 1 (fixed position)
                return NodeWithInv(Node(-1, None, None), inverted=True)
            elif token == vocab_size - 1:  # REF_TOKEN (if using memory DFS)
                return False  # REF_TOKEN is handled separately
            elif token < vocab_size - 5:
                # Invalid: token in the unused middle range (for circuits with < max_inputs)
                raise ValueError(f"Invalid token {token} for {num_inputs} inputs: token in unused range")
        else:
            if token == vocab_size - 4:  # AND (fixed position)
                return NodeWithInv(Node(None, None, None), inverted=False)
            elif token == vocab_size - 3:  # NAND (fixed position)
                return NodeWithInv(Node(None, None, None), inverted=True)
            elif token == vocab_size - 2:  # Constant 0 (fixed position)
                return NodeWithInv(Node(-1, None, None), inverted=False)
            elif token == vocab_size - 1:  # Constant 1 (fixed position)
                return NodeWithInv(Node(-1, None, None), inverted=True)
            raise ValueError(f"Invalid token {token} (exceeds vocab_size {vocab_size})")

    def encode_aig(self, roots: list[NodeWithInv], num_inputs: int) -> tuple[list[int], list[int]]:
        """
        Encode AIG to token sequence and position encoding.
        
        Uses memory DFS encoding if use_memory_dfs=True, otherwise uses memoryless DFS.

        Args:
            roots: List of output nodes
            num_inputs: Number of inputs in the circuit

        Returns:
            Tuple of (token_sequence, position_encoding)
        """
        if self.use_memory_dfs:
            return self.encode_aig_memory_dfs(roots, num_inputs)
        else:
            return self.encode_aig_memoryless(roots, num_inputs)
    
    def encode_aig_memoryless(self, roots: list[NodeWithInv], num_inputs: int) -> tuple[list[int], list[int]]:
        """
        Memoryless DFS encoding (original implementation).
        
        Each node is encoded every time it is encountered, even if it has been encoded before.
        This is the original encoding method for backward compatibility.

        Args:
            roots: List of output nodes
            num_inputs: Number of inputs in the circuit

        Returns:
            Tuple of (token_sequence, position_encoding)
        """
        def encode_aig_rec(root: NodeWithInv, seq_enc: list[int],
                          cur_pos_enc: int, pos_enc: list[int]):
            seq_enc.append(self.node_to_int(root, num_inputs))
            pos_enc.append(cur_pos_enc)
            if not root.is_leaf():
                encode_aig_rec(root.left, seq_enc, (cur_pos_enc << 2) + 1, pos_enc)
                encode_aig_rec(root.right, seq_enc, (cur_pos_enc << 2) + 2, pos_enc)

        # Ensure roots is always a list
        if not isinstance(roots, list):
            roots = [roots]

        seq_enc, pos_enc = [], []
        num_outputs = len(roots)

        # Encode each output with different root position
        for i, root in enumerate(roots):
            encode_aig_rec(root, seq_enc, i + 1, pos_enc)

        return seq_enc, pos_enc
    
    def encode_aig_memory_dfs(self, roots: list[NodeWithInv], num_inputs: int) -> tuple[list[int], list[int]]:
        """
        Memory DFS encoding with node reference support.
        
        Each node is encoded only once. When a node is encountered again, a reference
        to its previous encoding position is used instead.
        
        Reference format: [REF_TOKEN, position_in_sequence]
        - REF_TOKEN: self.REF_TOKEN (max_vocab_size)
        - position_in_sequence: position of the referenced node in the sequence

        Args:
            roots: List of output nodes
            num_inputs: Number of inputs in the circuit

        Returns:
            Tuple of (token_sequence, position_encoding)
        """
        def get_node_id(node: NodeWithInv) -> tuple:
            """Get unique identifier for a node (using parent node's id and inverted flag)"""
            # Include inverted flag to distinguish AND from NAND
            # Two nodes with same parent but different inverted flags are different nodes
            return (id(node.parent), node.inverted)

        # Ensure roots is always a list
        if not isinstance(roots, list):
            roots = [roots]

        seq_enc, pos_enc = [], []
        node_to_position = {}  # Map: node_id -> position in sequence (excluding REF_TOKEN and position tokens)
        seq_pos_counter = [0]  # Counter for actual node positions (excluding REF_TOKEN and position tokens)
        
        def encode_aig_rec(root: NodeWithInv, seq_enc: list[int],
                          cur_pos_enc: int, pos_enc: list[int],
                          node_to_position: dict, seq_pos_counter_ref: list):
            # Only AND/NAND nodes (non-leaf) can be referenced
            # Inputs (leaf nodes) are always encoded directly, not referenced
            is_and_nand = not root.is_leaf()
            
            if is_and_nand:
                node_id = get_node_id(root)
                
                # Check if node has been encoded before (only for AND/NAND nodes)
                if node_id in node_to_position:
                    # Reference to previously encoded node
                    ref_position = node_to_position[node_id]
                    seq_enc.append(self.REF_TOKEN)
                    seq_enc.append(ref_position)  # Position of referenced node (excluding REF_TOKEN)
                    pos_enc.append(cur_pos_enc)
                    pos_enc.append((cur_pos_enc << 2) + 1)
                    # REF_TOKEN and position token don't count towards seq_pos
                    # But we still need to increment seq_pos for the referenced node
                    seq_pos_counter_ref[0] += 2
                    return
            
            # Encode node for the first time
            seq_enc.append(self.node_to_int(root, num_inputs))
            pos_enc.append(cur_pos_enc)
            
            # Only store position for AND/NAND nodes (non-leaf)
            # Store the seq_pos_counter (excluding REF_TOKEN and position tokens)
            if is_and_nand:
                node_id = get_node_id(root)
                node_to_position[node_id] = seq_pos_counter_ref[0]
                seq_pos_counter_ref[0] += 1
            
            # Recursively encode children
            if not root.is_leaf():
                encode_aig_rec(root.left, seq_enc, (cur_pos_enc << 2) + 1, pos_enc, node_to_position, seq_pos_counter_ref)
                encode_aig_rec(root.right, seq_enc, (cur_pos_enc << 2) + 2, pos_enc, node_to_position, seq_pos_counter_ref)
            else:
                # Leaf nodes (inputs) also count towards seq_pos
                seq_pos_counter_ref[0] += 1
        
        # Encode each output with different root position
        k = self.max_outputs.bit_length() + 1 
        offset = (1 << k)
        for i, root in enumerate(roots):
            encode_aig_rec(root, seq_enc, offset + i + 1, pos_enc, node_to_position, seq_pos_counter)
        # print(f"node_to_position: {node_to_position}")
        return seq_enc, pos_enc

    def get_valid_token_mask(self, num_inputs: int) -> np.ndarray:
        """
        Get a boolean mask indicating which tokens are valid for a circuit with num_inputs.

        Args:
            num_inputs: Number of inputs in the circuit

        Returns:
            Boolean array of shape (vocab_size,) where True indicates valid tokens
        """
        vocab_size = self.get_vocab_size(num_inputs)
        mask = np.zeros(vocab_size, dtype=bool)
        
        # PAD and EOS (always valid)
        mask[0] = True  # PAD
        mask[1] = True  # EOS
        
        # Input variables: positions 2 to 2+2*num_inputs-1
        input_end = 2 + num_inputs * 2
        mask[2:input_end] = True
        
        # Fixed positions at end: AND, NAND, Constant 0, Constant 1
        # Use vocab_size (includes REF_TOKEN if using memory DFS), not max_vocab_size
        # If using memory DFS, REF_TOKEN is also valid

        if self.use_memory_dfs:
            mask[vocab_size - 5] = True  # AND
            mask[vocab_size - 4] = True  # NAND
            mask[vocab_size - 3] = True  # Constant 0
            mask[vocab_size - 2] = True  # Constant 1
            mask[vocab_size - 1] = True  # REF_TOKEN
        
        else:
            mask[vocab_size - 4] = True  # AND
            mask[vocab_size - 3] = True  # NAND
            mask[vocab_size - 2] = True  # Constant 0
            mask[vocab_size - 1] = True  # Constant 1
        

        
        # Positions input_end to vocab_size-6 are masked (unused for this circuit)
        # They remain False
        
        return mask

    def decode_sequence(self, tokens: list[int], num_inputs: int) -> list[NodeWithInv]:
        """
        Decode a token sequence back to AIG nodes.
        
        Supports both memoryless and memory DFS decoding.

        Args:
            tokens: List of token IDs
            num_inputs: Number of inputs in the circuit

        Returns:
            List of root nodes representing outputs
        """
        if self.use_memory_dfs:
            return self.decode_sequence_memory_dfs(tokens, num_inputs)
        else:
            return self.decode_sequence_memoryless(tokens, num_inputs)
    
    def decode_sequence_memoryless(self, tokens: list[int], num_inputs: int) -> list[NodeWithInv]:
        """
        Decode memoryless DFS encoded sequence (original implementation).
        
        Encoding order: DFS pre-order (node, left, right)
        Decoding: Need to process in reverse order (post-order)

        Args:
            tokens: List of token IDs
            num_inputs: Number of inputs in the circuit

        Returns:
            List of root nodes representing outputs
        """
        stack = []
        i = 0
        
        def decode_rec():
            """Recursive decode helper"""
            nonlocal i
            if i >= len(tokens):
                return None
            
            token = tokens[i]
            i += 1
            
            if token == 0:  # PAD
                return decode_rec()  # Skip and continue
            elif token == 1:  # EOS
                return None
            
            node = self.int_to_node(token, num_inputs)
            if node is False:
                return decode_rec()  # Skip and continue
            
            if not node.is_leaf():
                # AND gate: recursively decode children first
                node.left = decode_rec()
                node.right = decode_rec()
            
            return node
        
        # Decode all roots
        roots = []
        while i < len(tokens):
            if tokens[i] == 1:  # EOS
                break
            root = decode_rec()
            if root is not None:
                roots.append(root)
            else:
                break
        
        return roots
    
    def decode_sequence_memory_dfs(self, tokens: list[int], num_inputs: int) -> list[NodeWithInv]:
        """
        Decode memory DFS encoded sequence with node references.
        
        Encoding order: DFS pre-order (node, left, right) with references
        Decoding: Need to process in reverse order (post-order) with reference handling
        
        Handles REF_TOKEN references to previously decoded nodes.
        The reference position refers to the position in the sequence (not stack).

        Args:
            tokens: List of token IDs
            num_inputs: Number of inputs in the circuit

        Returns:
            List of root nodes representing outputs
        """
        seq_position_to_node = {}  # Map: sequence_position -> node (for references)
        seq_pos = 0  # Current position in sequence (excluding REF_TOKEN and position tokens)
        i = 0
        
        def decode_rec():
            """Recursive decode helper with reference support"""
            nonlocal i, seq_pos
            
            if i >= len(tokens):
                return None
            
            token = tokens[i]
            i += 1
            
            if token == 0:  # PAD
                return decode_rec()  # Skip and continue
            elif token == 1:  # EOS
                return None
            elif token == self.REF_TOKEN:
                # Reference to previously decoded node
                if i < len(tokens):
                    ref_seq_position = tokens[i]
                    i += 1
                    if ref_seq_position in seq_position_to_node:
                        # Get referenced node
                        # In AIG, shared nodes should share the same parent node
                        ref_node = seq_position_to_node[ref_seq_position]
                        seq_position_to_node[seq_pos] = ref_node
                        # REF occupies 2 positions (must match encode and generate_action_masks)
                        seq_pos += 2
                        return ref_node
                    else:
                        # Invalid reference
                        print(f"Warning: Invalid reference position {ref_seq_position} at token {i-2}")
                        print(f"  Available positions: {sorted(seq_position_to_node.keys())}")
                        print(f"  Current seq_pos: {seq_pos}")
                        return None
                else:
                    # Incomplete reference
                    print(f"Warning: Incomplete reference at token {i-1}")
                    return None
            else:
                # Regular token
                node = self.int_to_node(token, num_inputs)
                if node is False:
                    return decode_rec()  # Skip and continue
                
                # Store node position for future references (before decoding children)
                current_pos = seq_pos
                seq_position_to_node[current_pos] = node
                seq_pos += 1
                
                if not node.is_leaf():
                    # AND gate: recursively decode children first
                    node.left = decode_rec()
                    node.right = decode_rec()
                
                return node
        
        # Decode all roots
        roots = []
        while i < len(tokens):
            if tokens[i] == 1:  # EOS
                break
            root = decode_rec()
            if root is not None:
                roots.append(root)
            else:
                break
        
        return roots

    def get_circuit_metadata(self, roots: list[NodeWithInv]) -> dict:
        """
        Extract metadata from a circuit.

        Args:
            roots: List of output nodes

        Returns:
            Dictionary containing circuit metadata
        """
        visited = set()
        max_var = -1
        num_ands = 0

        def traverse(node: NodeWithInv):
            nonlocal max_var, num_ands

            if node.parent in visited:
                return
            visited.add(node.parent)

            if node.is_leaf():
                if node.var >= 0:
                    max_var = max(max_var, node.var)
            else:
                num_ands += 1
                if node.left:
                    traverse(node.left)
                if node.right:
                    traverse(node.right)

        for root in roots:
            traverse(root)

        return {
            'num_inputs': max_var + 1 if max_var >= 0 else 0,
            'num_outputs': len(roots),
            'num_ands': num_ands,
            'num_nodes': len(visited)
        }


def create_compatible_encoder(num_inputs: int = 8, **kwargs) -> DynamicEncoder:
    """
    Factory function to create an encoder compatible with legacy code.

    Args:
        num_inputs: Default number of inputs for backward compatibility
        **kwargs: Additional arguments passed to DynamicEncoder

    Returns:
        DynamicEncoder instance
    """
    return DynamicEncoder(**kwargs)
