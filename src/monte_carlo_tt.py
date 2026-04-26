"""
Monte Carlo Approximation for Truth Table Computation

This module provides Monte Carlo sampling-based approximate truth table computation,
similar to the C++ implementation in graph_partion/als/src/subCkt.cc.

Instead of exhaustively computing all 2^n truth table entries, we use Monte Carlo
sampling to approximate the truth table with significantly reduced computation.
"""

import numpy as np
import bitarray
import bitarray.util
from typing import Union, List, Optional, Dict
from src.utils import *
from src.utils import NodeWithInv, Node

# Note: No maximum input limit needed for Monte Carlo approximation
# Monte Carlo uses sampling (fixed memory O(n_samples)) instead of full truth tables
# (exponential memory O(2^num_inputs)), so can handle circuits with any number of inputs


class MonteCarloTT:
    """
    Monte Carlo approximate truth table computation
    """

    def __init__(self, num_inputs: int, n_samples: int = 256000, seed: Optional[int] = None):
        """
        Initialize Monte Carlo TT approximator

        Args:
            num_inputs: Number of circuit inputs
            n_samples: Number of Monte Carlo samples (should be multiple of 64)
            seed: Random seed for reproducibility
        """
        self.num_inputs = num_inputs
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Ensure n_samples is multiple of 64 for efficiency
        if n_samples % 64 != 0:
            self.n_samples = ((n_samples // 64) + 1) * 64
            print(f"Warning: n_samples adjusted to {self.n_samples} (multiple of 64)")

    def generate_samples(self, distribution='uniform') -> np.ndarray:
        """
        Generate Monte Carlo input samples

        Args:
            distribution: 'uniform' for uniform random sampling, 'enum' for enumeration

        Returns:
            samples: (n_samples, num_inputs) boolean array
        """
        if distribution == 'enum':
            # Full enumeration if n_samples >= 2^num_inputs
            total_inputs = 2 ** self.num_inputs
            if self.n_samples >= total_inputs:
                samples = np.zeros((total_inputs, self.num_inputs), dtype=bool)
                for i in range(total_inputs):
                    for j in range(self.num_inputs):
                        samples[i, j] = (i >> j) & 1
                return samples

        # Uniform random sampling
        samples = self.rng.randint(0, 2, size=(self.n_samples, self.num_inputs), dtype=np.uint8)
        return samples.astype(bool)

    def simulate_circuit(self, root: Union[NodeWithInv, Node], samples: np.ndarray, cache: Optional[Dict] = None) -> np.ndarray:
        """
        Simulate circuit on Monte Carlo samples

        Args:
            root: Circuit root node
            samples: (n_samples, num_inputs) input samples
            cache: Optional external cache dictionary (key: NodeWithInv, value: bitarray.bitarray)

        Returns:
            outputs: (n_samples,) output values
        """
        if type(root) is Node:
            root = NodeWithInv(root, inverted=False)

        # Cache for intermediate results (temporary cache for this simulation)
        node_cache = {}

        def simulate_node_rec(node: NodeWithInv) -> np.ndarray:
            """Recursively simulate node on samples"""
            # Check external cache first (if provided) - similar to compute_tt
            if isinstance(cache, dict) and node in cache:
                # Cache contains bitarray, convert to numpy array
                cached_tt = cache[node]
                if isinstance(cached_tt, bitarray.bitarray):
                    return np.array(cached_tt.tolist(), dtype=bool)
                elif isinstance(cached_tt, np.ndarray):
                    return cached_tt
                else:
                    # Fallback: try to convert
                    return np.array(cached_tt, dtype=bool)
            
            # Check internal cache
            cache_key = (id(node.parent), node.inverted)
            if cache_key in node_cache:
                return node_cache[cache_key]

            if node is None:
                raise ValueError('node is None')

            if node.is_leaf():
                if node.var >= 0:
                    # Input variable
                    result = samples[:, node.var].copy()
                elif node.var == -1:
                    # Constant 0
                    result = np.zeros(len(samples), dtype=bool)
                else:
                    raise ValueError(f'node.var = {node.var}, not supported')
            else:
                # Check if parent node is in cache (for exact TT computation compatibility)
                # Similar to compute_tt: elif isinstance(cache, dict) and root.parent in cache
                if isinstance(cache, dict) and node.parent in cache:
                    # Use cached parent result
                    cached_tt = cache[node.parent]
                    if isinstance(cached_tt, bitarray.bitarray):
                        result = np.array(cached_tt.tolist(), dtype=bool)
                    elif isinstance(cached_tt, np.ndarray):
                        result = cached_tt
                    else:
                        result = np.array(cached_tt, dtype=bool)
                else:
                    # AND gate - recursively compute
                    left_output = simulate_node_rec(node.left)
                    right_output = simulate_node_rec(node.right)
                    result = left_output & right_output

            # Apply inversion
            if node.inverted:
                result = ~result

            # Cache result in internal cache
            node_cache[cache_key] = result
            return result

        return simulate_node_rec(root)

    def compute_approximate_tt(self,
                              root: Union[NodeWithInv, Node],
                              distribution: str = 'uniform',
                              return_confidence: bool = False,
                              samples: Optional[np.ndarray] = None,
                              cache: Optional[Dict] = None) -> Union[bitarray.bitarray, tuple]:
        """
        Compute approximate truth table using Monte Carlo sampling
        
        Returns a truth table of length n_samples (not 2^num_inputs).
        This is the true Monte Carlo approach - only storing sampled patterns.

        Args:
            root: Circuit root node
            distribution: Sampling distribution ('uniform' or 'enum')
            return_confidence: If True, return confidence scores for each bit
            samples: Optional pre-generated samples (for consistency across multiple calls)
            cache: Optional external cache dictionary (key: NodeWithInv, value: bitarray.bitarray)

        Returns:
            tt: Approximate truth table of length n_samples
            confidence: (optional) Confidence scores for each bit
        """
        if type(root) is Node:
            root = NodeWithInv(root, inverted=False)
        
        # Check cache first - similar to compute_tt
        if isinstance(cache, dict) and root in cache:
            cached_tt = cache[root]
            if isinstance(cached_tt, bitarray.bitarray):
                if return_confidence:
                    confidence_scores = np.ones(len(cached_tt), dtype=np.float32)
                    return cached_tt, confidence_scores
                return cached_tt
        
        # Generate or use provided samples
        if samples is None:
            samples = self.generate_samples(distribution)
        else:
            # Validate samples shape
            if samples.shape[1] != self.num_inputs:
                raise ValueError(f"Samples shape mismatch: expected {self.num_inputs} inputs, got {samples.shape[1]}")
            if len(samples) != self.n_samples:
                # Allow different sample count, but warn
                import warnings
                warnings.warn(f"Sample count mismatch: expected {self.n_samples}, got {len(samples)}")

        # Simulate circuit on samples
        outputs = self.simulate_circuit(root, samples, cache=cache)

        # Build truth table directly from outputs (length = n_samples)
        # This is the true Monte Carlo approach - only storing sampled patterns
        tt = bitarray.bitarray(outputs.tolist())
        
        # Store in cache if provided (similar to how environment stores results)
        # Note: cache is typically managed by the caller, but we can store here for consistency
        if isinstance(cache, dict):
            cache[root] = tt
        
        if return_confidence:
            # For confidence, we can compute how consistent the output is for repeated patterns
            # But for simplicity, we'll use a fixed confidence (can be improved later)
            confidence_scores = np.ones(len(samples), dtype=np.float32)  # All sampled patterns have confidence 1.0
            return tt, confidence_scores
        
        return tt

    def compute_approximate_tts(self,
                               roots: Union[NodeWithInv, Node, List[Union[NodeWithInv, Node]]],
                               distribution: str = 'uniform',
                               return_confidence: bool = False,
                               samples: Optional[np.ndarray] = None,
                               cache: Optional[Dict] = None) -> Union[List[bitarray.bitarray], tuple]:
        """
        Compute approximate truth tables for multiple outputs
        
        All outputs use the same samples for consistency.

        Args:
            roots: Circuit root nodes (single root or list of roots)
            distribution: Sampling distribution
            return_confidence: If True, return confidence scores
            samples: Optional pre-generated samples (for consistency with input_tt)
            cache: Optional external cache dictionary (key: NodeWithInv, value: bitarray.bitarray)

        Returns:
            tts: List of approximate truth tables (all of length n_samples)
            confidences: (optional) List of confidence scores
        """
        # Handle single-output circuits where callers pass a single root.
        # This is common when the optimized circuit collapses to a constant node.
        if not isinstance(roots, (list, tuple)):
            roots = [roots]

        # Generate or use provided samples
        if samples is None:
            samples = self.generate_samples(distribution)
        else:
            # Validate samples shape
            if samples.shape[1] != self.num_inputs:
                raise ValueError(f"Samples shape mismatch: expected {self.num_inputs} inputs, got {samples.shape[1]}")
            if len(samples) != self.n_samples:
                # Allow different sample count, but warn
                import warnings
                warnings.warn(f"Sample count mismatch: expected {self.n_samples}, got {len(samples)}")
        
        if return_confidence:
            tts = []
            confidences = []
            for root in roots:
                tt, conf = self.compute_approximate_tt(root, distribution, return_confidence=True, samples=samples, cache=cache)
                tts.append(tt)
                confidences.append(conf)
            return tts, confidences
        else:
            return [self.compute_approximate_tt(root, distribution, return_confidence=False, samples=samples, cache=cache) for root in roots]

    def estimate_error_bound(self, n_samples: Optional[int] = None) -> float:
        """
        Estimate the error bound using Hoeffding's inequality

        For Monte Carlo sampling with n samples, the error bound is approximately:
        ε ≈ sqrt(ln(2/δ) / (2*n))

        where δ is the confidence level (e.g., 0.05 for 95% confidence)

        Args:
            n_samples: Number of samples (defaults to self.n_samples)

        Returns:
            error_bound: Estimated error bound
        """
        if n_samples is None:
            n_samples = self.n_samples

        # Use 95% confidence (δ = 0.05)
        delta = 0.05
        error_bound = np.sqrt(np.log(2.0 / delta) / (2.0 * n_samples))
        return error_bound


# Convenience functions for backward compatibility
def compute_tt_approximate(root: Union[NodeWithInv, Node],
                          num_inputs: int,
                          n_samples: int = 256000,
                          seed: Optional[int] = None,
                          input_tt: Optional[List[bitarray.bitarray]] = None,
                          cache: Optional[Dict] = None) -> bitarray.bitarray:
    """
    Compute approximate truth table using Monte Carlo sampling

    This is a drop-in replacement for compute_tt() in utils.py for large circuits.

    Args:
        root: Circuit root node
        num_inputs: Number of inputs
        n_samples: Number of Monte Carlo samples
        seed: Random seed
        input_tt: Optional pre-computed input truth tables (extracts samples from it)
        cache: Optional external cache dictionary (key: NodeWithInv, value: bitarray.bitarray)

    Returns:
        Approximate truth table
    """
    # Extract samples from input_tt if provided
    samples = None
    if input_tt is not None and len(input_tt) > 0:
        # Extract samples from input_tt (first positive literal contains the samples)
        # input_tt format: [pos_0, neg_0, pos_1, neg_1, ...]
        # Extract from pos_0 (index 0)
        if len(input_tt[0]) == n_samples:
            # Reconstruct samples from input_tt
            samples = np.zeros((n_samples, num_inputs), dtype=bool)
            for i in range(num_inputs):
                if i * 2 < len(input_tt):
                    samples[:, i] = np.array(input_tt[i * 2].tolist(), dtype=bool)
    
    mc_tt = MonteCarloTT(num_inputs, n_samples, seed)
    return mc_tt.compute_approximate_tt(root, samples=samples, cache=cache)


def compute_tts_approximate(roots: Union[NodeWithInv, Node, List[Union[NodeWithInv, Node]]],
                           num_inputs: int,
                           n_samples: int = 256000,
                           seed: Optional[int] = None,
                           input_tt: Optional[List[bitarray.bitarray]] = None,
                           cache: Optional[Dict] = None) -> List[bitarray.bitarray]:
    """
    Compute approximate truth tables for multiple outputs

    Args:
        roots: Circuit root nodes (single root or list of roots)
        num_inputs: Number of inputs
        n_samples: Number of Monte Carlo samples
        seed: Random seed
        input_tt: Optional pre-computed input truth tables (extracts samples from it)
        cache: Optional external cache dictionary (key: NodeWithInv, value: bitarray.bitarray)

    Returns:
        List of approximate truth tables
    """
    # Extract samples from input_tt if provided
    # print(seed)
    samples = None
    if input_tt is not None and len(input_tt) > 0:
        # Extract samples from input_tt (first positive literal contains the samples)
        # input_tt format: [pos_0, neg_0, pos_1, neg_1, ...]
        # Extract from pos_0 (index 0)
        if len(input_tt[0]) == n_samples:
            # Reconstruct samples from input_tt
            samples = np.zeros((n_samples, num_inputs), dtype=bool)
            for i in range(num_inputs):
                if i * 2 < len(input_tt):
                    samples[:, i] = np.array(input_tt[i * 2].tolist(), dtype=bool)
    
    mc_tt = MonteCarloTT(num_inputs, n_samples, seed)
    return mc_tt.compute_approximate_tts(roots, samples=samples, cache=cache)


# Adaptive computation: automatically choose exact or approximate based on circuit size
def compute_tt_adaptive(root: Union[NodeWithInv, Node],
                       num_inputs: int,
                       input_tt: Optional[List[bitarray.bitarray]] = None,
                       threshold: int = 12,
                       n_samples: int = 256000,
                       seed: Optional[int] = None,
                       cache: Optional[Dict] = None) -> bitarray.bitarray:
    """
    Adaptively compute truth table: exact for small circuits, approximate for large ones

    Args:
        root: Circuit root node
        num_inputs: Number of inputs
        input_tt: Pre-computed input truth tables (for exact computation or extracting samples)
        threshold: Use approximate computation if num_inputs > threshold
        n_samples: Number of Monte Carlo samples for approximate computation
        seed: Random seed
        cache: Optional external cache dictionary (key: NodeWithInv, value: bitarray.bitarray)

    Returns:
        Truth table (exact or approximate)
    """
    if num_inputs <= threshold:
        # Use exact computation for small circuits
        from src.utils import compute_tt
        return compute_tt(root, num_inputs=num_inputs, input_tt=input_tt, cache=cache)
    else:
        # Use approximate computation for large circuits
        # Pass input_tt to extract samples from it (ensures consistency)
        return compute_tt_approximate(root, num_inputs, n_samples, seed, input_tt=input_tt, cache=cache)


def compute_tts_adaptive(roots: Union[NodeWithInv, Node, List[Union[NodeWithInv, Node]]],
                        num_inputs: int,
                        input_tt: Optional[List[bitarray.bitarray]] = None,
                        threshold: int = 12,
                        n_samples: int = 256000,
                        seed: Optional[int] = None,
                        cache: Optional[Dict] = None) -> List[bitarray.bitarray]:
    """
    Adaptively compute truth tables for multiple outputs

    Args:
        roots: Circuit root nodes (single root or list of roots)
        num_inputs: Number of inputs
        input_tt: Pre-computed input truth tables (for exact computation or extracting samples)
        threshold: Use approximate computation if num_inputs > threshold
        n_samples: Number of Monte Carlo samples for approximate computation
        seed: Random seed
        cache: Optional external cache dictionary (key: NodeWithInv, value: bitarray.bitarray)

    Returns:
        List of truth tables (exact or approximate)
    """
    if not isinstance(roots, (list, tuple)):
        roots = [roots]

    # Note: Monte Carlo approximation now returns sampled truth tables (length n_samples)
    # instead of full truth tables (length 2^num_inputs), so memory is much more manageable
    # No need for strict memory limits when using Monte Carlo
    
    if num_inputs <= threshold:
        # Use exact computation for small circuits
        from src.utils import compute_tts
        return compute_tts(roots, num_inputs=num_inputs, input_tt=input_tt)
    else:
        # Use approximate computation for large circuits
        # Pass input_tt to extract samples from it (ensures consistency)
        return compute_tts_approximate(roots, num_inputs, n_samples, seed, input_tt=input_tt, cache=cache)

def generate_input_samples(num_inputs: int, num_samples: int = 8192, seed: int = None):
    """
    Generate random input samples for Monte Carlo simulation.

    Args:
        num_inputs: Number of circuit inputs
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        numpy array of shape (num_samples, num_inputs) with boolean values
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random binary samples
    samples = np.random.randint(0, 2, size=(num_samples, num_inputs), dtype=bool)

    # For smaller inputs, we can include some structured patterns
    if num_inputs <= 8 and num_samples >= 256:
        # Include all possible patterns for small circuits (if space permits)
        all_patterns = np.array([[bool(int(x)) for x in format(i, f'0{num_inputs}b')]
                                 for i in range(2**num_inputs)])

        # Replace first 2^num_inputs samples with all patterns
        samples[:len(all_patterns)] = all_patterns

    return samples


def compute_input_tt_approximate(num_inputs: int, num_samples: int = 8192, seed: Optional[int] = None):
    """
    Generate input truth tables using Monte Carlo sampling.
    
    Returns truth tables of length num_samples (not 2^num_inputs).
    This matches the sampling approach used for output truth tables.

    Args:
        num_inputs: Number of circuit inputs
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility (ensures same samples as output TT)

    Returns:
        List of bitarrays, one for each input (positive and negative), each of length num_samples
    """
    # Ensure num_samples is multiple of 64 for efficiency (same as MonteCarloTT)
    # This ensures consistency with MonteCarloTT.generate_samples
    # print(seed)
    adjusted_num_samples = num_samples
    if num_samples % 64 != 0:
        adjusted_num_samples = ((num_samples // 64) + 1) * 64
        if seed is not None:  # Only warn if seed is provided (for reproducibility)
            import warnings
            warnings.warn(f"num_samples adjusted from {num_samples} to {adjusted_num_samples} (multiple of 64) for consistency with MonteCarloTT")
    
    # Generate samples (same as used for output truth tables)
    # Use the same sampling logic as MonteCarloTT.generate_samples
    rng = np.random.RandomState(seed)
    samples = rng.randint(0, 2, size=(adjusted_num_samples, num_inputs), dtype=np.uint8).astype(bool)
    
    # Build input truth tables from samples
    input_tt = []
    for i in range(num_inputs):
        # Positive literal: values of input i across all samples
        tt_pos = bitarray.bitarray(samples[:, i].tolist())
        input_tt.append(tt_pos)
        
        # Negative literal: inverted values
        tt_neg = tt_pos.copy()
        tt_neg.invert()
        input_tt.append(tt_neg)
    
    # Note: input_tt length is adjusted_num_samples, not num_samples
    # This ensures consistency with MonteCarloTT output TT length
    return input_tt


def calculate_circuit_error_rate(
    roots_a: List[Union[Node, NodeWithInv]], 
    roots_b: List[Union[Node, NodeWithInv]], 
    num_inputs: int,
    tt_num_samples: int = 4096,
    tt_seed: int = 42,
    threshold: int = 12,
    input_tt: Optional[List[bitarray.bitarray]] = None,
    verbose: bool = False
) -> float:

    if input_tt is None:
        if num_inputs > threshold:
            if verbose:
                print(f"[Info] Inputs ({num_inputs}) > {threshold}, using Monte Carlo input TT (n={tt_num_samples}).")
            input_tt = compute_input_tt_approximate(num_inputs, tt_num_samples, seed=tt_seed)
        else:
            if verbose:
                print(f"[Info] Inputs ({num_inputs}) <= {threshold}, using Exact input TT.")
            input_tt = compute_input_tt(num_inputs)


    if verbose:
        print("[Info] Computing TTs for Circuit A...")
    tts_a = compute_tts_adaptive(
        roots_a, 
        num_inputs, 
        input_tt=input_tt, 
        n_samples=tt_num_samples,
        threshold=threshold,
        seed=tt_seed
    )

    if verbose:
        print("[Info] Computing TTs for Circuit B...")
    tts_b = compute_tts_adaptive(
        roots_b, 
        num_inputs, 
        input_tt=input_tt, 
        n_samples=tt_num_samples,
        threshold=threshold,
        seed=tt_seed
    )

    if len(tts_a) != len(tts_b):
        raise ValueError(f"Output count mismatch: Circuit A has {len(tts_a)} outputs, Circuit B has {len(tts_b)} outputs.")
    
    if len(tts_a) == 0:
        if verbose:
            print("[Warning] No outputs in circuits.")
        return 0.0

    length = len(tts_a[0])
    
    diff_map = bitarray.bitarray(length)
    diff_map.setall(0)

    for t1, t2 in zip(tts_a, tts_b):
        diff_map |= (t1 ^ t2)

    num_diff_samples = diff_map.count(1)
    error_rate = num_diff_samples / length

    if verbose:
        print("-" * 30)
        print(f"Circuit Equivalence Check:")
        print(f"  Inputs:          {num_inputs}")
        print(f"  Outputs:         {len(tts_a)}")
        print(f"  Samples Tested:  {length}")
        print(f"  Failed Samples:  {num_diff_samples}")
        print(f"  Error Rate:      {error_rate:.4f} ({error_rate * 100:.2f}%)")
        print("-" * 30)

    return error_rate