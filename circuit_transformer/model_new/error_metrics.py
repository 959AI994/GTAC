"""
Error Metrics Module for Approximate Circuit Transformer

Implements error measurement functions as described in the ACT paper:
- Error Rate (ER) 
- Mean Relative Error Distance (MRED)
- Mean Squared Error (MSE)
"""

import numpy as np
import bitarray
import bitarray.util
from typing import List, Tuple, Union
import sys

class ErrorMetrics:
    """
    Error metrics computation for approximate circuits
    """
    
    def __init__(self, num_inputs: int, delta: float = 1e-6):
        """
        Initialize error metrics calculator
        
        Args:
            num_inputs: Number of circuit inputs
            delta: Small constant to prevent division by zero in MRED
        """
        self.num_inputs = num_inputs
        self.delta = delta
        self.total_inputs = 2 ** num_inputs
    
    def compute_error_rate(self, 
                          g_outputs: Union[bitarray.bitarray, List[bitarray.bitarray]], 
                          f_outputs: Union[bitarray.bitarray, List[bitarray.bitarray]],
                          sample_inputs: List[int] = None) -> float:
        """
        Compute Error Rate (ER) between generated and target circuits
        
        ER = (1/2^N) * Σ I(g(x) ≠ f(x))
        
        Args:
            g_outputs: Generated circuit outputs
            f_outputs: Target circuit outputs  
            sample_inputs: Optional input subset for sampling (indices)
            
        Returns:
            Error rate in [0, 1]
        """
        if isinstance(g_outputs, bitarray.bitarray):
            g_outputs = [g_outputs]
        if isinstance(f_outputs, bitarray.bitarray):
            f_outputs = [f_outputs]
            
        if len(g_outputs) != len(f_outputs):
            raise ValueError("Number of outputs must match")
        
        total_errors = 0
        total_evaluations = 0
        mismatch_mask = bitarray.util.zeros(len(g_outputs[0]))
        
        for g_out, f_out in zip(g_outputs, f_outputs):
            if sample_inputs is not None:
                # Use subsampling for efficiency
                errors = sum(g_out[i] != f_out[i] for i in sample_inputs)
                total_errors += errors
                total_evaluations += len(sample_inputs)
            else:
                # Full evaluation
                mismatch_mask |= (g_out ^ f_out)  # XOR for difference

        total_errors = mismatch_mask.count()
        total_evaluations = len(mismatch_mask)
                # if hasattr(diff, 'count'):
                #     # bitarray has count() method
                #     total_errors += diff.count()
                # else:
                #     # numpy array or list
                #     total_errors += int(np.sum(diff))  # Count True values
                # total_evaluations += len(g_out)
        
        return total_errors / total_evaluations if total_evaluations > 0 else 0.0
    
    def compute_mred(self, 
                     g_values: Union[bitarray.bitarray, List[bitarray.bitarray]], 
                     f_values: Union[bitarray.bitarray, List[bitarray.bitarray]],
                     sample_inputs: List[int] = None) -> float:
        """
        Compute Mean Relative Error Distance (MRED)
        
        MRED = (1/2^N) * Σ |g(x) - f(x)| / max(|f(x)|, δ)
        
        Args:
            g_values: Generated circuit integer outputs
            f_values: Target circuit integer outputs
            sample_inputs: Optional input subset for sampling (indices)
            
        Returns:
            MRED value
        """
        
        if len(g_values) != len(f_values):
            raise ValueError("Output arrays must have same length")
        
        if sample_inputs is not None:
            g_vals = [g_values[i] for i in sample_inputs]
            f_vals = [f_values[i] for i in sample_inputs]
        else:
            # g_vals = [int.from_bytes(ba.tobytes(), byteorder=sys.byteorder) for ba in g_values]
            # f_vals = [int.from_bytes(ba.tobytes(), byteorder=sys.byteorder) for ba in f_values]
            g_vals = [(1 if ba1 else 0) * 2 + (1 if ba2 else 0) for ba1, ba2 in zip(g_values[1], g_values[0])]
            f_vals = [(1 if ba1 else 0) * 2 + (1 if ba2 else 0) for ba1, ba2 in zip(f_values[1], f_values[0])]
        relative_errors = []
        for g_val, f_val in zip(g_vals, f_vals):
            # For boolean values, convert to float for distance calculation
            g_float = float(g_val)
            f_float = float(f_val)
            abs_diff = abs(g_float - f_float)
            denominator = max(abs(f_float), 1)
            relative_errors.append(abs_diff / denominator)
        
        return np.mean(relative_errors) if relative_errors else 0.0
    
    def compute_mse(self, 
                     g_values: Union[bitarray.bitarray, List[bitarray.bitarray]], 
                     f_values: Union[bitarray.bitarray, List[bitarray.bitarray]],
                    sample_inputs: List[int] = None) -> float:
        """
        Compute Mean Squared Error (MSE)
        
        MSE = (1/2^N) * Σ (g(x) - f(x))²
        
        Args:
            g_values: Generated circuit integer outputs
            f_values: Target circuit integer outputs  
            sample_inputs: Optional input subset for sampling (indices)
            
        Returns:
            MSE value
        """
        if len(g_values) != len(f_values):
            raise ValueError("Output arrays must have same length")
        
        if sample_inputs is not None:
            g_vals = [g_values[i] for i in sample_inputs]
            f_vals = [f_values[i] for i in sample_inputs]
        else:
            g_vals = [(1 if ba1 else 0) * 2 + (1 if ba2 else 0) for ba1, ba2 in zip(g_values[1], g_values[0])]
            f_vals = [(1 if ba1 else 0) * 2 + (1 if ba2 else 0) for ba1, ba2 in zip(f_values[1], f_values[0])]
        
        squared_errors = [(float(g_val) - float(f_val)) ** 2 for g_val, f_val in zip(g_vals, f_vals)]
        return np.mean(squared_errors) if squared_errors else 0.0
    
    def generate_sample_inputs(self, sample_size: int = 1024, seed: int = None) -> List[int]:
        """
        Generate random input indices for subsampling
        
        Args:
            sample_size: Number of samples (K ≪ 2^N)
            seed: Random seed for reproducibility
            
        Returns:
            List of input indices
        """
        if seed is not None:
            np.random.seed(seed)
        
        max_samples = min(sample_size, self.total_inputs)
        return np.random.choice(self.total_inputs, max_samples, replace=False).tolist()
    
    def hoeffding_confidence_bound(self, sample_size: int, confidence: float = 0.95) -> float:
        """
        Compute confidence bound using Hoeffding's inequality
        
        Args:
            sample_size: Number of samples used
            confidence: Desired confidence level
            
        Returns:
            Error bound epsilon
        """
        alpha = 1 - confidence
        return np.sqrt(-np.log(alpha / 2) / (2 * sample_size))


# Convenience functions
def compute_error_rate(g_outputs, f_outputs, num_inputs: int, sample_inputs=None):
    """Convenience function for error rate computation"""
    metrics = ErrorMetrics(num_inputs)
    return metrics.compute_error_rate(g_outputs, f_outputs, sample_inputs)

def compute_mred(g_values, f_values, num_inputs: int, sample_inputs=None, delta=1e-6):
    """Convenience function for MRED computation"""  
    metrics = ErrorMetrics(num_inputs, delta)
    return metrics.compute_mred(g_values, f_values, sample_inputs)

def compute_mse(g_values, f_values, num_inputs: int, sample_inputs=None):
    """Convenience function for MSE computation"""
    metrics = ErrorMetrics(num_inputs)
    return metrics.compute_mse(g_values, f_values, sample_inputs)