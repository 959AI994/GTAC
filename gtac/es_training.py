"""
TensorFlow-based Evolution Strategies (ES) implementation for Circuit Transformer fine-tuning.

This module adapts the ES algorithm from the paper "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning"
to work with TensorFlow and circuit optimization tasks.

Key adaptations:
1. TensorFlow-based implementation instead of PyTorch
2. Circuit-specific reward function instead of text-based rewards
3. Integration with existing CircuitTransformer architecture
4. Support for circuit optimization metrics (area, delay, error rate)
"""

import os
import time
import datetime
import numpy as np
import tensorflow as tf
import tf_keras as keras
import keras.backend as K
import gc
import json
import shutil
import errno
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue, Value, Array
import copy

from gtac.model import CircuitTransformerRefactored
from gtac.utils import *
from gtac.environment import LogicNetworkEnv
from gtac.encoding import node_to_int, int_to_node, encode_aig, stack_to_encoding, deref_node


def check_disk_space(min_space_gb=1.0):
    """Check if disk space is enough"""
    try:
        stat = os.statvfs('.')
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        return available_gb >= min_space_gb, available_gb
    except:
        return True, 0.0  # If check fails, assume space is enough


def wait_for_disk_space(min_space_gb=1.0, max_wait_time=3600, check_interval=60):
    """Wait for disk space to be enough"""
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        has_space, available_gb = check_disk_space(min_space_gb)
        if has_space:
            print(f"Disk space enough ({available_gb:.1f}GB)，continue training")
            return True

        print(f"Disk space not enough ({available_gb:.1f}GB < {min_space_gb}GB)，wait {check_interval} seconds and try again...")
        time.sleep(check_interval)

    print(f"Wait {max_wait_time} seconds for disk space, but it's still not enough, give up")
    return False


def safe_file_operation(operation_func, *args, **kwargs):
    """Safe file operation, handle disk space not enough error"""
    max_retries = 3
    retry_delay = 10

    for attempt in range(max_retries):
        try:
            return operation_func(*args, **kwargs)
        except OSError as e:
            if e.errno == errno.ENOSPC:  # No space left on device
                print(f"Disk space not enough, try {attempt+1} times...")
                if attempt < max_retries - 1:
                    # Wait for disk space
                    if wait_for_disk_space(min_space_gb=2.0, max_wait_time=300, check_interval=30):
                        time.sleep(retry_delay)
                        continue
                    else:
                        print("Disk space wait timeout, give up")
                        break
                else:
                    print("Reach maximum retry times, give up")
                    break
            else:
                raise e
    
    return None


class ES_Trainer:
    """
    Evolution Strategies trainer for Circuit Transformer fine-tuning.
    
    This class implements the core ES algorithm adapted for TensorFlow and circuit optimization.
    """
    
    def __init__(self,
                 model,
                 num_iterations: int = 1000,
                 population_size: int = 30,
                 num_perturbations: int = None,
                 num_parallel_environments: int = None,
                 sigma: float = 0.001,
                 alpha: float = 0.0005,
                 max_seq_length: int = 200,
                 error_rate_tolerance: float = 0.1,
                 w_area: float = 1.4,
                 w_delay: float = 0.3,
                 w_error: float = 0.3,
                 verbose: bool = True,
                 save_frequency: int = 100,
                 initial_seed: int = 33,
                 finetune_decoder_only: bool = False,
                 num_processes: int = 1,
                 gpu_threads_per_process: int = 1,
                 log_file_name: str = "es_train_log.txt",
                 save_model: bool = True,
                 save_final_only: bool = False,
                 enable_log_print: bool = True):
        """
        Initialize ES trainer.
        
        Args:
            model: CircuitTransformer model instance
            num_iterations: Number of ES iterations (generations)
            population_size: Population size (legacy parameter, used as default for backward compatibility)
            num_perturbations: Number of weight perturbations per iteration (default: population_size)
            num_parallel_environments: Number of parallel circuit environments to evaluate (default: population_size)
            sigma: Standard deviation for weight perturbations (noise scale)
            alpha: Learning rate for weight updates
            max_seq_length: Maximum sequence length for circuit generation
            error_rate_tolerance: Maximum allowed error rate for circuits
            w_area: Weight for area optimization
            w_delay: Weight for delay optimization
            w_error: Weight for error rate penalty
            verbose: Whether to print detailed logs
            save_frequency: Frequency to save model checkpoints
            initial_seed: Initial random seed
            finetune_decoder_only: Whether to fine-tune only decoder layers
            num_processes: Number of parallel processes for evaluation
            gpu_threads_per_process: Number of threads per GPU process
            log_file_name: Log file name for training output
            save_model: Whether to save model weights and checkpoints
            save_final_only: Whether to save model only at the final iteration (overrides save_frequency)
            enable_log_print: Whether to write print output to log file
        """
        self.model = model
        self.num_iterations = num_iterations
        self.population_size = population_size  # Keep for backward compatibility
        
        # Set up separated parameters
        self.num_perturbations = num_perturbations if num_perturbations is not None else population_size
        self.num_parallel_environments = num_parallel_environments if num_parallel_environments is not None else population_size
        
        self.sigma = sigma
        self.alpha = alpha
        self.max_seq_length = max_seq_length
        self.error_rate_tolerance = error_rate_tolerance
        self.w_area = w_area
        self.w_delay = w_delay
        self.w_error = w_error
        self.verbose = verbose
        self.save_frequency = save_frequency
        self.initial_seed = initial_seed
        self.finetune_decoder_only = finetune_decoder_only
        self.num_processes = num_processes
        self.gpu_threads_per_process = gpu_threads_per_process
        self.log_file_name = log_file_name
        self.save_model = save_model
        self.save_final_only = save_final_only
        self.enable_log_print = enable_log_print
        
        # Initialize random number generator
        self.rng = np.random.RandomState(initial_seed)
        
        # Thread lock for model access
        self.model_lock = threading.Lock()
        
        # Training statistics
        self.training_stats = {
            'iteration': [],
            'mean_reward': [],
            'min_reward': [],
            'max_reward': [],
            'std_reward': [],
            'best_reward': -float('inf'),
            'best_iteration': 0,
            'val_mean_reward': [],
            'val_area_improvement': [],
            'val_delay_improvement': [],
            'val_error_rate': []
        }
        
        # Generate timestamp for unique file identification
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup log file
        if self.log_file_name == "es_train_log.txt":  # Use default filename
            self.log_file_name = f"es_train_log_{timestamp}.txt"
        
        # Fix log file path problem: if log_file_name already contains full path, use it directly; otherwise add log/ prefix
        if self.log_file_name.startswith("log/") or "/" in self.log_file_name:
            self.log_file_path = self.log_file_name
        else:
            self.log_file_path = "log/" + self.log_file_name
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = open(self.log_file_path, "a")
        
        # Store safe_log_write as a method for use throughout the class
        def safe_log_write(message):
            """Safely write to log file, handling disk space errors gracefully with retry mechanism."""
            import time
            import shutil
            
            # Configuration for retry behavior
            max_total_wait_time = 10 * 60  # 10 minutes in seconds
            initial_retry_delay = 5.0  # seconds
            max_retry_delay = 30.0  # seconds
            retry_delay = initial_retry_delay
            total_wait_time = 0.0
            attempt = 0
            
            while total_wait_time < max_total_wait_time:
                try:
                    self.log_file.write(message)
                    self.log_file.flush()
                    return  # Success, exit the function
                except OSError as e:
                    if e.errno == 28:  # No space left on device
                        attempt += 1
                        remaining_time = max_total_wait_time - total_wait_time
                        
                        if remaining_time > retry_delay:
                            # Check available disk space
                            try:
                                total, used, free = shutil.disk_usage(self.log_file_path)
                                free_gb = free / (1024**3)
                                print(f"WARNING: Disk space full (attempt {attempt}), "
                                      f"free space: {free_gb:.2f}GB, waiting {retry_delay:.1f}s before retry... "
                                      f"(total wait time: {total_wait_time/60:.1f}min/{max_total_wait_time/60:.1f}min)")
                            except:
                                print(f"WARNING: Disk space full (attempt {attempt}), "
                                      f"waiting {retry_delay:.1f}s before retry... "
                                      f"(total wait time: {total_wait_time/60:.1f}min/{max_total_wait_time/60:.1f}min)")
                            
                            time.sleep(retry_delay)
                            total_wait_time += retry_delay
                            # Exponential backoff with jitter to avoid thundering herd
                            retry_delay = min(retry_delay * 1.5, max_retry_delay)
                        else:
                            print(f"ERROR: Disk space full after waiting {total_wait_time/60:.1f} minutes, skipping log write: {message.strip()}")
                            return  # Give up after max wait time
                    else:
                        # Re-raise other OSErrors
                        raise
            
            # This should not be reached, but just in case
            print(f"ERROR: Disk space full after waiting {max_total_wait_time/60:.1f} minutes, skipping log write: {message.strip()}")
            return
        
        self.safe_log_write = safe_log_write
        
        # Create a custom print function that writes to both console and log
        def log_print(*args, **kwargs):
            """Print to both console and optionally to log file."""
            # Print to console
            print(*args, **kwargs)
            # Write to log file only if enabled
            if self.enable_log_print:
                message = ' '.join(str(arg) for arg in args)
                self.safe_log_write(message + '\n')
        
        # Store the log_print function for use throughout the class
        self.log_print = log_print
        
        # Create save directory
        self.save_dir = f"es_checkpoints_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Record all training parameters to log file
        def write_training_params():
            self.safe_log_write("=" * 80 + "\n")
            self.safe_log_write("ES Training Parameters\n")
            self.safe_log_write("=" * 80 + "\n")
            self.safe_log_write(f"num_iterations: {self.num_iterations}\n")
            self.safe_log_write(f"population_size: {self.population_size}\n")
            self.safe_log_write(f"num_perturbations: {self.num_perturbations}\n")
            self.safe_log_write(f"num_parallel_environments: {self.num_parallel_environments}\n")
            self.safe_log_write(f"sigma: {self.sigma}\n")
            self.safe_log_write(f"alpha: {self.alpha}\n")
            self.safe_log_write(f"max_seq_length: {self.max_seq_length}\n")
            self.safe_log_write(f"error_rate_tolerance: {self.error_rate_tolerance}\n")
            self.safe_log_write(f"w_area: {self.w_area}\n")
            self.safe_log_write(f"w_delay: {self.w_delay}\n")
            self.safe_log_write(f"w_error: {self.w_error}\n")
            self.safe_log_write(f"verbose: {self.verbose}\n")
            self.safe_log_write(f"save_frequency: {self.save_frequency}\n")
            self.safe_log_write(f"initial_seed: {self.initial_seed}\n")
            self.safe_log_write(f"finetune_decoder_only: {self.finetune_decoder_only}\n")
            self.safe_log_write(f"num_processes: {self.num_processes}\n")
            self.safe_log_write(f"gpu_threads_per_process: {self.gpu_threads_per_process}\n")
            self.safe_log_write(f"log_file_name: {self.log_file_name}\n")
            self.safe_log_write(f"save_model: {self.save_model}\n")
            self.safe_log_write(f"save_dir: {self.save_dir}\n")
            self.safe_log_write("=" * 80 + "\n")
        
        safe_file_operation(write_training_params)
        
        if self.verbose:
            self.log_print(f"ES Trainer initialized:")
            self.log_print(f"  - Iterations: {self.num_iterations}")
            self.log_print(f"  - Population size: {self.population_size}")
            self.log_print(f"  - Number of perturbations: {self.num_perturbations}")
            self.log_print(f"  - Number of parallel environments: {self.num_parallel_environments}")
            self.log_print(f"  - Sigma: {self.sigma}")
            self.log_print(f"  - Alpha: {self.alpha}")
            self.log_print(f"  - Processes: {self.num_processes}")
            self.log_print(f"  - GPU threads per process: {self.gpu_threads_per_process}")
            self.log_print(f"  - Total parallel evaluations: {self.num_processes * self.gpu_threads_per_process}")
            self.log_print(f"  - Log file: {self.log_file_path}")
            self.log_print(f"  - Save model: {self.save_model}")
            self.log_print(f"  - Save final only: {self.save_final_only}")
            self.log_print(f"  - Save directory: {self.save_dir}")
    
    def compute_circuit_reward(self,
                              original_aig: Any,
                              optimized_aig: Any,
                              baseline_ands: int,
                              optimized_ands: int) -> float:
        """
        Compute reward for circuit optimization.

        Args:
            original_aig: Original circuit AIG
            optimized_aig: Optimized circuit AIG
            baseline_ands: Baseline number of AND gates (reference for improvement)
            optimized_ands: Number of AND gates in optimized circuit

        Returns:
            Reward value (higher is better)
        """
        try:
            # Calculate error rate
            error_rate = checkER(original_aig, optimized_aig)
            
            # Calculate delay improvement
            original_ands = count_num_ands(original_aig)
            original_delay = compute_critical_path(original_aig)
            optimized_delay = compute_critical_path(optimized_aig)
            
            # Calculate area improvement
            # area_improvement = (baseline_ands - optimized_ands) / max(baseline_ands, 1)
            area_improvement = (original_ands - optimized_ands) / max(original_ands, 1)
            
            # Calculate delay improvement
            delay_improvement = (original_delay - optimized_delay) / max(original_delay, 1)
            
            # Calculate error penalty
            error_penalty = max(0, error_rate - self.error_rate_tolerance)
            
            # Compute composite reward
            reward = (self.w_area * area_improvement + 
                     self.w_delay * delay_improvement - 
                     self.w_error * error_penalty)
            
            return reward
            
        except (ValueError, AttributeError) as e:
            # If circuit is incomplete or invalid, return negative reward
            if self.verbose:
                self.log_print(f"Circuit evaluation failed: {e}")
            return -1.0
    
    def _evaluate_single_circuit(self, original_aig, baseline_area):
        """
        Evaluate a single circuit with the current model state.
        
        Args:
            original_aig: Original circuit AIG
            baseline_area: Baseline area for comparison
            
        Returns:
            Reward value for this circuit
        """
        try:
            # Use thread lock to prevent concurrent model access
            with self.model_lock:
                # Clear any existing model state to avoid conflicts
                if hasattr(self.model, '_transformer') and hasattr(self.model._transformer, 'reset_cache'):
                    self.model._transformer.reset_cache()
                
                # Generate optimized circuit using current model
                optimized_aig = self.model.optimize([original_aig],
                                                   error_rate_threshold=self.error_rate_tolerance)[0]

            # Calculate metrics (outside lock to avoid blocking)
            original_ands = count_num_ands(original_aig)
            optimized_ands = count_num_ands(optimized_aig)

            # Compute reward (baseline_area is used as the reference for improvement)
            reward = self.compute_circuit_reward(original_aig, optimized_aig,
                                               baseline_area, optimized_ands)

            return reward

        except Exception as e:
            if self.verbose:
                self.log_print(f"Evaluation failed for circuit with {len(original_aig)} nodes: {e}")
                # Print more detailed error information
                import traceback
                self.log_print(f"Error details: {traceback.format_exc()}")
            return -1.0
    
    def evaluate_model_with_perturbation(self,
                                       model_weights: List[tf.Variable],
                                       perturbation_seed: int,
                                       dataset: List[Tuple[Any, int]]) -> float:
        """
        Evaluate model with weight perturbation on a dataset using batch processing.

        Args:
            model_weights: List of model weight variables
            perturbation_seed: Random seed for perturbation
            dataset: List of (original_aig, baseline_area) tuples

        Returns:
            Average reward across dataset
        """
        # Set random seed for reproducible perturbation
        tf.random.set_seed(perturbation_seed)

        # Apply perturbation to weights
        original_weights = []
        total_noise_magnitude = 0.0
        for weight in model_weights:
            original_weights.append(weight.numpy().copy())
            noise = tf.random.normal(weight.shape, stddev=self.sigma, dtype=weight.dtype)
            weight.assign_add(noise)
            total_noise_magnitude += tf.reduce_sum(tf.abs(noise))
        
        if self.verbose:
            self.log_print(f"    Applied noise with magnitude: {total_noise_magnitude:.6f}")
            self.log_print(f"    Sigma: {self.sigma}, Number of weights: {len(model_weights)}")

        # Extract circuits and baseline areas
        circuits = [original_aig for original_aig, _ in dataset]
        baseline_areas = [baseline_area for _, baseline_area in dataset]

        # Evaluate model on dataset using batch processing
        total_reward = 0.0
        valid_evaluations = 0
        
        try:
            # Use thread lock to prevent concurrent model access
            with self.model_lock:
                # Clear any existing model state
                if hasattr(self.model, '_transformer') and hasattr(self.model._transformer, 'reset_cache'):
                    self.model._transformer.reset_cache()
                
                # Batch optimize all circuits at once
                optimized_circuits = self.model.optimize(circuits, error_rate_threshold=self.error_rate_tolerance)
            
            # Calculate rewards for each circuit
            for i, (original_aig, optimized_aig, baseline_area) in enumerate(zip(circuits, optimized_circuits, baseline_areas)):
                try:
                    # Calculate metrics
                    original_ands = count_num_ands(original_aig)
                    optimized_ands = count_num_ands(optimized_aig)

                    # Compute reward
                    reward = self.compute_circuit_reward(original_aig, optimized_aig,
                                                       baseline_area, optimized_ands)
                    
                    total_reward += reward
                    valid_evaluations += 1
                    
                except Exception as e:
                    if self.verbose:
                        self.log_print(f"Reward calculation failed for circuit {i}: {e}")
                    continue

        except Exception as e:
            if self.verbose:
                self.log_print(f"    Batch evaluation failed: {e}")
                import traceback
                self.log_print(f"    Error details: {traceback.format_exc()}")
            # Fallback to individual evaluation
            return self._evaluate_individual_circuits(model_weights, perturbation_seed, dataset)

        # Restore original weights
        for weight, original_weight in zip(model_weights, original_weights):
            weight.assign(original_weight)

        # Return average reward
        avg_reward = total_reward / max(valid_evaluations, 1)
        if self.verbose:
            self.log_print(f"    Valid evaluations: {valid_evaluations}/{len(dataset)}")
            self.log_print(f"    Average reward: {avg_reward:.6f}")
        return avg_reward
    
    def _evaluate_individual_circuits(self,
                                    model_weights: List[tf.Variable],
                                    perturbation_seed: int,
                                    dataset: List[Tuple[Any, int]]) -> float:
        """
        Fallback method: Evaluate circuits individually with thread safety.
        """
        total_reward = 0.0
        valid_evaluations = 0

        for original_aig, baseline_area in dataset:
            try:
                with self.model_lock:
                    if hasattr(self.model, '_transformer') and hasattr(self.model._transformer, 'reset_cache'):
                        self.model._transformer.reset_cache()
                    
                optimized_aig = self.model.optimize([original_aig],
                                                   error_rate_threshold=self.error_rate_tolerance)[0]

                # Calculate metrics
                original_ands = count_num_ands(original_aig)
                optimized_ands = count_num_ands(optimized_aig)

                # Compute reward
                reward = self.compute_circuit_reward(original_aig, optimized_aig,
                                                   baseline_area, optimized_ands)

                total_reward += reward
                valid_evaluations += 1

            except Exception as e:
                if self.verbose:
                    self.log_print(f"Individual evaluation failed: {e}")
                continue

        return total_reward / max(valid_evaluations, 1)
    
    def evaluate_validation_set(self, val_data_dir: str) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_data_dir: Directory containing validation circuit files
            
        Returns:
            Dictionary containing validation metrics
        """
        if not val_data_dir:
            return {}
            
        val_aigs = []
        val_original_ands = []
        
        # Load validation data (same as PPO)
        for root, dirs, files in os.walk(val_data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        aig, _ = read_aiger(aiger_str=content)
                        and_num = count_num_ands(aig)
                        
                        val_aigs.append(aig)
                        val_original_ands.append(and_num)
                except Exception as e:
                    if self.verbose:
                        self.log_print(f"Failed to load validation file {file_path}: {e}")
                    continue
        
        if not val_aigs:
            if self.verbose:
                self.log_print("No valid validation circuits found")
            return {}
        
        # Evaluate on validation set (same logic as PPO)
        val_area = []
        val_err = []
        val_delay = []
        val_original_delay = []
        
        # Use deterministic optimization (same as PPO)
        # Debug: Check if model weights are being used
        if self.verbose:
            weight_norm = tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in self.model._transformer.trainable_variables])
            self.log_print(f"  Validation using model with weight norm: {weight_norm:.6f}")
        
        # Test: Use a subset of validation circuits to see if there's any change
        test_circuits = val_aigs[:5]  # Use only first 5 circuits for testing
        optimized_aigs_subset = self.model.optimize(test_circuits, error_rate_threshold=self.error_rate_tolerance)
        
        # Check if optimization results are identical
        if self.verbose:
            self.log_print(f"  Testing first 5 circuits for changes...")
            for i, (orig, opt) in enumerate(zip(test_circuits, optimized_aigs_subset)):
                orig_ands = count_num_ands(orig)
                opt_ands = count_num_ands(opt)
                self.log_print(f"    Circuit {i}: {orig_ands} -> {opt_ands} ANDs")
        
        optimized_aigs = self.model.optimize(val_aigs, error_rate_threshold=self.error_rate_tolerance)
        
        for m in range(len(val_aigs)):
            original_aig = val_aigs[m]
            optimized_aig = optimized_aigs[m]
            
            original_ands = val_original_ands[m]
            optimized_ands = count_num_ands(optimized_aig)
            
            # Calculate error rate, handle incomplete circuits (same as PPO)
            try:
                error_rate = checkER(original_aig, optimized_aig)
            except (ValueError, AttributeError) as e:
                # If circuit is incomplete, set a high error rate
                error_rate = 1.0
                if self.verbose:
                    self.log_print(f"Validation: Circuit {m} incomplete, setting error_rate = 1.0")
            
            # Calculate delay metrics (same as PPO)
            original_delay = compute_critical_path(original_aig)
            optimized_delay = compute_critical_path(optimized_aig)
            
            val_area.append(optimized_ands)
            val_err.append(error_rate)
            val_delay.append(optimized_delay)
            val_original_delay.append(original_delay)
        
        # Calculate metrics (same as PPO)
        avg_area_ori = np.mean(val_original_ands) if val_original_ands else 0
        avg_area = np.mean(val_area) if val_area else 0
        avg_err = np.mean(val_err) if val_err else 0
        avg_delay_ori = np.mean(val_original_delay) if val_original_delay else 0
        avg_delay = np.mean(val_delay) if val_delay else 0
        
        # Calculate optimization effect (same as PPO)
        total_area_improvement = avg_area_ori - avg_area
        area_improvement_rate = (total_area_improvement / avg_area_ori * 100) if avg_area_ori > 0 else 0
        total_delay_improvement = avg_delay_ori - avg_delay
        delay_improvement_rate = (total_delay_improvement / avg_delay_ori * 100) if avg_delay_ori > 0 else 0
        
        # Statistics improvement (same as PPO)
        area_improvements = [val_original_ands[i] - val_area[i] for i in range(len(val_area))]
        delay_improvements = [val_original_delay[i] - val_delay[i] for i in range(len(val_delay))]
        positive_area_improvements = [imp for imp in area_improvements if imp > 0]
        positive_delay_improvements = [imp for imp in delay_improvements if imp > 0]
        
        # Print validation results (same format as PPO)
        if self.verbose:
            self.log_print(f"Avg val_area: {avg_area:.4f} | Avg val_err: {avg_err:.4f} | Avg val_area_ori: {avg_area_ori:.4f}")
            self.log_print(f"Avg val_delay: {avg_delay:.4f} | Avg val_delay_ori: {avg_delay_ori:.4f}")
            self.log_print(f"Area Improvement: {total_area_improvement:.4f} ANDs ({area_improvement_rate:.2f}%)")
            self.log_print(f"Delay Improvement: {total_delay_improvement:.4f} levels ({delay_improvement_rate:.2f}%)")
            self.log_print(f"Positive area improvements: {len(positive_area_improvements)}/{len(val_area)} circuits")
            self.log_print(f"Positive delay improvements: {len(positive_delay_improvements)}/{len(val_delay)} circuits")
            if positive_area_improvements:
                self.log_print(f"Max area improvement: {max(positive_area_improvements):.0f} ANDs")
            if positive_delay_improvements:
                self.log_print(f"Max delay improvement: {max(positive_delay_improvements):.0f} levels")
        
        # Return comprehensive metrics (same as PPO)
        return {
            'avg_area': avg_area,
            'avg_err': avg_err,
            'avg_area_ori': avg_area_ori,
            'avg_delay': avg_delay,
            'avg_delay_ori': avg_delay_ori,
            'total_area_improvement': total_area_improvement,
            'area_improvement_rate': area_improvement_rate,
            'total_delay_improvement': total_delay_improvement,
            'delay_improvement_rate': delay_improvement_rate,
            'positive_area_improvements': len(positive_area_improvements),
            'positive_delay_improvements': len(positive_delay_improvements),
            'total_circuits': len(val_area),
            'max_area_improvement': max(positive_area_improvements) if positive_area_improvements else 0,
            'max_delay_improvement': max(positive_delay_improvements) if positive_delay_improvements else 0,
            # Keep backward compatibility
            'mean_reward': total_area_improvement + total_delay_improvement - avg_err,  # Simple reward calculation
            'area_improvement': area_improvement_rate / 100,  # Convert percentage to ratio
            'delay_improvement': delay_improvement_rate / 100,  # Convert percentage to ratio
            'error_rate': avg_err,
            'num_circuits': len(val_area)
        }
    
    def generate_perturbation_seeds(self) -> List[int]:
        """Generate random seeds for weight perturbations."""
        return self.rng.randint(0, 2**30, size=self.num_perturbations).tolist()
    
    def _process_seed_worker(self, args):
        """
        Worker function for processing a single seed in a separate process.
        This implements the core ES evaluation logic from Algorithm 2.
        
        Args:
            args: Tuple containing (seed_idx, seed, model_weights, dataset, process_id, thread_id)
            
        Returns:
            Tuple of (seed_idx, reward)
        """
        seed_idx, seed, model_weights, dataset, process_id, thread_id = args
        
        try:
            # Set random seed for reproducible perturbation
            tf.random.set_seed(seed)
            
            # Apply perturbation to weights (Algorithm 2, steps 8-9)
            original_weights = []
            for weight in model_weights:
                original_weights.append(weight.numpy().copy())
                noise = tf.random.normal(weight.shape, stddev=self.sigma, dtype=weight.dtype)
                weight.assign_add(noise)
            
            # Evaluate model with perturbed weights (Algorithm 2, step 11)
            total_reward = 0.0
            valid_evaluations = 0
            
            for original_aig, baseline_area in dataset:
                try:
                    # Use greedy decoding for deterministic evaluation
                    optimized_aig = self.model.optimize([original_aig], 
                                                      error_rate_threshold=self.error_rate_tolerance,
                                                      use_greedy_decoding=True)[0]
                    
                    # Calculate metrics
                    original_ands = count_num_ands(original_aig)
                    optimized_ands = count_num_ands(optimized_aig)
                    
                    # Compute reward
                    reward = self.compute_circuit_reward(original_aig, optimized_aig,
                                                       baseline_area, optimized_ands)
                    
                    total_reward += reward
                    valid_evaluations += 1
                    
                except Exception as e:
                    continue
            
            # Restore original weights (Algorithm 2, steps 14-15)
            tf.random.set_seed(seed)  # Reset seed for same noise
            for weight, original_weight in zip(model_weights, original_weights):
                noise = tf.random.normal(weight.shape, stddev=self.sigma, dtype=weight.dtype)
                weight.assign_sub(noise)
            
            avg_reward = total_reward / max(valid_evaluations, 1)
            return seed_idx, avg_reward
            
        except Exception as e:
            if self.verbose:
                self.log_print(f"Process {process_id} Thread {thread_id} failed for seed {seed_idx}: {e}")
            return seed_idx, -1.0
    
    def evaluate_population_parallel(self, 
                                   model_weights: List[tf.Variable],
                                   seeds: List[int],
                                   dataset: List[Tuple[Any, int]]) -> List[float]:
        """
        Evaluate population using parallel processes as described in Algorithm 2.
        
        Args:
            model_weights: List of model weight variables
            seeds: List of perturbation seeds
            dataset: List of (original_aig, baseline_area) tuples
            
        Returns:
            List of rewards corresponding to each seed (averaged across environments)
        """
        if self.num_processes == 1:
            # Fallback to sequential evaluation
            return self._evaluate_population_sequential(model_weights, seeds, dataset)
        
        # Prepare arguments for parallel processing
        args_list = []
        for seed_idx, seed in enumerate(seeds):
            # Calculate the range of environments for this perturbation
            start_env_idx = seed_idx * self.num_parallel_environments
            end_env_idx = start_env_idx + self.num_parallel_environments
            perturbation_dataset = dataset[start_env_idx:end_env_idx]
            
            # Assign seeds to processes (Algorithm 2, step 4)
            process_id = seed_idx % self.num_processes
            thread_id = (seed_idx // self.num_processes) % self.gpu_threads_per_process
            
            args_list.append((seed_idx, seed, model_weights, perturbation_dataset, process_id, thread_id))
        
        # Use ProcessPoolExecutor for parallel evaluation
        rewards = [0.0] * len(seeds)
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Submit all tasks
                future_to_seed = {executor.submit(self._process_seed_worker, args): args[0] 
                                 for args in args_list}
                
                # Collect results
                for future in future_to_seed:
                    seed_idx, reward = future.result()
                    rewards[seed_idx] = reward
                    
        except Exception as e:
            if self.verbose:
                self.log_print(f"Parallel evaluation failed: {e}")
                self.log_print("Falling back to sequential evaluation")
            return self._evaluate_population_sequential(model_weights, seeds, dataset)
        
        return rewards
    
    def _evaluate_population_sequential(self,
                                      model_weights: List[tf.Variable],
                                      seeds: List[int],
                                      dataset: List[Tuple[Any, int]]) -> List[float]:
        """
        Sequential evaluation fallback when parallel processing fails.
        """
        rewards = []
        for seed_idx, seed in enumerate(seeds):
            if self.verbose:
                self.log_print(f"  Evaluating perturbation {seed_idx + 1}/{self.num_perturbations}")
            
            # Calculate the range of environments for this perturbation
            start_env_idx = seed_idx * self.num_parallel_environments
            end_env_idx = start_env_idx + self.num_parallel_environments
            perturbation_dataset = dataset[start_env_idx:end_env_idx]
            
            reward = self.evaluate_model_with_perturbation(model_weights, seed, perturbation_dataset)
            rewards.append(reward)
            
            if self.verbose:
                self.log_print(f"    Reward: {reward:.6f}")
        
        return rewards
    
    def _compute_weight_update(self, weight_shape, weight_dtype, seeds_tensor, rewards_normalized_tensor, sigma, alpha, num_perturbations):
        """Optimized weight update computation."""
        # Initialize update tensor
        update = tf.zeros(weight_shape, dtype=weight_dtype)

        # Process each seed in the population
        for i in range(num_perturbations):
            # Get current seed
            current_seed = seeds_tensor[i]

            # Create seed pair [seed, 0] for stateless_normal
            seed_pair = tf.stack([current_seed, tf.constant(0, dtype=tf.int32)])

            # Generate noise for this seed using stateless_normal
            noise = tf.random.stateless_normal(
                shape=weight_shape,
                seed=seed_pair,
                dtype=weight_dtype,
                stddev=1.0
            )

            # Scale noise by corresponding normalized reward
            scaled_noise = noise * rewards_normalized_tensor[i]

            # Add to total update
            update = update + scaled_noise

        # Average across population (exactly like original paper)
        update = update / num_perturbations

        # Apply learning rate (exactly like original paper)
        final_update = alpha * update
        
        # # Debug: Log update statistics
        # if hasattr(self, 'verbose') and self.verbose:
        #     self.log_print(f"    Update debug: alpha={alpha}")
        #     self.log_print(f"    Update magnitude before alpha: {tf.reduce_sum(tf.abs(update)):.6f}")
        #     self.log_print(f"    Final update magnitude: {tf.reduce_sum(tf.abs(final_update)):.6f}")
        
        return final_update

    def update_model_weights(self, 
                           model_weights: List[tf.Variable], 
                           seeds: List[int], 
                           rewards: List[float]) -> None:
        """
        Update model weights using ES algorithm with optimized vectorized operations.
        
        Args:
            model_weights: List of model weight variables
            seeds: List of perturbation seeds
            rewards: List of corresponding rewards
        """
        # Normalize rewards using z-score (Algorithm 2, step 18)
        rewards_array = np.array(rewards, dtype=np.float32)
        rewards_mean = rewards_array.mean()
        rewards_std = rewards_array.std()
        
        if self.verbose:
            self.log_print(f"  Rewards: {rewards}")
            self.log_print(f"  Rewards mean: {rewards_mean:.6f}")
            self.log_print(f"  Rewards std: {rewards_std:.6f}")
        
        if rewards_std < 1e-6:
            if self.verbose:
                self.log_print(f"  Warning: All rewards are identical ({rewards_mean:.6f}), skipping weight update")
            return
        
        rewards_normalized = (rewards_array - rewards_mean) / (rewards_std + 1e-8)
        
        # Convert to tensors for vectorized operations
        seeds_tensor = tf.constant(seeds, dtype=tf.int32)
        rewards_normalized_tensor = tf.constant(rewards_normalized, dtype=tf.float32)
        
        # Debug: Track weight changes
        weight_norm_before = tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in model_weights])
        
        # Vectorized weight update - process all weights in parallel
        total_update_magnitude = 0.0
        
        for weight_idx, weight in enumerate(model_weights):
            if self.verbose and weight_idx % 50 == 0:  # Reduced logging frequency
                self.log_print(f"    Updating weight {weight_idx + 1}/{len(model_weights)}")
            
            # Compute update using optimized function
            update = self._compute_weight_update(
                weight_shape=weight.shape,
                weight_dtype=weight.dtype,
                seeds_tensor=seeds_tensor,
                rewards_normalized_tensor=rewards_normalized_tensor,
                sigma=self.sigma,
                alpha=self.alpha,
                num_perturbations=self.num_perturbations
            )
            
            # Apply update
            weight_before = tf.reduce_sum(tf.abs(weight))
            weight.assign_add(update)
            weight_after = tf.reduce_sum(tf.abs(weight))
            weight_change = weight_after - weight_before
            total_update_magnitude += tf.reduce_sum(tf.abs(update))
            
            # Debug: Check if weight actually changed
            if self.verbose and weight_idx == 0:  # Only log first weight
                self.log_print(f"    First weight change: {weight_change:.6f}, update_mag: {tf.reduce_sum(tf.abs(update)):.6f}")
                self.log_print(f"    First weight before: {weight_before:.6f}, after: {weight_after:.6f}")
                self.log_print(f"    Update tensor shape: {update.shape}, dtype: {update.dtype}")
                self.log_print(f"    Update tensor min/max: {tf.reduce_min(update):.6f}/{tf.reduce_max(update):.6f}")
        
        # Debug: Track weight changes
        weight_norm_after = tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in model_weights])
        weight_change = weight_norm_after - weight_norm_before
        
        if self.verbose:
            self.log_print(f"  Weight norm before: {weight_norm_before:.6f}")
            self.log_print(f"  Weight norm after: {weight_norm_after:.6f}")
            self.log_print(f"  Weight change: {weight_change:.6f}")
            self.log_print(f"  Update magnitude: {total_update_magnitude:.6f}")
            self.log_print(f"  Relative change: {(weight_change / weight_norm_before * 100):.6f}%")
    
    def save_checkpoint(self, iteration: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f"es_checkpoint_iter_{iteration}")
        self.model._transformer.save_weights(checkpoint_path)
        # Save training statistics
        stats_path = os.path.join(self.save_dir, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        if self.verbose:
            self.log_print(f"Checkpoint saved: {checkpoint_path}")
    
    def _circuit_iterator(self, circuit_files):
        """Create infinite iterator for circuit files"""
        while True:
            # shuffle circuit files
            self.rng.shuffle(circuit_files)
            for circuit_file in circuit_files:
                yield circuit_file

    def _create_circuit_env(self, circuit_file, error_rate_tolerance, w_area, w_delay, w_error):
        """Create environment for given circuit file"""
        # read circuit
        with open(circuit_file, 'r') as f:
            roots_aiger, num_ands, opt_roots_aiger, opt_num_ands = json.load(f)

        roots, info = read_aiger(aiger_str=roots_aiger)
        num_inputs, num_outputs = info[1], info[3]

        # compute truth table
        tts = compute_tts(roots, num_inputs=num_inputs)

        # create environment
        return LogicNetworkEnv(
            tts=tts,
            num_inputs=num_inputs,
            context_num_inputs=num_inputs,
            max_length=self.max_seq_length,
            eos_id=self.model.eos_id,
            pad_id=self.model.pad_id,
            max_tree_depth=self.model.max_tree_depth,
            max_inference_tree_depth=16,
            use_controllability_dont_cares=True,
            error_rate_threshold=error_rate_tolerance,
            w_gate=w_area,
            w_delay=w_delay,
            w_error=w_error,
            verbose=0
        ), opt_num_ands, roots

    def train(self,
              train_data_dir: str,
              val_data_dir: Optional[str] = None,
              validation_frequency: int = 10) -> Dict[str, Any]:
        """
        Train model using Evolution Strategies.

        Args:
            train_data_dir: Directory containing training circuit files
            val_data_dir: Directory containing validation circuit files (optional)
            validation_frequency: Frequency of validation evaluation (every N iterations)

        Returns:
            Training statistics dictionary
        """
        train_data_dir = train_data_dir + ("/" if train_data_dir[-1] != "/" else "")

        # Read circuit files and setup the infinite circuit iterator
        circuit_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)]
        if not circuit_files:
            raise ValueError(f"No circuit files found in directory: {train_data_dir}")
        circuit_iterator = self._circuit_iterator(circuit_files)

        # Write training start information to log
        def write_training_start():
            self.safe_log_write("=" * 80 + "\n")
            self.safe_log_write("=== Starting ES Training ===\n")
            self.safe_log_write(f"Starting ES training with {len(circuit_files)} training files from {train_data_dir}\n")
            if val_data_dir:
                val_circuit_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir)]
                self.safe_log_write(f"Validation directory: {val_data_dir} ({len(val_circuit_files)} files)\n")
        
        safe_file_operation(write_training_start)
        
        if self.verbose:
            self.log_print(f"Starting ES training with {len(circuit_files)} training files from {train_data_dir}")
            if val_data_dir:
                val_circuit_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir)]
                self.log_print(f"Validation directory: {val_data_dir} ({len(val_circuit_files)} files)")

        # Get model weights - support selective fine-tuning
        if hasattr(self, 'finetune_decoder_only') and self.finetune_decoder_only:
            # Freeze encoder layers like PPO training
            if hasattr(self.model, 'freeze_layers'):
                self.model.freeze_layers(freeze_encoder=True)
                if self.verbose:
                    self.log_print("Frozen encoder layers using model.freeze_layers()")
            else:
                if self.verbose:
                    self.log_print("Warning: model.freeze_layers() not available, using name-based filtering")
            
            # Get all trainable variables (encoder should be frozen)
            model_weights = self.model._transformer.trainable_variables
            if self.verbose:
                self.log_print(f"Decoder-only fine-tuning: {len(model_weights)} trainable parameters")
                # Debug: show which layers are trainable
                encoder_vars = [w for w in model_weights if 'encoder' in w.name.lower()]
                decoder_vars = [w for w in model_weights if 'decoder' in w.name.lower()]
                other_vars = [w for w in model_weights if 'encoder' not in w.name.lower() and 'decoder' not in w.name.lower()]
                self.log_print(f"  Encoder variables: {len(encoder_vars)}")
                self.log_print(f"  Decoder variables: {len(decoder_vars)}")
                self.log_print(f"  Other variables: {len(other_vars)}")
        else:
            # Fine-tune all parameters (default)
            model_weights = self.model._transformer.trainable_variables
            if self.verbose:
                self.log_print(f"Full model fine-tuning: {len(model_weights)} total parameters")

        # Record training start time
        training_start_time = time.time()

        # Main ES training loop
        for iteration in range(self.num_iterations):
            iter_start_time = time.time()

            if self.verbose:
                self.log_print(f"\nIteration {iteration + 1}/{self.num_iterations}")

            # Generate perturbation seeds
            seeds = self.generate_perturbation_seeds()

            # Create dataset for this iteration (sample from circuit files)
            # We need enough environments for all perturbations
            total_environments_needed = self.num_perturbations * self.num_parallel_environments
            dataset = []
            for _ in range(total_environments_needed):
                circuit_file = next(circuit_iterator)
                env, baseline_area, original_aig = self._create_circuit_env(
                    circuit_file, self.error_rate_tolerance, self.w_area, self.w_delay, self.w_error
                )
                # Only store the data we need, then immediately delete env to free memory
                # (env contains tt_cache_bitarray which can be large)
                dataset.append((original_aig, baseline_area))
                del env  # Explicitly delete env to free its cache memory

            # Evaluate population using parallel processes (Algorithm 2, steps 5-17)
            if self.verbose:
                self.log_print(f"  Evaluating population with {self.num_processes} processes...")
            
            rewards = self.evaluate_population_parallel(model_weights, seeds, dataset)

            # Update model weights using parameter decomposition (Algorithm 2, steps 18-25)
            self.update_model_weights(model_weights, seeds, rewards)

            # Calculate statistics
            rewards_array = np.array(rewards)
            mean_reward = rewards_array.mean()
            min_reward = rewards_array.min()
            max_reward = rewards_array.max()
            std_reward = rewards_array.std()

            # Perform validation if needed
            val_metrics = {}
            if val_data_dir and (iteration + 1) % validation_frequency == 0:
                if self.verbose:
                    self.log_print(f"  Performing validation evaluation...")
                val_metrics = self.evaluate_validation_set(val_data_dir)
                if val_metrics:
                    # Write validation results to log
                    def write_validation_log():
                        self.safe_log_write(f"Iteration {iteration + 1}/{self.num_iterations} | Validation Results:\n")
                        self.safe_log_write(f"  Avg val_area: {val_metrics['avg_area']:.4f} | Avg val_err: {val_metrics['avg_err']:.4f} | Avg val_area_ori: {val_metrics['avg_area_ori']:.4f}\n")
                        self.safe_log_write(f"  Avg val_delay: {val_metrics['avg_delay']:.4f} | Avg val_delay_ori: {val_metrics['avg_delay_ori']:.4f}\n")
                        self.safe_log_write(f"  Area Improvement: {val_metrics['total_area_improvement']:.4f} ANDs ({val_metrics['area_improvement_rate']:.2f}%)\n")
                        self.safe_log_write(f"  Delay Improvement: {val_metrics['total_delay_improvement']:.4f} levels ({val_metrics['delay_improvement_rate']:.2f}%)\n")
                        self.safe_log_write(f"  Positive area improvements: {val_metrics['positive_area_improvements']}/{val_metrics['total_circuits']} circuits\n")
                        self.safe_log_write(f"  Positive delay improvements: {val_metrics['positive_delay_improvements']}/{val_metrics['total_circuits']} circuits\n")
                        if val_metrics['max_area_improvement'] > 0:
                            self.safe_log_write(f"  Max area improvement: {val_metrics['max_area_improvement']:.0f} ANDs\n")
                        if val_metrics['max_delay_improvement'] > 0:
                            self.safe_log_write(f"  Max delay improvement: {val_metrics['max_delay_improvement']:.0f} levels\n")
                    
                    safe_file_operation(write_validation_log)
                    
                    if self.verbose:
                        self.log_print(f"Iteration {iteration + 1}/{self.num_iterations} | Validation Results:")
                        self.log_print(f"  Avg val_area: {val_metrics['avg_area']:.4f} | Avg val_err: {val_metrics['avg_err']:.4f} | Avg val_area_ori: {val_metrics['avg_area_ori']:.4f}")
                        self.log_print(f"  Avg val_delay: {val_metrics['avg_delay']:.4f} | Avg val_delay_ori: {val_metrics['avg_delay_ori']:.4f}")
                        self.log_print(f"  Area Improvement: {val_metrics['total_area_improvement']:.4f} ANDs ({val_metrics['area_improvement_rate']:.2f}%)")
                        self.log_print(f"  Delay Improvement: {val_metrics['total_delay_improvement']:.4f} levels ({val_metrics['delay_improvement_rate']:.2f}%)")
                        self.log_print(f"  Positive area improvements: {val_metrics['positive_area_improvements']}/{val_metrics['total_circuits']} circuits")
                        self.log_print(f"  Positive delay improvements: {val_metrics['positive_delay_improvements']}/{val_metrics['total_circuits']} circuits")
                        if val_metrics['max_area_improvement'] > 0:
                            self.log_print(f"  Max area improvement: {val_metrics['max_area_improvement']:.0f} ANDs")
                        if val_metrics['max_delay_improvement'] > 0:
                            self.log_print(f"  Max delay improvement: {val_metrics['max_delay_improvement']:.0f} levels")

            # Update training statistics
            self.training_stats['iteration'].append(iteration + 1)
            self.training_stats['mean_reward'].append(mean_reward)
            self.training_stats['min_reward'].append(min_reward)
            self.training_stats['max_reward'].append(max_reward)
            self.training_stats['std_reward'].append(std_reward)
            
            # Add validation statistics (compatible with both old and new format)
            if val_metrics:
                self.training_stats['val_mean_reward'].append(val_metrics.get('mean_reward', val_metrics.get('total_area_improvement', 0)))
                self.training_stats['val_area_improvement'].append(val_metrics.get('area_improvement', val_metrics.get('area_improvement_rate', 0) / 100))
                self.training_stats['val_delay_improvement'].append(val_metrics.get('delay_improvement', val_metrics.get('delay_improvement_rate', 0) / 100))
                self.training_stats['val_error_rate'].append(val_metrics.get('error_rate', val_metrics.get('avg_err', 0)))
            else:
                self.training_stats['val_mean_reward'].append(None)
                self.training_stats['val_area_improvement'].append(None)
                self.training_stats['val_delay_improvement'].append(None)
                self.training_stats['val_error_rate'].append(None)

            # Update best reward
            if max_reward > self.training_stats['best_reward']:
                self.training_stats['best_reward'] = max_reward
                self.training_stats['best_iteration'] = iteration + 1

            # Print iteration results
            iter_time = time.time() - iter_start_time
            
            # Write iteration results to log
            def write_iteration_log():
                self.safe_log_write(f"Iteration {iteration + 1}/{self.num_iterations} | Time: {iter_time:.2f}s\n")
                self.safe_log_write(f"  Mean reward: {mean_reward:.4f} | Min: {min_reward:.4f} | Max: {max_reward:.4f} | Std: {std_reward:.4f}\n")
                self.safe_log_write(f"  Best reward so far: {self.training_stats['best_reward']:.4f} (iter {self.training_stats['best_iteration']})\n")
                if val_metrics:
                    self.safe_log_write(f"  Validation - Area: {val_metrics['area_improvement_rate']:.2f}% | Delay: {val_metrics['delay_improvement_rate']:.2f}% | Error: {val_metrics['avg_err']:.4f}\n")
                self.safe_log_write("-" * 80 + "\n")
            
            safe_file_operation(write_iteration_log)
            
            if self.verbose:
                self.log_print(f"  Time: {iter_time:.2f}s")
                self.log_print(f"  Mean reward: {mean_reward:.4f}")
                self.log_print(f"  Min reward: {min_reward:.4f}")
                self.log_print(f"  Max reward: {max_reward:.4f}")
                self.log_print(f"  Std reward: {std_reward:.4f}")
                self.log_print(f"  Best reward so far: {self.training_stats['best_reward']:.4f} (iter {self.training_stats['best_iteration']})")

            # Save checkpoint (if enabled)
            if self.save_model and not self.save_final_only and (iteration + 1) % self.save_frequency == 0:
                self.save_checkpoint(iteration + 1)

            # Clean up iteration-specific data to prevent memory accumulation
            # 1. Clear dataset (env objects are already deleted immediately after use in the dataset creation loop)
            del dataset
            
            # 2. Clear intermediate variables
            del rewards
            del rewards_array
            
            # 3. Clear validation metrics if they exist
            if val_metrics:
                del val_metrics
            
            # 4. Reset model cache to prevent KV cache accumulation
            if hasattr(self.model, '_transformer') and hasattr(self.model._transformer, 'reset_cache'):
                self.model._transformer.reset_cache()
            
            # 5. Force garbage collection
            gc.collect()
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()

        # Save final model (if enabled)
        if self.save_model:
            final_path = os.path.join(self.save_dir, "es_final_model")
            self.model._transformer.save_weights(final_path)
            
        else:
            final_path = "Not saved (save_model=False)"

        total_time = time.time() - training_start_time

        # Write final training summary to log
        def write_final_summary():
            self.safe_log_write("=" * 80 + "\n")
            self.safe_log_write("ES Training Completed\n")
            self.safe_log_write("=" * 80 + "\n")
            self.safe_log_write(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n")
            self.safe_log_write(f"Final model saved: {final_path}\n")
            self.safe_log_write(f"Best reward: {self.training_stats['best_reward']:.4f} (iter {self.training_stats['best_iteration']})\n")
            self.safe_log_write("=" * 80 + "\n")
        
        safe_file_operation(write_final_summary)

        if self.verbose:
            self.log_print(f"\nES training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
            self.log_print(f"Final model saved: {final_path}")
            self.log_print(f"Best reward: {self.training_stats['best_reward']:.4f} (iter {self.training_stats['best_iteration']})")
            self.log_print(f"Training log saved: {self.log_file_path}")

        # Close log file
        self.log_file.close()

        return self.training_stats


def create_es_trainer(model, **kwargs) -> ES_Trainer:
    """
    Factory function to create ES trainer with default parameters.
    
    Args:
        model: CircuitTransformer model instance
        **kwargs: Additional parameters for ES_Trainer
        
    Returns:
        ES_Trainer instance
    """
    default_params = {
        'num_iterations': 1000,
        'population_size': 30,
        'sigma': 0.001,
        'alpha': 0.0005,
        'error_rate_tolerance': 0.1,
        'w_area': 1.4,
        'w_delay': 0.3,
        'w_error': 0.3,
        'verbose': True,
        'save_frequency': 100,
        'initial_seed': 33,
        'finetune_decoder_only': False,
        'num_processes': 1,
        'gpu_threads_per_process': 1,
        'log_file_name': 'es_train_log.txt',
        'save_model': True,
        'save_final_only': False,
        'enable_log_print': True
    }
    
    # Update defaults with provided parameters
    default_params.update(kwargs)
    
    return ES_Trainer(model, **default_params)


# Example usage and testing functions
def test_es_trainer():
    """Test function for ES trainer."""
    print("Testing ES Trainer...")

    # This would be used with an actual CircuitTransformer model
    circuit_transformer = CircuitTransformerRefactored(return_cache=True)
    circuit_transformer.build_model()
    circuit_transformer.load('/home/gst/repo/master/Approximate-Circuit-Transformer/ckpt_save/approximate-model-0026')

    es_trainer = create_es_trainer(circuit_transformer, num_iterations=100, population_size=100)

    # Load test dataset from directory
    train_data_dir = "/home/gst/repo/master/Approximate-Circuit-Transformer/datasets/random_circuit_0.1_200k"  # Replace with actual path
    val_data_dir = "/home/gst/repo/master/Approximate-Circuit-Transformer/datasets/t/extract_ori"  # Replace with actual path

    # Train with ES (including validation)
    stats = es_trainer.train(train_data_dir, val_data_dir=val_data_dir, validation_frequency=5)
    print(stats)
    print(f"Best reward: {stats['best_reward']:.4f}")
    
    # Print validation statistics
    val_rewards = [r for r in stats['val_mean_reward'] if r is not None]
    if val_rewards:
        print(f"Final validation reward: {val_rewards[-1]:.4f}")
        print(f"Best validation reward: {max(val_rewards):.4f}")

    print("ES Trainer test completed (placeholder - need actual data directory)")


if __name__ == "__main__":
    test_es_trainer()