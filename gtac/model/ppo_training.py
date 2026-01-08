import os
import time
import datetime
import shutil
import errno
import numpy as np
import tensorflow as tf
import tf_keras as keras
import keras.backend as K
import gc
import json
from gtac.utils import *
from gtac.environment import LogicNetworkEnv
from gtac.encoding import node_to_int, int_to_node, encode_aig, stack_to_encoding, deref_node


def check_disk_space(min_space_gb=1.0):
    """Check if disk space is sufficient"""
    try:
        stat = os.statvfs('.')
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        return available_gb >= min_space_gb, available_gb
    except:
        return True, 0.0  # If check fails, assume space is sufficient


def wait_for_disk_space(min_space_gb=1.0, max_wait_time=3600, check_interval=60):
    """Wait for sufficient disk space"""
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        has_space, available_gb = check_disk_space(min_space_gb)
        if has_space:
            print(f"Sufficient disk space ({available_gb:.1f}GB), continuing training")
            return True

        print(f"Insufficient disk space ({available_gb:.1f}GB < {min_space_gb}GB), retrying after {check_interval} seconds...")
        time.sleep(check_interval)

    print(f"Disk space still insufficient after waiting {max_wait_time} seconds, giving up")
    return False


def safe_file_operation(operation_func, *args, **kwargs):
    """Safely execute file operation, handling disk space insufficient errors"""
    max_retries = 3
    retry_delay = 10

    for attempt in range(max_retries):
        try:
            return operation_func(*args, **kwargs)
        except OSError as e:
            if e.errno == errno.ENOSPC:  # No space left on device
                print(f"Insufficient disk space, attempting retry {attempt+1}...")
                if attempt < max_retries - 1:
                    # Wait for disk space
                    if wait_for_disk_space(min_space_gb=2.0, max_wait_time=300, check_interval=30):
                        continue
                    else:
                        raise e
            else:
                raise e

    raise OSError("Unable to complete file operation after multiple attempts, disk space may be persistently insufficient")


def train_ppo(self,
              train_data_dir,
              ckpt_save_path=None,
              val_data_dir=None,
              MCTS=False,
              input_tts: list = None,
              epochs=400,                   # Training epochs (match train.py default)
              max_inference_seq_length = 400,  # Increase to match max_seq_length
              use_controllability_dont_cares = True,
              tts_compressed=None,
              care_set_tts=None,
              error_rate_tolerance=[0.1],  # Error rate tolerance (match train.py default)
              steps_per_epoch=512,          # Number of circuit episodes per epoch
              batch_envs=32,                # Number of parallel circuit environments (match train.py default)
              batch_size=64,                # Batch size for training (match train.py default)
              gamma=0.95,                   # Discount factor (match train.py default)
              clip_ratio=0.1,               # PPO clip ratio (match train.py default)
              policy_lr=1e-7,               # Policy learning rate (match train.py default)
              value_lr=1e-4,                # Value function learning rate (match train.py default)
              freeze_layers=True,           # Freeze layers (match train.py default)
              use_ppo_validation=True,
              ppo_train_epoch=2,            # PPO training epochs (match train.py default)
              target_kl=0.01,               # Target KL divergence (reduced to allow more policy updates)
              w_area=1.4,        # Area weight (match train.py default)
              w_delay=0.3,       # Delay weight (match train.py default)
              w_error=0.3,       # Error weight (match train.py default)
              validation_frequency=1,       # Validation frequency (match train.py default)
              log_file_name="ppo_train_log.txt",  # Log file name (match train.py default)
              seed=None,                    # Random seed for reproducibility (None for random)
              save_checkpoints=True         # Whether to save checkpoints during training
              ):  # Validate every few epochs

    train_data_dir = train_data_dir + ("/" if train_data_dir[-1] != "/" else "")
    
    # Set random seeds for reproducibility
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(f"Random seeds set to {seed} for reproducibility")
    
    # Generate timestamp for unique file identification
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update log file name to include timestamp if not provided
    if log_file_name == "ppo_train_log.txt":  # Use default filename
        log_file_name = f"ppo_train_log_{timestamp}.txt"
    
    # Fix log file path issue: if log_file_name already contains full path, use it directly; otherwise add log/ prefix
    if log_file_name.startswith("log/") or "/" in log_file_name:
        log_file_path = log_file_name
    else:
        log_file_path = "log/" + log_file_name

    log_file = open(log_file_path, "a")
    
    # Initialize memory management tracking
    self._memory_cleanup_count = 0
    self._last_cleanup_epoch = -1
    self._performance_tracking = {
        'epoch_times': [],
        'cleanup_epochs': []
    }

    # Initialize best validation metrics tracking
    self._best_val_area = float('inf')  # Track best (smallest) validation area
    self._best_val_epoch = -1
    
    # record all training parameters to log file (with disk space protection)
    def write_training_params():
        log_file.write("=" * 80 + "\n")
        log_file.write("PPO Training Parameters\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"train_data_dir: {train_data_dir}\n")
        log_file.write(f"ckpt_save_path: {ckpt_save_path}\n")
        log_file.write(f"val_data_dir: {val_data_dir}\n")
        log_file.write(f"epochs: {epochs}\n")
        log_file.write(f"batch_envs: {batch_envs}\n")
        log_file.write(f"batch_size: {batch_size}\n")
        log_file.write(f"gamma: {gamma}\n")
        log_file.write(f"clip_ratio: {clip_ratio}\n")
        log_file.write(f"policy_lr: {policy_lr}\n")
        log_file.write(f"value_lr: {value_lr}\n")
        log_file.write(f"freeze_layers: {freeze_layers}\n")
        log_file.write(f"ppo_train_epoch: {ppo_train_epoch}\n")
        log_file.write(f"target_kl: {target_kl}\n")
        log_file.write(f"use_ppo_validation: {use_ppo_validation}\n")
        log_file.write(f"w_area: {w_area}\n")
        log_file.write(f"w_delay: {w_delay}\n")
        log_file.write(f"w_error: {w_error}\n")
        log_file.write(f"error_rate_tolerance: {error_rate_tolerance}\n")
        log_file.write(f"validation_frequency: {validation_frequency}\n")
        log_file.write(f"log_file_name: {log_file_name}\n")
        log_file.write(f"seed: {seed}\n")
        log_file.write(f"save_checkpoints: {save_checkpoints}\n")
        log_file.write("=" * 80 + "\n")
        log_file.flush()

    safe_file_operation(write_training_params)

    # import tensorflow as tf  # Remove local import, use global import

    if ckpt_save_path is None:
        # Generate timestamp-based checkpoint path automatically
        base_ckpt_dir = "ckpt"
        ckpt_save_path = f"{base_ckpt_dir}/ppo_training_{timestamp}"
        def write_auto_checkpoint():
            print(f"AUTO-GENERATED CHECKPOINT PATH: {ckpt_save_path}")
        safe_file_operation(write_auto_checkpoint)
    else:
        # If path is provided but ends with certain defaults, append timestamp
        if ckpt_save_path.endswith("ckpt") or ckpt_save_path.endswith("ckpt_app"):
            ckpt_save_path = f"{ckpt_save_path}_{timestamp}"
        
    ckpt_save_path = ckpt_save_path + ("/" if ckpt_save_path[-1] != "/" else "")
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path, exist_ok=True)
        def write_checkpoint_dir():
            print(f"Created checkpoint directory: {ckpt_save_path}")
        safe_file_operation(write_checkpoint_dir)

    if freeze_layers:
        self.freeze_layers(freeze_encoder=True)


    if max_inference_seq_length is None:
        max_inference_seq_length = self.max_seq_length

    # optimizer
    policy_optimizer = keras.optimizers.AdamW(learning_rate=policy_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0)
    
    # Safe dynamic learning rate update function
    def update_policy_lr_safe(new_lr):
        """Safely update policy learning rate without causing memory issues"""
        try:
            policy_optimizer.learning_rate.assign(new_lr)
        except Exception as e:
            print(f"Warning: Failed to update learning rate: {e}")
    
    value_optimizer = keras.optimizers.Adam(learning_rate=value_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0)

    # read circuit files and setup the infinite circuit iterator
    circuit_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)]
    if not circuit_files:
        raise ValueError(f"No circuit files found in directory: {train_data_dir}")
    circuit_iterator = self._circuit_iterator(circuit_files, seed=seed)

    # build a new circuit env
    def _new_env_from_iterator():
        cf = next(circuit_iterator)
        return self._create_circuit_env(cf, error_rate_tolerance[0], w_area, w_delay, w_error), cf

    for epoch in range(epochs):
        epoch_start_time = time.time()
        def write_epoch_start():
            print(f"Epoch {epoch+1}/{epochs}")
        safe_file_operation(write_epoch_start)
        
        # Clear any remaining variables from previous epoch
        if hasattr(self, "_step_action_masks"):
            del self._step_action_masks
        if hasattr(self, "_training_action_masks"):
            del self._training_action_masks
        
        # Calculate dynamic temperature for this epoch
        base_temperature = 1.5  # Keep consistent with fixed temperature during training
        min_temperature = 1.2  # Raise minimum temperature to maintain better exploration
        max_temperature = 1.5
        decay_rate = 0.9995  # Slow down decay rate
        current_temperature = max(min_temperature, base_temperature * (decay_rate ** epoch))

        # Calculate dynamic learning rate based on temperature
        # Use more conservative learning rate decay strategy
        base_lr = policy_lr  # Use the original learning rate as base
        # Keep higher learning rate for first 200 epochs, start decay after epoch 200
        if epoch < 200:
            lr_scale = 1.0
        else:
            # Slowly decay for last 200 epochs, from 1.0 to 0.8
            progress = (epoch - 200) / 200  # 0 to 1
            lr_scale = 1.0 - 0.2 * progress  # Decay from 1.0 to 0.8
        current_policy_lr = base_lr * lr_scale
        
        print(f"Current temperature: {current_temperature:.2f}, Policy LR: {current_policy_lr:.2e}")
        def write_epoch_header():
            log_file.write(f"Epoch {epoch+1}/{epochs} | Temperature: {current_temperature:.2f} | Policy LR: {current_policy_lr:.2e}\n")
            if (epoch + 1) % 5 == 0:
                log_file.flush()
        safe_file_operation(write_epoch_header)
        
        # setup #epochs batch_envs parallelly
        envs = []
        aigs = []
        encoded_aigs = []
        enc_action_masks = []
        baseline_rewards = []
        env_source_files = []
        for m in range(batch_envs):
            pack, f = _new_env_from_iterator()
            e, baseline_reward, aig = pack
            baseline_rewards.append(baseline_reward)
            envs.append(e)
            env_source_files.append(f)
            aigs.append(aig)

            seq_enc, pos_enc = encode_aig(aigs[m], self.num_inputs)
            input_tt = self.input_tt if input_tts is None else input_tts[m]
            tts = [compute_tt(root, input_tt=input_tt) for root in aig]
            enc_action_masks.append(self.generate_action_masks(tts,
                                                            input_tt,
                                                            None if care_set_tts is None else care_set_tts[m],
                                                            seq_enc,
                                                            use_controllability_dont_care=use_controllability_dont_cares,
                                                            tts_compressed=None if tts_compressed is None else tts_compressed[m]))
            encoded_aigs.append(self._encode_postprocess(seq_enc, pos_enc))

        enc_action_masks = np.stack(enc_action_masks)
        seq_enc, pos_enc = tuple(map(lambda x: np.stack(x, axis=0), zip(*encoded_aigs)))
        batch_size = len(aigs)
        inputs_list = []
        inputs = {'inputs': seq_enc, 'enc_pos_encoding': pos_enc, 'enc_action_mask': enc_action_masks}
        targets = np.zeros((batch_size, 1), dtype=np.int32)
        dec_pos_encoding = np.zeros((batch_size, 1, self.max_tree_depth * 2), dtype=np.float32)
        
        # Reset KV cache at the beginning of each epoch to avoid memory accumulation
        inputs['cache'] = None

        per_env_episodes = [[] for _ in range(batch_envs)]
        trajectories = []
        env_completed = [False] * batch_envs
        completed_circuits = 0  # Initialize counter for successful circuits

        for ii in range(max_inference_seq_length):
            if all(env_completed):
                def write_all_finished():
                    print("all finished")
                safe_file_operation(write_all_finished)
                break

            inputs['targets'], inputs['dec_pos_encoding'] = targets, dec_pos_encoding
            # Batch process action masks, use the current step action mask (consistent with inference.py)
            action_masks = np.stack([e.action_masks[ii] if ii < len(e.action_masks) else np.zeros(self.vocab_size, dtype=bool) for e in envs], axis=0)
            inputs['dec_action_mask'] = np.expand_dims(action_masks, axis=1)
            
            # Store action mask for each step in trajectory
            if not hasattr(self, '_step_action_masks'):
                self._step_action_masks = []
            self._step_action_masks.append(action_masks.copy())
            

            # Use KV cache with intelligent management strategy
            logits, pack = self._transformer_inference(inputs, return_kv_cache=True, return_last_token=True, return_value=True)
            values = pack['value']
            cache = pack['cache']
            
            # Use KV cache directly, no additional restrictions needed
            # Because cache is reset to None at the beginning of each epoch
            inputs['cache'] = cache
            # logits: [B, V], values: [B, 1]
            # Use tf.stop_gradient to prevent gradient flow during sampling
            values = np.array(tf.stop_gradient(values))[:,0]
            logits_np = logits if isinstance(logits, np.ndarray) else tf.stop_gradient(logits).numpy()
            # Use fixed temperature 1.0 for consistency
            logits_np = logits_np / 1.0
            
            # Ensure values_np is correctly defined
            values_np = values if isinstance(values, np.ndarray) else tf.stop_gradient(values).numpy()
            
            # Batch sampling to reduce loop overhead
            # Note: logits are already masked in transformer (tensorflow_transformer.py line 411)
            # No need to apply mask again here
            probs = logits_np.copy()
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Batch softmax computation
            probs = np.exp(probs - np.max(probs, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            
            # Vectorized sampling to improve GPU utilization
            actions = np.zeros(batch_envs, dtype=np.int32)
            log_probs = np.zeros(batch_envs, dtype=np.float32)

           
            # Batch process incomplete environments
            unfinished_mask = np.array([not env.is_finished for env in envs])
            if np.any(unfinished_mask):
                unfinished_indices = np.where(unfinished_mask)[0]
                unfinished_probs = probs[unfinished_indices]
                # Batch sampling
                for i, idx in enumerate(unfinished_indices):
                    try:
                        action = np.random.choice(len(unfinished_probs[i]), p=unfinished_probs[i])
                        while not envs[idx].is_finished and action == envs[idx].PAD:
                            action = np.random.choice(len(unfinished_probs[i]), p=unfinished_probs[i])
                        log_probs[idx] = np.log(max(unfinished_probs[i][action], 1e-12))
                    except ValueError:
                        action = np.argmax(logits_np[idx])  # Use original logits (already masked)
                        log_probs[idx] = np.log(1e-12)
                        print(f"WARNING: Random choice failed, using argmax: {action}")
                    actions[idx] = int(action)
            
            # Process completed environments
            finished_mask = ~unfinished_mask
            if np.any(finished_mask):
                finished_indices = np.where(finished_mask)[0]
                for idx in finished_indices:
                    actions[idx] = envs[idx].PAD
                    log_probs[idx] = 0.0

            print(actions)
            
            # Batch create states to reduce loop overhead
            for j in range(batch_envs):
                single_state = {
                    'inputs': inputs['inputs'][j:j+1],
                    'enc_pos_encoding': inputs['enc_pos_encoding'][j:j+1],
                    'enc_action_mask': inputs['enc_action_mask'][j:j+1],
                    'targets': inputs['targets'][j:j+1],
                    'dec_pos_encoding': inputs['dec_pos_encoding'][j:j+1],
                    'dec_action_mask': inputs['dec_action_mask'][j:j+1],
                    'cache': inputs.get('cache', ())
                }
                inputs_list.append(single_state)
            # Vectorized environment stepping to improve GPU utilization
            rewards = np.zeros(batch_envs, dtype=np.float32)
            dones = np.zeros(batch_envs, dtype=bool)
            
            # Batch process incomplete environments
            for j in range(batch_envs):
                if not envs[j].is_finished:
                    reward, done = envs[j].step(actions[j])
                    rewards[j] = reward * 0.5
                    dones[j] = done
                else:
                    rewards[j] = 0.0
                    dones[j] = True
            
            # Batch process positional encodings
            pos_encodings = np.stack([e.positional_encodings[-1] for e in envs], axis=0)
            pos_encodings = np.expand_dims(pos_encodings, axis=1)

            targets_new = np.expand_dims(actions, axis=1)
            if self.use_kv_cache:
                targets = targets_new
                dec_pos_encoding = pos_encodings
            else:
                targets = np.concatenate([targets, targets_new], axis=1)
                dec_pos_encoding = np.concatenate([dec_pos_encoding, pos_encodings], axis=1)


            # Vectorized trajectory recording to improve GPU utilization
            for k in range(batch_envs):
                step_record = {
                    'state': inputs_list[k],
                    'action': int(actions[k]),
                    'reward': float(rewards[k]),
                    'log_prob': float(log_probs[k]),
                    'value': float(values_np[k].reshape(-1)[0]) if np.ndim(values_np[k]) > 0 else float(values_np[k]),
                    'done': bool(dones[k]),
                    'action_mask': np.array(action_masks[k], dtype=bool, copy=True)  # CRITICAL FIX: Ensure proper copy
                }
                
                per_env_episodes[k].append(step_record)
            
            # clear inputs_list, avoid memory accumulation
            inputs_list.clear()
            
            # Minimize memory cleanup to avoid performance loss
            del pos_encodings, targets_new, actions, action_masks, log_probs
            del logits, pack, values, cache, logits_np, values_np
            
            # Clean memory more frequently to prevent OOM
            if ii % 50 == 0:
                import gc
                gc.collect()
            

            for k in range(batch_envs):
                if dones[k] and not env_completed[k]:
                    ep = copy.deepcopy(per_env_episodes[k])
                    if len(ep) > 0:
                        total_reward = sum([s['reward'] for s in ep])
                        # for s in ep:
                        #     s['reward'] = 0.0
                        
                        
                        # check if circuit is complete
                        if envs[k].success and len(envs[k].roots) == envs[k].num_outputs:
                            # Calculate comprehensive reward: include area, delay, error rate
                            final_reward = self._calculate_comprehensive_reward(
                                envs[k].roots, aigs[k], baseline_rewards[k], error_rate_tolerance[0],
                                w_area, w_delay, w_error, log_file=log_file
                            )
                            # Count successful circuit
                            completed_circuits += 1
                        else:
                            # Circuit is incomplete, give severe punishment
                            final_reward = -1.0
                            print(f"Circuit incomplete[{k}]: success={envs[k].success}, roots={len(envs[k].roots)}, outputs={envs[k].num_outputs}, reward = -1.0")
                        ep[-1]['reward'] = final_reward
                    
                    trajectories.append((ep, k))  # Store both episode and its env index
                    env_completed[k] = True     # mark the completed env
                    
                    # print(f"env ({k}) finished, total steps: {len(ep)}, total states length: {len(ep[0]['state'])}")
            
            # Clean up step-level variables after trajectory processing
            if 'rewards' in locals():
                del rewards
            if 'dones' in locals():
                del dones
        

        # First record trajectory information for logging
        num_episodes = len(trajectories)


        all_rewards = []
        for ep, _ in trajectories:  # Unpack the tuple (ep, env_idx)
            for step in ep:
                all_rewards.append(step['reward'])
        
        # Make sampling_action_mask available for debugging in training
        if 'sampling_action_mask' not in locals():
            sampling_action_mask = None
        globals()['sampling_action_mask'] = sampling_action_mask
        
        states_proc, actions_proc, old_log_probs_proc, returns_proc, advantages_proc = self._process_trajectories(
            trajectories, gamma, log_file=log_file, epoch=epoch, epochs=epochs
        )
        
        
        # Immediately clean up trajectories to release large amounts of memory
        del trajectories
        
        # Additional cleanup after trajectory processing
        import gc
        gc.collect()
        

        print(f"collected {num_episodes} episodes this epoch")
        def write_episode_info():
            log_file.write(f"Epoch {epoch+1}/{epochs} | Collected {num_episodes} episodes\n")
        safe_file_operation(write_episode_info)
        
        # Calculate circuit completion rate
        circuit_completion_rate = (completed_circuits / num_episodes * 100) if num_episodes > 0 else 0
        print(f"Circuit completion rate: {circuit_completion_rate:.1f}% ({completed_circuits}/{num_episodes} circuits)")
        def write_completion_rate():
            log_file.write(f"Epoch {epoch+1}/{epochs} | Circuit completion rate: {circuit_completion_rate:.1f}% ({completed_circuits}/{num_episodes} circuits)\n")
        safe_file_operation(write_completion_rate)
        
        # Monitor reward range and completion rate (using GAE returns and advantages)
        if returns_proc is not None and len(returns_proc) > 0 and advantages_proc is not None and len(advantages_proc) > 0:
            returns_array = np.array(returns_proc)
            advantages_array = np.array(advantages_proc)

            # Calculate completion rate based on positive returns (successful circuits)
            completed_returns = [r for r in returns_proc if r > 0.0]  # Positive returns indicate successful circuits
            completion_rate = len(completed_returns) / len(returns_proc) * 100

            print(f"Return range: min={returns_array.min():.4f}, max={returns_array.max():.4f}, mean={returns_array.mean():.4f}")

            def write_return_stats():
                log_file.write(f"Epoch {epoch+1}/{epochs} | Return range: min={returns_array.min():.4f}, max={returns_array.max():.4f}, mean={returns_array.mean():.4f}\n")
            safe_file_operation(write_return_stats)

            if completed_returns:
                completed_returns_array = np.array(completed_returns)
                print(f"Completed returns: min={completed_returns_array.min():.4f}, max={completed_returns_array.max():.4f}, mean={completed_returns_array.mean():.4f}")
                def write_completed_returns():
                    log_file.write(f"Epoch {epoch+1}/{epochs} | Completed returns: min={completed_returns_array.min():.4f}, max={completed_returns_array.max():.4f}, mean={completed_returns_array.mean():.4f}\n")
                safe_file_operation(write_completed_returns)

            # Calculate advantage statistics for successful vs failed episodes
            if len(completed_returns) > 0 and len(returns_proc) > len(completed_returns):
                failed_indices = [i for i, r in enumerate(returns_proc) if r <= 0.0]
                successful_indices = [i for i, r in enumerate(returns_proc) if r > 0.0]

                if failed_indices:
                    failed_advantages = advantages_array[failed_indices]
                    print(f"Failed advantages: min={failed_advantages.min():.4f}, max={failed_advantages.max():.4f}, mean={failed_advantages.mean():.4f}")
                    def write_failed_advantages():
                        log_file.write(f"Epoch {epoch+1}/{epochs} | Failed advantages: min={failed_advantages.min():.4f}, max={failed_advantages.max():.4f}, mean={failed_advantages.mean():.4f}\n")
                    safe_file_operation(write_failed_advantages)

            log_file.flush()
        
        # Update policy learning rate based on current temperature
        update_policy_lr_safe(current_policy_lr)

        start_time = time.time()
        policy_loss = self._update_policy(
            states_proc, actions_proc, old_log_probs_proc, advantages_proc,
            policy_optimizer, clip_ratio, target_kl, batch_size, ppo_train_epoch,
            log_file=log_file, epoch=epoch, epochs=epochs
        )
        step1_time = time.time()
        print(f"Policy update total time: {step1_time - start_time:.4f} seconds")
        def write_policy_time():
            log_file.write(f"Epoch {epoch+1}/{epochs} | Policy update time: {step1_time - start_time:.4f}s\n")
        safe_file_operation(write_policy_time)

        value_loss = self._update_value_function(
            states_proc, returns_proc, value_optimizer, batch_size, ppo_train_epoch
        )
        step2_time = time.time()
        print(f"Value update total time: {step2_time - step1_time:.4f} seconds")
        def write_value_time():
            log_file.write(f"Epoch {epoch+1}/{epochs} | Value update time: {step2_time - step1_time:.4f}s\n")
        safe_file_operation(write_value_time)
        # Reduce flush frequency to avoid I/O blocking
        if (epoch + 1) % 5 == 0:
            log_file.flush()

        # Calculate epoch time and track performance
        epoch_time = time.time() - epoch_start_time
        self._performance_tracking['epoch_times'].append(epoch_time)
        
        # Enhanced logging with training stability metrics and performance tracking
        policy_stability = "STABLE" if policy_loss < 1.0 else "UNSTABLE"
        value_stability = "GOOD" if value_loss < 2.0 else "POOR" if value_loss > 4.0 else "FAIR"
        
        # Check if this epoch was after a cleanup
        is_after_cleanup = (self._last_cleanup_epoch >= 0 and epoch > self._last_cleanup_epoch)
        cleanup_indicator = " [AFTER_CLEANUP]" if is_after_cleanup else ""
        
        print(f"Epoch {epoch+1}/{epochs} | Policy Loss: {policy_loss:.4f} ({policy_stability}) | Value Loss: {value_loss:.4f} ({value_stability}) | Avg Return: {np.mean(returns_proc):.2f} | Time: {epoch_time:.2f}s{cleanup_indicator}")
        def write_epoch_summary():
            log_msg = f"Epoch {epoch+1}/{epochs} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | Avg Return: {np.mean(returns_proc):.2f} | Time: {epoch_time:.2f}s{cleanup_indicator}"
            log_file.write(log_msg + "\n")
        safe_file_operation(write_epoch_summary)
        
        # Track performance impact of cleanup
        if is_after_cleanup and len(self._performance_tracking['epoch_times']) > 1:
            prev_epoch_time = self._performance_tracking['epoch_times'][-2]
            time_increase = epoch_time - prev_epoch_time
            if time_increase > 5.0:  # Significant time increase
                print(f"WARNING: Epoch time increased by {time_increase:.2f}s after cleanup (performance degradation detected)")
                def write_performance_warning():
                    log_file.write(f"WARNING: Epoch time increased by {time_increase:.2f}s after cleanup (performance degradation detected)\n")
                safe_file_operation(write_performance_warning)
        
        # Reduce flush frequency to avoid I/O blocking
        if (epoch + 1) % 5 == 0:
            log_file.flush()
        
        # Clear processed data, release memory
        del states_proc, actions_proc, old_log_probs_proc, returns_proc, advantages_proc
        # Clean up all large variables in the training loop
        del envs, aigs, encoded_aigs, enc_action_masks, seq_enc, pos_enc
        del inputs, targets, dec_pos_encoding, per_env_episodes
        
        # Force cleanup of trajectory-related variables
        if 'env_completed' in locals():
            del env_completed
        if 'num_episodes' in locals():
            del num_episodes
        if 'all_rewards' in locals():
            del all_rewards
        if 'completed_rewards' in locals():
            del completed_rewards
        if 'completed_circuits' in locals():
            del completed_circuits
        
        # Force clean up cache-related variables to ensure complete reset each epoch
        if 'cache' in locals():
            del cache
        if 'inputs_list' in locals():
            del inputs_list
        
        # Clear dataset caches every epoch to prevent accumulation
        # This is the key fix for memory accumulation issue
        if hasattr(self, "_policy_dataset_cache"):
            del self._policy_dataset_cache
        if hasattr(self, "_value_dataset_cache"):
            del self._value_dataset_cache
        
        # Clear action masks to prevent memory accumulation
        if hasattr(self, "_step_action_masks"):
            del self._step_action_masks
        if hasattr(self, "_training_action_masks"):
            del self._training_action_masks
        
        pass
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


        # Validation - reduce frequency to avoid GPU idle time
        val_area = []
        val_original_ands = []
        val_aigs = []
        val_err = []
        val_delay = []
        val_original_delay = []
        # import tensorflow as tf  # Remove local import, use global import
        tf.config.run_functions_eagerly(True)
        try:
            # Only validate every 10 epochs to prevent OOM
            if val_data_dir is not None and (epoch + 1) % max(1, validation_frequency) == 0:
                # Clear validation set data, avoid accumulation
                val_aigs.clear()
                val_original_ands.clear()
                for root, dirs, files in os.walk(val_data_dir):
                    for m, file in enumerate(files):
                        file_path = os.path.join(root, file)
                        
                        with open(file_path, 'r') as f:
                            content = f.read()
                            aag1 = content
                            aig1, _ = read_aiger(aiger_str=aag1)
                            and_num = count_num_ands(aig1)
                            
                            val_aigs.append(aig1)
                            val_original_ands.append(and_num)
                
                if val_aigs:  
                    for error_threshold in error_rate_tolerance:
                        if not MCTS:
                            if use_ppo_validation:
                                print(f"Using PPO validation strategy for error_threshold={error_threshold}")
                                optimized_aigs = self._optimize_with_ppo_policy(val_aigs, error_rate_threshold=error_threshold, seed=seed)
                            else:
                                print(f"Using deterministic validation strategy for error_threshold={error_threshold}")
                                optimized_aigs = self.optimize(val_aigs, error_rate_threshold=error_threshold)
                        else:
                            optimized_aigs = self.optimize(
                                aigs=val_aigs,
                                num_mcts_steps=1,
                                num_mcts_playouts_per_step=2,
                                error_rate_threshold=error_threshold
                            )
                        
                        total_area = 0.0
                        total_err = 0.0
                        total_area_ori = 0.0
                        num_circuits = len(val_aigs)
                        successful_optimizations = 0
                        
                        for m in range(num_circuits):
                            original_aig = val_aigs[m]
                            optimized_aig = optimized_aigs[m]
                            
                            original_ands = val_original_ands[m]
                            optimized_ands = count_num_ands(optimized_aig)

                            # Calculate error rate, handle incomplete circuits
                            try:
                                error_rate = checkER(original_aig, optimized_aig)
                            except (ValueError, AttributeError) as e:
                                # If circuit is incomplete, set a high error rate
                                error_rate = 1.0
                                # print(f"Validation: Circuit {m} incomplete, setting error_rate = 1.0")

                            # Calculate delay metrics
                            from circuit_transformer.utils import compute_critical_path
                            original_delay = compute_critical_path(original_aig)
                            optimized_delay = compute_critical_path(optimized_aig)
                            
                            val_area.append(optimized_ands)
                            val_err.append(error_rate)
                            val_delay.append(optimized_delay)
                            val_original_delay.append(original_delay)

        finally:
            tf.config.run_functions_eagerly(False)
            avg_area_ori = np.mean(val_original_ands) if val_original_ands else 0
            avg_area = np.mean(val_area) if val_area else 0
            avg_err = np.mean(val_err) if val_err else 0
            avg_delay_ori = np.mean(val_original_delay) if val_original_delay else 0
            avg_delay = np.mean(val_delay) if val_delay else 0

            # Calculate optimization effect
            total_area_improvement = avg_area_ori - avg_area
            area_improvement_rate = (total_area_improvement / avg_area_ori * 100) if avg_area_ori > 0 else 0
            total_delay_improvement = avg_delay_ori - avg_delay
            delay_improvement_rate = (total_delay_improvement / avg_delay_ori * 100) if avg_delay_ori > 0 else 0

            # Calculate improvement statistics
            area_improvements = [val_original_ands[i] - val_area[i] for i in range(len(val_area))]
            delay_improvements = [val_original_delay[i] - val_delay[i] for i in range(len(val_delay))]
            positive_area_improvements = [imp for imp in area_improvements if imp > 0]
            positive_delay_improvements = [imp for imp in delay_improvements if imp > 0]
            
            print(f"Epoch {epoch+1}/{epochs} | Avg val_area: {avg_area:.4f} | Avg val_err: {avg_err:.4f} | Avg val_area_ori: {avg_area_ori:.4f}")
            print(f"Epoch {epoch+1}/{epochs} | Avg val_delay: {avg_delay:.4f} | Avg val_delay_ori: {avg_delay_ori:.4f}")
            print(f"Epoch {epoch+1}/{epochs} | Area Improvement: {total_area_improvement:.4f} ANDs ({area_improvement_rate:.2f}%)")
            print(f"Epoch {epoch+1}/{epochs} | Delay Improvement: {total_delay_improvement:.4f} levels ({delay_improvement_rate:.2f}%)")
            print(f"Epoch {epoch+1}/{epochs} | Positive area improvements: {len(positive_area_improvements)}/{len(val_area)} circuits")
            print(f"Epoch {epoch+1}/{epochs} | Positive delay improvements: {len(positive_delay_improvements)}/{len(val_delay)} circuits")
            if positive_area_improvements:
                print(f"Epoch {epoch+1}/{epochs} | Max area improvement: {max(positive_area_improvements):.0f} ANDs")
            if positive_delay_improvements:
                print(f"Epoch {epoch+1}/{epochs} | Max delay improvement: {max(positive_delay_improvements):.0f} levels")
            
            def write_validation_results():
                log_msg = f"Epoch {epoch+1}/{epochs} | Avg val_area: {avg_area:.4f} | Avg val_err: {avg_err:.4f} | Avg val_area_ori: {avg_area_ori:.4f} | Avg val_delay: {avg_delay:.4f} | Avg val_delay_ori: {avg_delay_ori:.4f} | Area Improvement: {total_area_improvement:.4f} ANDs ({area_improvement_rate:.2f}%) | Delay Improvement: {total_delay_improvement:.4f} levels ({delay_improvement_rate:.2f}%)"
                log_file.write(log_msg + "\n")
                log_file.flush()
            safe_file_operation(write_validation_results)

            # Save checkpoint only if validation area improved (smaller is better) and save_checkpoints is enabled
            if ckpt_save_path is not None and save_checkpoints:
                should_save = False
                save_reason = ""

                if avg_area < self._best_val_area:
                    should_save = True
                    save_reason = f"Better validation area: {avg_area:.4f} < {self._best_val_area:.4f}"
                    self._best_val_area = avg_area
                    self._best_val_epoch = epoch + 1

                if should_save:
                    # Delete previous checkpoint if it exists
                    if hasattr(self, '_current_checkpoint_path') and self._current_checkpoint_path is not None:
                        if os.path.exists(self._current_checkpoint_path):
                            import shutil
                            shutil.rmtree(self._current_checkpoint_path)
                            def write_checkpoint_deleted():
                                print(f"Deleted previous checkpoint: {self._current_checkpoint_path}")
                            safe_file_operation(write_checkpoint_deleted)

                    save_path = os.path.join(ckpt_save_path, f"model-{epoch+1:04d}")
                    self._transformer.save_weights(save_path)
                    self._current_checkpoint_path = save_path  # Track current checkpoint path
                    def write_checkpoint_saved():
                        print(f"Model weights saved at {save_path} ({save_reason})")
                    safe_file_operation(write_checkpoint_saved)

            # Clear validation set related variables, prevent memory leaks
            if 'optimized_aigs' in locals():
                del optimized_aigs
            if 'val_aigs' in locals():
                del val_aigs
            if 'val_original_ands' in locals():
                del val_original_ands
            if 'val_area' in locals():
                del val_area
            if 'val_err' in locals():
                del val_err
            if 'val_delay' in locals():
                del val_delay
            if 'val_original_delay' in locals():
                del val_original_delay

            # Force clean up all validation-related variables (if they exist)
            try:
                if 'optimized_aigs' in locals():
                    del optimized_aigs
                if 'val_aigs' in locals():
                    del val_aigs
                if 'val_original_ands' in locals():
                    del val_original_ands
                if 'val_area' in locals():
                    del val_area
                if 'val_err' in locals():
                    del val_err
                if 'val_delay' in locals():
                    del val_delay
                if 'val_original_delay' in locals():
                    del val_original_delay
                if 'positive_area_improvements' in locals():
                    del positive_area_improvements
                if 'positive_delay_improvements' in locals():
                    del positive_delay_improvements
            except NameError:
                pass  # Variable does not exist, ignore

            # Clear GPU memory, prevent memory leaks
            import gc
            gc.collect()
            

            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            try:
                gpu_devices = tf.config.list_physical_devices('GPU')
                if gpu_devices:
                    gpu_memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    current_mb = gpu_memory_info['current'] / (1024**2)
                    
                    # Only clear session at very high memory usage to prevent OOM
                    if current_mb > 15000:  # Only when memory is critically high
                        print(f"Critical memory usage ({current_mb:.1f}MB), clearing session...")
                        self._memory_cleanup_count += 1
                        self._last_cleanup_epoch = epoch
                        self._performance_tracking['cleanup_epochs'].append(epoch)
                        
                        tf.keras.backend.clear_session()
                        # Mark that we need to rebuild functions
                        if hasattr(self, '_policy_train_step_fn'):
                            self._policy_train_step_fn = None
                        if hasattr(self, '_value_train_step_fn'):
                            self._value_train_step_fn = None
                        
                        print(f"Memory cleanup #{self._memory_cleanup_count} performed at epoch {epoch+1}")
                        def write_memory_cleanup():
                            log_file.write(f"Epoch {epoch+1}/{epochs} | Memory cleanup #{self._memory_cleanup_count} performed (memory: {current_mb:.1f}MB)\n")
                        safe_file_operation(write_memory_cleanup)

                        # Force cleanup of CUDA cache
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass
            except Exception as e:
                pass

            # Monitor GPU memory usage
            try:
                gpu_devices = tf.config.list_physical_devices('GPU')
                if gpu_devices:
                    gpu_memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    current_mb = gpu_memory_info['current'] / (1024**2)
                    peak_mb = gpu_memory_info['peak'] / (1024**2)
                    print(f"GPU Memory: {current_mb:.1f}MB current, {peak_mb:.1f}MB peak")
                    def write_gpu_memory():
                        log_file.write(f"Epoch {epoch+1}/{epochs} | GPU Memory: {current_mb:.1f}MB current, {peak_mb:.1f}MB peak\n")
                    safe_file_operation(write_gpu_memory)

                    # If memory usage exceeds 5GB, emit a warning
                    if current_mb > 5000:
                        print(f"WARNING: High GPU memory usage: {current_mb:.1f}MB")
                        def write_gpu_memory_warning():
                            log_file.write(f"Epoch {epoch+1}/{epochs} | WARNING: High GPU memory usage: {current_mb:.1f}MB\n")
                        safe_file_operation(write_gpu_memory_warning)
                    log_file.flush()
            except Exception as e:
                pass

        # Clear environment variables and trajectory data
        if 'envs' in locals():
            del envs
        if 'trajectories' in locals():
            del trajectories
        if 'per_env_episodes' in locals():
            del per_env_episodes
        if 'inputs_list' in locals():
            del inputs_list
        if 'inputs_batch' in locals():
            del inputs_batch
        if 'returns' in locals():
            del returns
        if 'advantages' in locals():
            del advantages
        if 'optimized_aigs' in locals():
            del optimized_aigs
        if 'inputs' in locals():
            del inputs
        if 'cache' in locals():
            del cache
        if 'states_proc' in locals():
            del states_proc
        if 'actions_proc' in locals():
            del actions_proc
        if 'old_log_probs_proc' in locals():
            del old_log_probs_proc
        if 'returns_proc' in locals():
            del returns_proc
        if 'advantages_proc' in locals():
            del advantages_proc

    # Training completed - save final best model if we have a checkpoint path and save_checkpoints is enabled
    if ckpt_save_path is not None and save_checkpoints and self._best_val_area < float('inf'):
        final_best_path = os.path.join(ckpt_save_path, "final_best_model")
        self._transformer.save_weights(final_best_path)
        def write_training_completed():
            print(f"Training completed. Final best model saved at {final_best_path}")
            print(f"Best validation area achieved: {self._best_val_area:.4f} at epoch {self._best_val_epoch}")
        safe_file_operation(write_training_completed)

        # Clean up intermediate checkpoints, keep only final_best_model (only if save_checkpoints is enabled)
        if save_checkpoints:
            import shutil
            import glob
            intermediate_checkpoints = glob.glob(os.path.join(ckpt_save_path, "model-*"))
            for checkpoint in intermediate_checkpoints:
                if os.path.exists(checkpoint):
                    shutil.rmtree(checkpoint)
                    def write_checkpoint_deleted():
                        print(f"Deleted intermediate checkpoint: {checkpoint}")
                    safe_file_operation(write_checkpoint_deleted)
            
            def write_cleanup_info():
                print(f"Cleaned up {len(intermediate_checkpoints)} intermediate checkpoints")
            safe_file_operation(write_cleanup_info)

            # Also save a summary file (with disk space protection)
            if ckpt_save_path is not None:
                summary_path = os.path.join(ckpt_save_path, "training_summary.txt")
                def write_summary():
                    with open(summary_path, 'w') as f:
                        f.write(f"Training completed on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Total epochs: {epochs}\n")
                        f.write(f"Best validation area: {self._best_val_area:.4f}\n")
                        f.write(f"Achieved at epoch: {self._best_val_epoch}\n")
                        f.write(f"Checkpoint directory: {ckpt_save_path}\n")
                safe_file_operation(write_summary)
                def write_summary_saved():
                    print(f"Training summary saved at {summary_path}")
                safe_file_operation(write_summary_saved)


def _circuit_iterator(self, circuit_files, seed=None):
    """Create infinite iterator for circuit files"""
    while True:
        # Shuffle circuit files
        np.random.shuffle(circuit_files)
        for circuit_file in circuit_files:
            yield circuit_file


def _create_circuit_env(self, circuit_file, error_rate_threshold, w_area, w_delay, w_error):
    """Create environment for given circuit file"""
    # Read circuit
    with open(circuit_file, 'r') as f:
        roots_aiger, num_ands, opt_roots_aiger, opt_num_ands = json.load(f)

    roots, info = read_aiger(aiger_str=roots_aiger)
    num_inputs, num_outputs = info[1], info[3]

    # Compute truth table
    tts = compute_tts(roots, num_inputs=num_inputs)

    # Create environment
    return LogicNetworkEnv(
        tts=tts,
        num_inputs=num_inputs,
        context_num_inputs=num_inputs,
        max_length=self.max_seq_length,
        eos_id=self.eos_id,
        pad_id=self.pad_id,
        max_tree_depth=self.max_tree_depth,
        max_inference_tree_depth=32,
        use_controllability_dont_cares=True,
        error_rate_threshold=error_rate_threshold,
        w_gate=w_area,
        w_delay=w_delay,
        w_error=w_error,
        verbose=0
    ), opt_num_ands, roots


def _prepare_batch_input(self, states):
    """
    Stack states and collapse an extra middle axis if present.
    Expected element shapes (per state):
      - inputs: either (seq,) or (a, b) where seq = a*b
      - enc_pos_encoding: either (seq, pos_dim) or (a, b, pos_dim) -> collapse to (seq, pos_dim)
      - enc_action_mask: either (seq, vocab) or (a, b, vocab) -> collapse to (seq, vocab)
      - targets: either (dec_seq,) or (a, dec_seq) -> collapse to (dec_seq,) or keep (dec_seq,)
      - dec_pos_encoding: either (dec_seq, pos_dim) or (a, dec_seq, pos_dim) -> collapse to (dec_seq, pos_dim)
      - dec_action_mask: either (dec_seq, vocab) or (a, dec_seq, vocab) -> collapse to (dec_seq, vocab)
    Returns dict of tensors ready to feed the transformer.
    """

    def stack_np(arr_list):
        return np.stack(arr_list, axis=0)

    def collapse_middle_axis(x_np):
        # If x_np has shape (B, A, C, ...) and we want (B, A*C, ...)
        if x_np.ndim >= 3:
            B = x_np.shape[0]
            # combine axis 1 and 2
            new_second = x_np.shape[1] * x_np.shape[2]
            rest = x_np.shape[3:]  # may be empty tuple
            return x_np.reshape((B, new_second) + rest)
        return x_np

    # collect lists
    inputs = []
    enc_pos_encoding = []
    enc_action_mask = []
    targets = []
    dec_pos_encoding = []
    dec_action_mask = []

    for i, s in enumerate(states):
        inputs.append(s['inputs'])
        enc_pos_encoding.append(s['enc_pos_encoding'])
        enc_action_mask.append(s['enc_action_mask'])
        targets.append(s['targets'])
        dec_pos_encoding.append(s['dec_pos_encoding'])
        dec_action_mask.append(s['dec_action_mask'])
        

    # stack to numpy arrays
    inp_np = stack_np(inputs)                     # maybe (B, a, c) or (B, seq)
    enc_pos_np = stack_np(enc_pos_encoding)       # maybe (B, a, c, pos_dim) or (B, seq, pos_dim)
    enc_mask_np = stack_np(enc_action_mask)       # maybe (B, a, c, V) or (B, seq, V)
    tgt_np = stack_np(targets)                    # maybe (B, dec_seq) or (B, a, dec_seq)
    dec_pos_np = stack_np(dec_pos_encoding)       # maybe (B, a, dec_seq, pos_dim) or (B, dec_seq, pos_dim)
    dec_mask_np = stack_np(dec_action_mask)       # maybe (B, a, dec_seq, V) or (B, dec_seq, V)

    
    def collapse_if_needed(arr, expected_ndim):
        """Helper function to collapse dimensions if needed"""
        if arr.ndim > expected_ndim:
            # If we have extra dimensions, flatten the middle ones
            if arr.ndim == expected_ndim + 1:
                # (B, A, C, ...) -> (B, A*C, ...)
                B = arr.shape[0]
                if expected_ndim == 2:
                    # For 2D target: (B, A, C) -> (B, A*C)
                    new_shape = (B, arr.shape[1] * arr.shape[2])
                else:
                    # For 3D target: (B, A, C, D) -> (B, A*C, D)
                    remaining_dims = arr.shape[3:]
                    new_shape = (B, arr.shape[1] * arr.shape[2]) + remaining_dims
                # print(f"    Collapsing 3D->{expected_ndim}D: {arr.shape} -> {new_shape}")
                arr = arr.reshape(new_shape)
            elif arr.ndim == expected_ndim + 2:
                # (B, A, C, D, ...) -> (B, A*C*D, ...) for 2D target
                # (B, A, C, D, ...) -> (B, A*C, D, ...) for 3D target
                B = arr.shape[0]
                if expected_ndim == 2:
                    # For 2D target: flatten all middle dimensions
                    middle_size = 1
                    for i in range(1, arr.ndim - 1):
                        middle_size *= arr.shape[i]
                    remaining_dims = arr.shape[-1:] if arr.ndim > 2 else ()
                    new_shape = (B, middle_size) + remaining_dims
                    # print(f"    Collapsing 4D->2D: {arr.shape} -> {new_shape}")
                else:
                    # For 3D target: flatten first two middle dimensions
                    remaining_dims = arr.shape[3:]
                    new_shape = (B, arr.shape[1] * arr.shape[2]) + remaining_dims
                    # print(f"    Collapsing 4D->3D: {arr.shape} -> {new_shape}")
                arr = arr.reshape(new_shape)
        return arr

    # Apply dimension collapse where needed
    inp_np = collapse_if_needed(inp_np, 2)
    enc_pos_np = collapse_if_needed(enc_pos_np, 3)
    enc_mask_np = collapse_if_needed(enc_mask_np, 3)
    tgt_np = collapse_if_needed(tgt_np, 2)
    dec_pos_np = collapse_if_needed(dec_pos_np, 3)
    dec_mask_np = collapse_if_needed(dec_mask_np, 3)


    # Final sanity checks (optional): ensure inputs is rank-2 etc.
    assert inp_np.ndim == 2, f"inputs must be rank-2 after preprocessing, got {inp_np.shape}"
    assert enc_pos_np.ndim == 3, f"enc_pos_encoding must be rank-3 after preprocessing, got {enc_pos_np.shape}"
    assert enc_mask_np.ndim == 3, f"enc_action_mask must be rank-3 after preprocessing, got {enc_mask_np.shape}"
    assert tgt_np.ndim == 2, f"targets must be rank-2 after preprocessing, got {tgt_np.shape}"
    assert dec_pos_np.ndim == 3, f"dec_pos_encoding must be rank-3 after preprocessing, got {dec_pos_np.shape}"
    assert dec_mask_np.ndim == 3, f"dec_action_mask must be rank-3 after preprocessing, got {dec_mask_np.shape}"

    return {
        'inputs': tf.convert_to_tensor(inp_np, dtype=tf.int32),                # [B, seq]
        'enc_pos_encoding': tf.convert_to_tensor(enc_pos_np, dtype=tf.float32),# [B, seq, pos_dim]
        'enc_action_mask': tf.convert_to_tensor(enc_mask_np, dtype=tf.bool),   # [B, seq, V]
        'targets': tf.convert_to_tensor(tgt_np, dtype=tf.int32),              # [B, dec_seq]
        'dec_pos_encoding': tf.convert_to_tensor(dec_pos_np, dtype=tf.float32),# [B, dec_seq, pos_dim]
        'dec_action_mask': tf.convert_to_tensor(dec_mask_np, dtype=tf.bool),   # [B, dec_seq, V]
    }




def _sample_action(self, logits, mask, seed=None):
    """Sample action based on logits and mask"""
    # Apply mask to logits (set logits of non-actionable actions to minimum value)
    masked_logits = np.where(mask, logits, np.finfo(np.float32).min)

    # Calculate softmax
    probs = np.exp(masked_logits - np.max(masked_logits))
    probs /= np.sum(probs)
    probs = np.squeeze(probs)
    # Sample action
    action = np.random.choice(len(probs), p=probs)
    log_prob = np.log(probs[action])

    return action, log_prob


def _optimize_with_ppo_policy(self, aigs, error_rate_threshold=0.1, seed=None):
    """Optimize using PPO strategy (batch processing to avoid memory issues)"""
    if not aigs:
        return []
    
    print(f"PPO validation: processing {len(aigs)} circuits with error_threshold={error_rate_threshold}")
    
    # Use smaller batch_size to reduce memory usage and prevent OOM
    validation_batch_size = min(32, len(aigs))  # Further reduce validation batch_size to prevent OOM
    optimized_aigs = []
    
    # Process validation circuits in batches
    for start_idx in range(0, len(aigs), validation_batch_size):
        end_idx = min(start_idx + validation_batch_size, len(aigs))
        batch_aigs = aigs[start_idx:end_idx]
        
        # Process current batch
        batch_optimized = self._optimize_batch_with_ppo_policy(batch_aigs, error_rate_threshold, seed=seed)
        optimized_aigs.extend(batch_optimized)
        
        # Clean up current batch memory
        del batch_aigs
        del batch_optimized
        if 'inputs' in locals():
            del inputs
        if 'cache' in locals():
            del cache
        # Force GC after each validation batch to prevent OOM
        import gc
        gc.collect()
        
        # Clean up CUDA cache after each validation batch
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        # Remove aggressive session cleanup after validation to prevent performance degradation
        # Only perform light cleanup to preserve tf.function caches
        try:
            import gc
            gc.collect()
        except:
            pass
    
    return optimized_aigs

def _optimize_batch_with_ppo_policy(self, aigs, error_rate_threshold=0.1, seed=None):
    """Optimize a batch of circuits using PPO strategy"""
    if not aigs:
        return []
    
    # Prepare batch inputs
    batch_size = len(aigs)
    encoded_aigs = []
    tts_list = []
    enc_action_masks = []
    
    for i, aig in enumerate(aigs):
        # Calculate truth table
        tts = compute_tts(aig, num_inputs=self.num_inputs)
        tts_list.append(tts)
        
        # Encode AIG
        seq_enc, pos_enc = encode_aig(aig, self.num_inputs)
        encoded_aigs.append(self._encode_postprocess(seq_enc, pos_enc))
        
        # Generate action mask
        input_tt = self.input_tt
        enc_action_masks.append(self.generate_action_masks(tts,
                                                          input_tt,
                                                          None,
                                                          seq_enc,
                                                          use_controllability_dont_care=True,
                                                          tts_compressed=None))
    
    # Stack batch data
    enc_action_masks = np.stack(enc_action_masks)
    seq_enc, pos_enc = tuple(map(lambda x: np.stack(x, axis=0), zip(*encoded_aigs)))
    
    # Create environments
    envs = [LogicNetworkEnv(
        tts=tts_list[i],
        num_inputs=self.num_inputs,
        max_length=self.max_seq_length,
        eos_id=self.eos_id,
        pad_id=self.pad_id,
        max_tree_depth=self.max_tree_depth,
        max_inference_tree_depth=16,
        error_rate_threshold=error_rate_threshold,
        input_tt=self.input_tt,
        init_care_set_tt=None,
        use_controllability_dont_cares=True
    ) for i in range(batch_size)]
    
    # True parallel optimization
    optimized_aigs = []
    
    # Initialize batch inputs
    inputs = {
        'inputs': seq_enc,
        'enc_pos_encoding': pos_enc,
        'enc_action_mask': enc_action_masks,
        'targets': np.zeros((batch_size, 1), dtype=np.int32),
        'dec_pos_encoding': np.zeros((batch_size, 1, self.max_tree_depth * 2), dtype=np.float32),
        'dec_action_mask': np.ones((batch_size, 1, self.vocab_size), dtype=bool)
    }
    
    # Parallel optimization loop
    for step in range(self.max_seq_length):
        # Check if all environments are completed
        if all([env.is_finished for env in envs]):
            break
            
        # Batch generate action masks
        action_masks = np.stack([env.action_masks[-1] if env.action_masks else np.zeros(self.vocab_size, dtype=bool) for env in envs], axis=0)
        inputs['dec_action_mask'] = np.expand_dims(action_masks, axis=1)
        
        
        # Batch inference
        logits, pack = self._transformer_inference(inputs, return_kv_cache=True, return_last_token=True, return_value=True)
        cache = pack.get('cache', ())
        inputs['cache'] = cache
        logits = np.array(logits)
        
        # Batch apply masks
        masked_logits = np.where(action_masks, logits, np.finfo(np.float32).min)
        
        # Batch compute probabilities
        probs = np.exp(masked_logits - np.max(masked_logits, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        # Batch random action sampling (PPO strategy) - borrow from PPO training logic
        actions = []
        for i in range(batch_size):
            if not envs[i].is_finished:
                # Borrow action sampling logic from PPO training
                probs_i = probs[i].copy()
                probs_i = np.nan_to_num(probs_i, nan=0.0, posinf=0.0, neginf=0.0)
                probs_i = np.maximum(probs_i, 0)
                
                # Get action mask and ensure dimension matching
                action_mask = action_masks[i]
                mask1d = np.asarray(action_mask).reshape(-1).astype(bool)
                
                # Ensure probs_i and mask1d dimensions match
                if len(probs_i) != len(mask1d):
                    # If dimensions do not match, truncate or pad to vocab_size
                    vocab_size = envs[i].vocab_size
                    if len(probs_i) > vocab_size:
                        probs_i = probs_i[:vocab_size]
                    elif len(probs_i) < vocab_size:
                        # Pad to vocab_size
                        padded_probs = np.zeros(vocab_size)
                        padded_probs[:len(probs_i)] = probs_i
                        probs_i = padded_probs
                    
                    # Regenerate mask
                    mask1d = np.zeros(vocab_size, dtype=bool)
                    mask1d[:len(action_mask)] = action_mask
                
                probs_i[~mask1d] = 0
                
                # Normalize probabilities
                probs_sum = probs_i.sum()
                if probs_sum > 0:
                    probs_i = probs_i / probs_sum
                else:
                    probs_i = np.ones_like(probs_i) / len(probs_i)
                
                # Sample actions, handle boundary cases
                try:
                    action = np.random.choice(len(probs_i), p=probs_i)
                    # Avoid selecting PAD token
                    while not envs[i].is_finished and action == envs[i].PAD:
                        action = np.random.choice(len(probs_i), p=probs_i)
                except ValueError:
                    # Handle case where probability sum is 0
                    valid_actions = np.where(mask1d)[0]
                    if len(valid_actions) > 0:
                        # Only select valid actions (non-PAD/EOS)
                        non_special_actions = valid_actions[valid_actions >= 2]
                        if len(non_special_actions) > 0:
                            action = non_special_actions[0]
                        else:
                            action = 1  # EOS
                    else:
                        action = 1  # EOS fallback
                
                actions.append(int(action))
            else:
                actions.append(0)  # Placeholder
        
        # Batch execute actions
        for action, env in zip(actions, envs):
            if not env.is_finished:
                env.step(action)
        
        # Update input states
        pos_encodings = [env.positional_encodings[-1] for env in envs]
        pos_encodings = np.expand_dims(np.stack(pos_encodings, axis=0), axis=1)
        targets_new = np.expand_dims(actions, axis=1)
        
        if self.use_kv_cache:
            inputs['targets'] = targets_new
            inputs['dec_pos_encoding'] = pos_encodings
        else:
            inputs['targets'] = np.concatenate([inputs['targets'], targets_new], axis=1)
            inputs['dec_pos_encoding'] = np.concatenate([inputs['dec_pos_encoding'], pos_encodings], axis=1)
        
        # Clean up intermediate variables to avoid memory accumulation
        del pos_encodings, targets_new, actions, action_masks, logits, masked_logits, probs
    
    # Get optimized AIGs
    batch_optimized_aigs = []
    for i, env in enumerate(envs):
        if env.success and len(env.roots) == env.num_outputs:
            batch_optimized_aigs.append(env.roots)
        else:
            # If optimization fails or circuit is incomplete, use original AIG
            batch_optimized_aigs.append(aigs[i])
    
    # Clean up environment objects and all intermediate variables
    del envs
    del inputs
    del encoded_aigs, tts_list, enc_action_masks, seq_enc, pos_enc
    if 'cache' in locals():
        del cache
    import gc
    gc.collect()
    
    return batch_optimized_aigs


def _process_trajectories(self, trajectories, gamma, lam=0.8, log_file=None, epoch=None, epochs=None):  # Reduce lambda to decrease advantage function variance
    """Process trajectory data, calculate return and advantage function"""
    states = []
    actions = []
    old_log_probs = []
    returns = []
    advantages = []
    
    # Clear training action masks at the start of each epoch
    self._training_action_masks = []
    
    total_steps = 0
    for episode_idx, (episode, env_idx) in enumerate(trajectories):
        # Extract trajectory data
        episode_states = [step['state'] for step in episode]
        episode_actions = [step['action'] for step in episode]
        episode_rewards = [step['reward'] for step in episode]
        episode_values = [step['value'] for step in episode]
        episode_log_probs = [step['log_prob'] for step in episode]
        episode_dones = [step['done'] for step in episode]
        
        total_steps += len(episode_states)

        # Ensure data type consistency and prevent gradient flow
        episode_rewards = [float(tf.stop_gradient(r).numpy() if hasattr(r, 'numpy') else r) for r in episode_rewards]
        episode_values = [float(tf.stop_gradient(v).numpy() if hasattr(v, 'numpy') else v) for v in episode_values]
        episode_log_probs = [float(tf.stop_gradient(lp).numpy() if hasattr(lp, 'numpy') else lp) for lp in episode_log_probs]

        # Calculate Monte Carlo return
        R = 0.0
        discounted_returns = []
        for r in reversed(episode_rewards):
            R = r + gamma * R
            discounted_returns.insert(0, R)

        # Calculate Generalized Advantage Estimation (GAE)
        advantages_ep = []
        last_gae = 0.0
        next_value = 0.0
        next_done = True  # Assume episode ends when done=True

        for t in reversed(range(len(episode))):
            if t == len(episode) - 1:
                next_non_terminal = 1.0 - float(next_done)
                next_value = next_value
            else:
                next_non_terminal = 1.0 - float(episode_dones[t+1])
                next_value = episode_values[t+1]
            
            delta = episode_rewards[t] + gamma * next_value * next_non_terminal - episode_values[t]
            gae = delta + gamma * lam * next_non_terminal * last_gae
            last_gae = gae
            advantages_ep.insert(0, gae)

        # Standardize advantage function
        advantages_ep = np.array(advantages_ep, dtype=np.float32)
        # if advantages_ep.std() > 0:
        advantages_ep = (advantages_ep - advantages_ep.mean()) / (advantages_ep.std() + 1e-8)
        # Further limit advantage function range to avoid extreme values
        advantages_ep = np.clip(advantages_ep, -3.0, 3.0)

        # Add to result list
        states.extend(episode_states)
        actions.extend(episode_actions)
        old_log_probs.extend(episode_log_probs)
        returns.extend(discounted_returns)
        advantages.extend(advantages_ep)
        
        # Store action masks for each step
        for step_idx, step in enumerate(episode):
            # CRITICAL FIX: Ensure action_mask is properly copied to avoid reference issues
            action_mask_copy = np.array(step['action_mask'], dtype=bool, copy=True)
            self._training_action_masks.append(action_mask_copy)
            
        
        # Clean up episode-level variables after processing
        del episode_states, episode_actions, episode_rewards, episode_values
        del episode_log_probs, episode_dones, discounted_returns, advantages_ep
    
    print(f"Processed {len(trajectories)} episodes, total {total_steps} steps")
    print(f"Average steps per episode: {total_steps / len(trajectories):.1f}")

    # Monitor advantage function range
    if advantages:
        advantages_array = np.array(advantages)
        print(f"Advantages range: min={advantages_array.min():.4f}, max={advantages_array.max():.4f}, std={advantages_array.std():.4f}")
        # Log trajectory processing details
        if log_file is not None and epoch is not None and epochs is not None:
            def write_trajectory_stats():
                log_file.write(f"Epoch {epoch+1}/{epochs} | Processed {len(trajectories)} episodes, total {total_steps} steps, avg {total_steps / len(trajectories):.1f} steps/episode\n")
                log_file.write(f"Epoch {epoch+1}/{epochs} | Advantages range: min={advantages_array.min():.4f}, max={advantages_array.max():.4f}, std={advantages_array.std():.4f}\n")
                log_file.flush()
            safe_file_operation(write_trajectory_stats)
        del advantages_array
    
    # Clean up memory - delete temporary variables
    import gc
    gc.collect()
    
    return states, actions, old_log_probs, returns, advantages



def _update_policy(self, inputs_batch, actions, old_log_probs, advantages, optimizer, clip_ratio, target_kl, batch_size, ppo_train_epoch, log_file=None, epoch=None, epochs=None):
    """
    Avoid creating recursive @tf.function to avoid retracing.
    - states: list of state dicts
    - actions, old_log_probs, advantages: list/np arrays
    """
    inputs_batch = self._prepare_batch_input(inputs_batch)
    # for key, tensor in inputs_batch.items():
    #     print(f"{key}: {tensor.shape}")
    actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
    old_log_probs_tensor = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
    advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)

    training_action_masks = tf.convert_to_tensor(self._training_action_masks, dtype=tf.bool)
    training_action_masks = tf.expand_dims(training_action_masks, axis=1)  # [batch_size, 1, vocab_size]
    
    dataset = tf.data.Dataset.from_tensor_slices((
        inputs_batch['inputs'],
        inputs_batch['enc_pos_encoding'],
        inputs_batch['enc_action_mask'],
        inputs_batch['targets'],
        inputs_batch['dec_pos_encoding'],
        training_action_masks,  # Use stored action masks
        actions_tensor,
        old_log_probs_tensor,
        advantages_tensor
    )).shuffle(len(actions), reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Filter out policy variables (not including value head)
    policy_vars = [v for v in self._transformer.trainable_variables if 'value_head' not in v.name]

    # Define (and cache) train step in graph mode (only once)
    if not hasattr(self, "_policy_train_step_fn") or self._policy_train_step_fn is None:
        @tf.function(reduce_retracing=True)
        def _policy_train_step(inputs_tokens, enc_pos, enc_action_mask, targets, dec_pos, batch_action_masks, batch_actions, batch_old_log_probs, batch_advantages):
            # Construct model inputs dict matching your call signature
            model_inputs = {
                'inputs': inputs_tokens,
                'enc_pos_encoding': enc_pos,
                'enc_action_mask': enc_action_mask,
                'targets': targets,
                'dec_pos_encoding': dec_pos,
                'dec_action_mask': batch_action_masks
            }

            with tf.GradientTape() as tape:
                # Use same inference method as sampling for complete consistency
                logits, pack = self._transformer_inference(model_inputs, return_kv_cache=True, return_last_token=True, return_value=True)
                # Ensure data type consistency with sampling
                logits = tf.cast(logits, tf.float32)  # shape [B, V]

                # Use fixed temperature 1.0 for consistency with training
                logits = logits / 1.0

                enc_action_mask = enc_action_mask[:, -1, :]  # shape [B, V]
                # Log prob - Keep calculation method completely consistent with sampling
                # 1. Manual numerical stability processing (consistent with sampling)
                logits_max = tf.reduce_max(logits, axis=-1, keepdims=True)
                logits_shifted = logits - logits_max
                probs = tf.exp(logits_shifted)
                probs_sum = tf.reduce_sum(probs, axis=-1, keepdims=True)
                probs = probs / (probs_sum + 1e-12)

                # 2. Calculate log probability (completely consistent with sampling)
                # Directly take the probability of selected action, not one-hot weighted
                batch_size = tf.shape(batch_actions)[0]
                batch_indices = tf.range(batch_size)
                indices = tf.stack([batch_indices, batch_actions], axis=1)
                selected_probs = tf.gather_nd(probs, indices)
                new_log_probs = tf.math.log(selected_probs + 1e-12)

                # Calculate log probability difference with numerical stability
                log_prob_diff = new_log_probs - batch_old_log_probs
                # Clip log probability difference to prevent exponential explosion
                log_prob_diff_clipped = tf.clip_by_value(log_prob_diff, -1.0, 1.0)  # Prevents exp() overflow
                ratio = tf.exp(log_prob_diff_clipped)

                # Monitor ratio range to avoid overly aggressive policy updates
                ratio_mean = tf.reduce_mean(ratio)
                ratio_max = tf.reduce_max(ratio)
                ratio_min = tf.reduce_min(ratio)
                
                # Standard PPO clipping - clip the raw ratio, not pre-clipped
                unclipped = ratio * batch_advantages
                clipped = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
                ppo_clip_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))
                
                # Calculate KL divergence consistently
                # Since transformer already applies action mask internally, use same probabilities as ratio calculation
                # Use the stored action masks for validation only
                training_action_mask = batch_action_masks[:, 0, :]  # [batch_size, vocab_size]
                
                # For KL calculation, use the same new_log_probs we used for ratio calculation
                # This ensures perfect consistency between ratio and KL computation
                masked_new_log_probs = new_log_probs
                
                
                # Convert log probabilities to probabilities
                old_probs = tf.exp(batch_old_log_probs)
                new_probs = tf.exp(masked_new_log_probs)
                
                # Calculate ratio r = p_old/p_new
                # Add small epsilon to avoid division by zero
                epsilon = 1e-8
                ratio = old_probs / (new_probs + epsilon)
                
                # Apply Schulman (2020) KL approximation: r - log(r) - 1
                # This form is more stable than standard KL divergence
                kl_terms = ratio - tf.math.log(ratio + epsilon) - 1.0
                
                # Scale down to match standard KL divergence scale
                # Schulman KL tends to be larger, so we scale it down
                kl_terms_scaled = kl_terms * 0.3  # Scale factor to match standard KL scale
                
                # Clip extreme values for numerical stability
                kl_terms_clipped = tf.clip_by_value(kl_terms_scaled, -2.0, 2.0)
                
                # Calculate mean KL divergence
                approx_kl = tf.reduce_mean(kl_terms_clipped)
                
                # Add KL penalty to control policy updates (common practice in PPO)
                # Use higher KL penalty coefficient for stricter control
                kl_coeff = 2.0  # Higher KL penalty coefficient for stricter control
                kl_penalty = kl_coeff * tf.maximum(0.0, approx_kl - target_kl)
                policy_loss = ppo_clip_loss + kl_penalty

            grads = tape.gradient(policy_loss, policy_vars)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, policy_vars))
            
            # Clean up memory
            del grads
            
            # Return ratio statistics for debugging
            return policy_loss, approx_kl, ratio_mean, ratio_max, ratio_min, new_log_probs, log_prob_diff

        self._policy_train_step_fn = _policy_train_step

    # run multiple epochs over dataset (batch)
    total_loss = 0.0
    total_kl = 0.0
    iters = 0

    for ppo_epoch in range(ppo_train_epoch):
        for batch_idx, batch in enumerate(dataset):
            inputs_tokens, enc_pos, enc_action_mask, targets, dec_pos, batch_action_masks, batch_actions, batch_old_log_probs, batch_advantages = batch
            
            # print(f"inputs_tokens(batch):{inputs_tokens.shape}")
            # print(f"enc_pos(batch)::{enc_pos.shape}")
            # print(f"enc_action_mask(batch):{enc_action_mask.shape}")
            loss_val, kl_val, ratio_mean_val, ratio_max_val, ratio_min_val, new_log_probs_val, log_prob_diff_val = self._policy_train_step_fn(inputs_tokens, enc_pos, enc_action_mask, targets, dec_pos, batch_action_masks, batch_actions, batch_old_log_probs, batch_advantages)
            total_loss += float(loss_val)
            total_kl += float(kl_val)
            iters += 1
            
            # Monitor KL divergence and ratio (print every 20 batches)
            if iters % 20 == 1:
                print(f"Batch {iters}: KL={total_kl/iters:.4f}, Loss={total_loss/iters:.4f}, Ratio mean={float(ratio_mean_val):.4f}, max={float(ratio_max_val):.4f}, min={float(ratio_min_val):.4f}")

                def write_policy_stats():
                    log_file.write(f"Epoch {epoch+1}/{epochs} | Policy Batch {iters}: KL={total_kl/iters:.4f}, Loss={total_loss/iters:.4f}, Ratio mean={float(ratio_mean_val):.4f}, max={float(ratio_max_val):.4f}, min={float(ratio_min_val):.4f}\n")
                    # Convert tensors to numpy values to avoid graph retention
                    # Use tf.stop_gradient to prevent gradient flow during logging
                    old_log_probs_mean = float(tf.stop_gradient(tf.reduce_mean(batch_old_log_probs)).numpy())
                    old_log_probs_std = float(tf.stop_gradient(tf.math.reduce_std(batch_old_log_probs)).numpy())
                    new_log_probs_mean = float(tf.stop_gradient(tf.reduce_mean(new_log_probs_val)).numpy())
                    new_log_probs_std = float(tf.stop_gradient(tf.math.reduce_std(new_log_probs_val)).numpy())
                    log_prob_diff_mean = float(tf.stop_gradient(tf.reduce_mean(log_prob_diff_val)).numpy())
                    log_prob_diff_std = float(tf.stop_gradient(tf.math.reduce_std(log_prob_diff_val)).numpy())

                    log_file.write(f"Epoch {epoch+1}/{epochs} |   Old log probs: mean={old_log_probs_mean:.4f}, std={old_log_probs_std:.4f}\n")
                    log_file.write(f"Epoch {epoch+1}/{epochs} |   New log probs: mean={new_log_probs_mean:.4f}, std={new_log_probs_std:.4f}\n")
                    log_file.write(f"Epoch {epoch+1}/{epochs} |   Log prob diff: mean={log_prob_diff_mean:.4f}, std={log_prob_diff_std:.4f}\n")
                    log_file.flush()
                safe_file_operation(write_policy_stats)
            # Enable KL divergence check to prevent overly large policy updates
            if (total_kl / iters) > target_kl:  # Stricter early stopping
                print(f"Early stopping at KL divergence {total_kl/iters:.4f} > {target_kl:.4f}")
                def write_early_stopping():
                    log_file.write(f"Epoch {epoch+1}/{epochs} | Early stopping at KL divergence {total_kl/iters:.4f} > {target_kl:.4f}\n")
                    log_file.flush()
                safe_file_operation(write_early_stopping)
                break
        # Enable KL divergence check to prevent overly large policy updates
        if (iters > 0) and (total_kl / iters) > 1.5 * target_kl:
            break

    avg_loss = total_loss / max(1, iters)
    
    # Clean up memory
    import gc
    gc.collect()
    
    return avg_loss


def _update_value_function(self, inputs_batch, returns, optimizer, batch_size, ppo_train_epoch):
    """
    Update value head (MSE), avoid retracing.
    - states: list of state dicts
    - returns: list/np array
    """

    # Prepare data for value function training
    returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
    inputs_batch = self._prepare_batch_input(inputs_batch)
    
    # Create value dataset fresh each time to avoid memory accumulation
    dataset = tf.data.Dataset.from_tensor_slices((
        inputs_batch['inputs'],
        inputs_batch['enc_pos_encoding'],
        inputs_batch['enc_action_mask'],
        inputs_batch['targets'],
        inputs_batch['dec_pos_encoding'],
        inputs_batch['dec_action_mask'],
        returns_tensor
    )).shuffle(len(returns)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Value vars: assume value_head is named 'value_head' in transformer
    try:
        value_vars = self._transformer.value_head.trainable_variables
    except Exception:
        # Fallback: filter out variables with name containing 'value' in all trainable variables
        value_vars = [v for v in self._transformer.trainable_variables if 'value' in v.name]

    # Define and cache value train step
    if not hasattr(self, "_value_train_step_fn") or self._value_train_step_fn is None:
        @tf.function(reduce_retracing=True)
        def _value_train_step(inputs_tokens, enc_pos, enc_action_mask, targets, dec_pos, dec_action_mask, batch_returns):
            model_inputs = {
                'inputs': inputs_tokens,
                'enc_pos_encoding': enc_pos,
                'enc_action_mask': enc_action_mask,
                'targets': targets,
                'dec_pos_encoding': dec_pos,
                'dec_action_mask': dec_action_mask
            }
            with tf.GradientTape() as tape:
                _, pack = self._transformer(model_inputs, training=False, return_last_token=True, return_value=True)
                values = pack['value']
                values = tf.cast(values, tf.float32)  # [B, 1] or [B]
                # Ensure shape [B]
                values_flat = tf.reshape(values, [-1])
                returns_flat = tf.cast(batch_returns, tf.float32)
                
                # Use Huber loss for stability (less sensitive to outliers than MSE)
                delta = 2.0  # Huber loss threshold - larger delta makes it behave more like MSE
                huber_loss = tf.where(
                    tf.abs(returns_flat - values_flat) < delta,
                    0.5 * tf.square(returns_flat - values_flat),
                    delta * tf.abs(returns_flat - values_flat) - 0.5 * delta * delta
                )
                value_loss = tf.reduce_mean(huber_loss)
            grads = tape.gradient(value_loss, value_vars)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, value_vars))
            
            # Clean up memory
            del grads
            
            return value_loss

        self._value_train_step_fn = _value_train_step

    total_value_loss = 0.0
    iters = 0
    for epoch in range(ppo_train_epoch):
        for batch in dataset:
            inputs_tokens, enc_pos, enc_action_mask, targets, dec_pos, dec_action_mask, batch_returns = batch
            loss_val = self._value_train_step_fn(inputs_tokens, enc_pos, enc_action_mask, targets, dec_pos, dec_action_mask, batch_returns)
            total_value_loss += float(loss_val)
            iters += 1

    avg_value_loss = total_value_loss / max(1, iters)
    
    # Clean up memory
    import gc
    gc.collect()
    
    return avg_value_loss


def _log_prob(self, logits, actions):
    """Calculate log probability of given actions"""
    logits = tf.cast(logits, tf.float32)

    # Use same softmax computation as during sampling and training
    # Manual numerical stability processing (consistent with sampling)
    logits_max = tf.reduce_max(logits, axis=-1, keepdims=True)
    logits_shifted = logits - logits_max
    probs = tf.exp(logits_shifted)
    probs_sum = tf.reduce_sum(probs, axis=-1, keepdims=True)
    probs = probs / (probs_sum + 1e-12)

    # Create one-hot encoding of actions
    actions_one_hot = tf.one_hot(actions, depth=self.vocab_size, dtype=tf.float32)

    # Calculate log probability
    return tf.math.log(tf.reduce_sum(probs * actions_one_hot, axis=-1) + 1e-10)


def _batch_step(self, envs, actions):
    """Batch execute environment steps (vectorized operation)"""
    next_states = []
    rewards = []
    dones = []

    for env, action in zip(envs, actions):
        next_state, reward, done, _ = env.ppo_step(action)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)

    return next_states, np.array(rewards), np.array(dones)


@tf.function(reduce_retracing=True)
def value_forward_pass(self, inputs, returns):
    """Value network forward pass (graph mode)"""
    # Predict value
    _, pack = self._transformer(inputs, training=True, return_value=True)
    values = pack['value']
    # Ensure data type consistency - convert values to float32
    values = tf.cast(values, tf.float32)
    returns = tf.cast(returns, tf.float32)

    # Calculate value loss
    return tf.reduce_mean(tf.square(returns - values))


@tf.function(reduce_retracing=True)
def policy_forward_pass(self, inputs, actions, old_log_probs, advantages, clip_ratio):
    """Policy network forward pass (graph mode)"""
    # Get policy network output
    logits, _ = self._transformer(inputs, training=True, return_value=True)

    # Ensure logits is float32 type
    logits = tf.cast(logits, tf.float32)

    # Calculate log probability of new policy
    new_log_probs = self._log_prob(logits, actions)

    # Ensure all tensors are float32 type
    new_log_probs = tf.cast(new_log_probs, tf.float32)
    old_log_probs = tf.cast(old_log_probs, tf.float32)
    advantages = tf.cast(advantages, tf.float32)

    # Calculate PPO loss
    ratio = tf.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

    # Calculate KL divergence using Schulman (2020) approximation
    # KL[θ_FT || θ_BASE] = Σ(r - log(r) - 1) where r = p_old/p_new
    # Convert log probabilities to probabilities
    old_probs = tf.exp(old_log_probs)
    new_probs = tf.exp(new_log_probs)
    
    # Calculate ratio r = p_old/p_new
    epsilon = 1e-8
    ratio = old_probs / (new_probs + epsilon)
    
    # Apply Schulman (2020) KL approximation: r - log(r) - 1
    kl_terms = ratio - tf.math.log(ratio + epsilon) - 1.0
    
    # Scale down to match standard KL divergence scale
    kl_terms_scaled = kl_terms * 0.3  # Scale factor to match standard KL scale
    kl_terms_clipped = tf.clip_by_value(kl_terms_scaled, -2.0, 2.0)
    kl = tf.reduce_mean(kl_terms_clipped)
    
    return policy_loss, kl, new_log_probs 


def _calculate_comprehensive_reward(self, optimized_roots, original_roots, baseline_area, error_rate_threshold, w_area=1.0, w_delay=0.5, w_error=2.0, log_file=None):
    """
    Calculate comprehensive reward including area, delay, and error rate metrics
    
    Args:
        optimized_roots: List of optimized circuit root nodes
        original_roots: List of original circuit root nodes
        baseline_area: Baseline area (optimal number of AND gates)
        error_rate_threshold: Error rate threshold
        w_area: Area weight
        w_delay: Delay weight
        w_error: Error weight
    
    Returns:
        final_reward: Comprehensive reward value
    """
    from circuit_transformer.utils import count_num_ands, compute_critical_path, checkER
    
    # Calculate optimized metrics
    optimized_area = count_num_ands(optimized_roots)
    optimized_delay = compute_critical_path(optimized_roots)

    # Calculate original circuit metrics
    original_area = count_num_ands(original_roots)
    original_delay = compute_critical_path(original_roots)

    # Calculate error rate, handle incomplete circuits
    try:
        error_rate = checkER(original_roots, optimized_roots)
    except (ValueError, AttributeError) as e:
        # If circuit is incomplete (contains None nodes), give severe penalty
        # print(f"Circuit incomplete (None nodes detected), reward = -1.0")
        return -1.0

    else:
        # Calculate improvements
        # Use tf.stop_gradient for baseline values to prevent gradient flow
        baseline_area_stopped = tf.stop_gradient(tf.constant(baseline_area, dtype=tf.float32))
        area_improvement = baseline_area_stopped - optimized_area  # Compare with baseline
        delay_improvement = original_delay - optimized_delay  # Compare with original circuit

        # Normalize improvements (avoid division by zero)
        # Use tf.stop_gradient for normalization to prevent gradient flow
        baseline_area_norm = tf.stop_gradient(tf.constant(max(baseline_area, 1), dtype=tf.float32))
        original_delay_norm = tf.stop_gradient(tf.constant(max(original_delay, 1), dtype=tf.float32))
        area_reward = area_improvement / baseline_area_norm if baseline_area > 0 else 0
        delay_reward = delay_improvement / original_delay_norm if original_delay > 0 else 0
        
        # Error reward (smaller error is better)
        error_reward = max(0, 1.0 - error_rate / error_rate_threshold) if error_rate_threshold > 0 else 1.0
        
        # Combined reward
        final_reward = w_area * area_reward + w_delay * delay_reward + w_error * error_reward
        
        # Ensure reward is in reasonable range to avoid tanh saturation
        # Ideal range: -2 to 2, so tanh gives about -0.96 to 0.96, maintaining discrimination
        final_reward = tf.clip_by_value(final_reward, -2.0, 2.0)
        
        # Use tanh for smoothing, limit reward range
        final_reward = tf.tanh(final_reward)
        
        # Convert to numpy for return
        final_reward = float(final_reward.numpy())

        # Debug info: show reward breakdown
        if final_reward > 0.3:  # Print medium and above reward cases
            raw_reward = w_area * area_reward + w_delay * delay_reward + w_error * error_reward

            if log_file is not None:
                def write_reward_breakdown():
                    log_file.write(f"High reward case: Area {original_area}->{optimized_area} (baseline:{baseline_area}, +{area_improvement}), "
                                  f"Delay {original_delay}->{optimized_delay} (+{delay_improvement:.1f}), "
                                  f"Error {error_rate:.4f} (threshold:{error_rate_threshold:.4f})\n")
                    log_file.write(f"  Raw components: area={area_reward:.3f}*{w_area}={w_area*area_reward:.3f}, "
                                  f"delay={delay_reward:.3f}*{w_delay}={w_delay*delay_reward:.3f}, "
                                  f"error={error_reward:.1f}*{w_error}={w_error*error_reward:.3f}\n")
                    log_file.write(f"  Raw total: {raw_reward:.3f} -> Clipped: {np.clip(raw_reward, -2.0, 2.0):.3f} -> Final: {final_reward:.4f}\n")
                    log_file.flush()
                safe_file_operation(write_reward_breakdown)
    
    return final_reward 