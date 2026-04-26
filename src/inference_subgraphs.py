#!/usr/bin/env python3
"""
Inference script for subgraphs using Scalable Approximate Circuit Transformer(GTAC)

This script performs circuit optimization on subgraphs using the Approximate Circuit Transformer(GTAC) model.
"""

import os
from pathlib import Path
import numpy as np
import time
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.scalable_model import ScalableCircuitTransformer
from src.dynamic_encoding import DynamicEncoder
from src.utils import read_aiger, write_aiger, count_num_ands, compute_tts, compute_input_tt, plot_network


def parse_aig_header(aig_file: str) -> dict:
    """Parse AIG file header to get circuit metadata."""
    with open(aig_file, 'rb') as f:
        first_line = f.readline().decode('ascii').strip()
        parts = first_line.split()

        if parts[0] not in ['aig', 'aag']:
            raise ValueError(f"Not a valid AIG file: {aig_file}")

        return {
            'format': parts[0],
            'M': int(parts[1]),
            'I': int(parts[2]),
            'L': int(parts[3]),
            'O': int(parts[4]),
            'A': int(parts[5])
        }


def load_subgraphs(subgraph_dir: str, max_subgraphs: int = None):
    """
    Load subgraph files from directory.

    Returns:
        List of (filepath, metadata) tuples
    """
    subgraph_files = sorted(Path(subgraph_dir).glob("*.aig"))
    subgraph_files += sorted(Path(subgraph_dir).glob("*.aag"))

    if max_subgraphs:
        subgraph_files = subgraph_files[:max_subgraphs]

    subgraphs = []
    for aig_file in subgraph_files:
        try:
            metadata = parse_aig_header(str(aig_file))
            subgraphs.append((str(aig_file), metadata))
        except Exception as e:
            print(f"Warning: Could not parse {aig_file}: {e}")
            continue

    return subgraphs


def optimize_single_circuit(model: ScalableCircuitTransformer,
                            roots: list,
                            num_inputs: int,
                            num_outputs: int,
                            max_seq_length: int = None,
                            num_mcts_steps: int = 0,
                            use_controllability_dont_cares: bool = True,
                            overflow_option: str = 'origin'):
    """
    Optimize a single circuit using the model's autoregressive decoding.

    Args:
        model: SACT model
        roots: Circuit roots
        num_inputs: Number of inputs
        num_outputs: Number of outputs
        max_seq_length: Maximum sequence length for inference
        num_mcts_steps: Number of MCTS steps (0 for greedy decoding)
        use_controllability_dont_cares: Use controllability don't cares

    Returns:
        Optimized circuit roots or None if failed
    """
    if max_seq_length is None:
        max_seq_length = model.max_seq_length

    try:
        # Ensure roots is a list
        if not isinstance(roots, list):
            roots = [roots]
        
        # Use model's optimize_batch method for actual autoregressive decoding
        # This uses DynamicEncoder (from dynamic_encoding.py) for encoding, same as training
        # This implements:
        # 1. Encoding the input circuit with dynamic encoding (model.encoder.encode_aig)
        # 2. Using the transformer to generate optimized sequence autoregressively
        # 3. Decoding the sequence back to circuit roots
        # 4. Verifying equivalence via environment success flag
        
        optimized_roots = model.optimize_batch(
            aigs=[roots],  # Pass as list for batch processing
            max_inference_seq_length=max_seq_length,
            max_mcts_inference_seq_length=max_seq_length,
            context_num_inputs=None,
            input_tts=None,  # Will use model.get_input_tt(num_inputs) internally
            care_set_tts=None,
            ffws=None,
            context_hash_list=None,
            max_inference_reward_list=None,
            num_mcts_steps=num_mcts_steps,
            num_leaf_parallelization=8,
            num_mcts_playouts_per_step=10,
            use_controllability_dont_cares=use_controllability_dont_cares,
            tts_compressed=None,
            overflow_option=overflow_option,  # Return original if optimization fails
            return_envs=False,
            return_mcts_roots=False,
            return_input_encodings=False,
            puct_explore_ratio=1.,
            w_gate=model.w_gate,
            w_delay=model.w_delay
        )
        
        # optimize_batch returns a list of optimized circuits
        if optimized_roots and len(optimized_roots) > 0:
            return optimized_roots[0]
        else:
            return None

    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        if model.verbose > 0:
            traceback.print_exc()
        return None


def inference_on_subgraph(model: ScalableCircuitTransformer,
                         subgraph_file: str,
                         output_dir: str = None,
                         verbose: bool = True,
                         overflow_option: str = 'origin'):
    """
    Perform inference on a single subgraph.

    Args:
        model: SACT model
        subgraph_file: Path to subgraph .aig file
        output_dir: Directory to save optimized circuit (optional)
        verbose: Print progress

    Returns:
        dict with results
    """
    if verbose:
        print(f"\nOptimizing: {Path(subgraph_file).name}")

    metadata = parse_aig_header(subgraph_file)
    num_inputs = metadata['I']
    num_outputs = metadata['O']

    # Read circuit
    roots, _ = read_aiger(subgraph_file)

    # Ensure roots is a list
    if not isinstance(roots, list):
        roots = [roots]

    num_ands_original = count_num_ands(roots)

    if verbose:
        print(f"  Original: {num_inputs}I x {num_outputs}O, {num_ands_original} ANDs")

    try:
        start_time = time.time()

        # Perform optimization
        optimized_roots = optimize_single_circuit(
            model, roots, num_inputs, num_outputs,
            max_seq_length=model.max_seq_length,
            use_controllability_dont_cares=True,
            overflow_option=overflow_option
        )

        elapsed_time = time.time() - start_time

        if optimized_roots is None:
            if verbose:
                print(f"  ✗ Optimization failed")
            return {
                'file': Path(subgraph_file).name,
                'success': False,
                'error': 'Optimization returned None'
            }

        # Ensure optimized_roots is a list
        if not isinstance(optimized_roots, list):
            optimized_roots = [optimized_roots]

        num_ands_optimized = count_num_ands(optimized_roots)
        reduction = (num_ands_original - num_ands_optimized) / max(num_ands_original, 1) * 100

        # Save optimized circuit if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = Path(output_dir) / Path(subgraph_file).name
            # Pass num_inputs to ensure all inputs are preserved in the same order as original
            write_aiger(optimized_roots, str(output_file), num_inputs=num_inputs)
            # plot_network(optimized_roots, num_inputs, num_outputs, filename=str(output_file)+".pdf")

        result = {
            'file': Path(subgraph_file).name,
            'success': True,
            'inputs': num_inputs,
            'outputs': num_outputs,
            'ands_original': num_ands_original,
            'ands_optimized': num_ands_optimized,
            'reduction': reduction,
            'time': elapsed_time
        }

        if verbose:
            print(f"  ✓ Optimized: {num_ands_optimized} ANDs ({reduction:+.2f}%), {elapsed_time:.2f}s")

        return result

    except Exception as e:
        if verbose:
            print(f"  ✗ Failed: {e}")
        return {
            'file': Path(subgraph_file).name,
            'success': False,
            'error': str(e)
        }


def batch_inference(model: ScalableCircuitTransformer,
                   subgraph_dir: str,
                   output_dir: str = None,
                   max_subgraphs: int = None,
                   overflow_option: str = 'origin',
                   batch_size: int = 1):
    """
    Perform inference on all subgraphs in a directory with batch processing.

    Args:
        model: SACT model
        subgraph_dir: Directory containing subgraph .aig files
        output_dir: Directory to save optimized circuits
        max_subgraphs: Maximum number of subgraphs to process
        batch_size: Number of circuits to process per batch (default: 1, sequential)

    Returns:
        List of results
    """
    print("\n" + "=" * 70)
    print("Batch Inference on Subgraphs")
    print("=" * 70)

    # Load subgraphs
    subgraphs = load_subgraphs(subgraph_dir, max_subgraphs)
    print(f"\nFound {len(subgraphs)} subgraph files")

    if len(subgraphs) == 0:
        print("No subgraph files found!")
        return []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    start_time = time.time()

    if batch_size <= 1:
        for i, (subgraph_file, metadata) in enumerate(subgraphs):
            print(f"\n[{i+1}/{len(subgraphs)}]", end=" ")
            result = inference_on_subgraph(
                model, subgraph_file, output_dir, verbose=True, overflow_option=overflow_option
            )
            results.append(result)
    else:
        num_batches = (len(subgraphs) + batch_size - 1) // batch_size
        print(f"Using batch_size={batch_size}, total batches={num_batches}")
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(subgraphs))
            batch = subgraphs[start_idx:end_idx]
            print(f"\n[{batch_idx + 1}/{num_batches}] Processing batch ({len(batch)} circuits)...")
            # Prepare batch roots and metadata
            batch_roots = []
            batch_meta = []
            for subgraph_file, metadata in batch:
                roots, _ = read_aiger(subgraph_file)
                if not isinstance(roots, list):
                    roots = [roots]
                batch_roots.append(roots)
                batch_meta.append({
                    'file': Path(subgraph_file).name,
                    'path': subgraph_file,
                    'inputs': metadata['I'],
                    'outputs': metadata['O'],
                    'ands_original': count_num_ands(roots),
                })

            batch_start = time.time()
            try:
                optimized_roots_list = model.optimize_batch(
                    aigs=batch_roots,
                    max_inference_seq_length=model.max_seq_length,
                    max_mcts_inference_seq_length=model.max_seq_length,
                    context_num_inputs=None,
                    input_tts=None,
                    care_set_tts=None,
                    ffws=None,
                    context_hash_list=None,
                    max_inference_reward_list=None,
                    num_mcts_steps=0,
                    num_leaf_parallelization=8,
                    num_mcts_playouts_per_step=10,
                    use_controllability_dont_cares=True,
                    tts_compressed=None,
                    overflow_option=overflow_option,
                    return_envs=False,
                    return_mcts_roots=False,
                    return_input_encodings=False,
                    puct_explore_ratio=1.,
                    w_gate=model.w_gate,
                    w_delay=model.w_delay
                )
            except Exception as e:
                print(f"  ✗ Batch optimization failed: {e}")
                import traceback
                traceback.print_exc()
                for meta in batch_meta:
                    results.append({
                        'file': meta['file'],
                        'success': False,
                        'error': str(e),
                    })
                continue

            batch_elapsed = time.time() - batch_start
            if optimized_roots_list is None or len(optimized_roots_list) != len(batch_meta):
                msg = "Batch optimization returned unexpected result size"
                print(f"  ✗ {msg}")
                for meta in batch_meta:
                    results.append({
                        'file': meta['file'],
                        'success': False,
                        'error': msg,
                    })
                continue

            for i_local, (optimized_roots, meta) in enumerate(zip(optimized_roots_list, batch_meta)):
                if optimized_roots is None:
                    results.append({
                        'file': meta['file'],
                        'success': False,
                        'error': 'Optimization returned None',
                    })
                    continue

                if not isinstance(optimized_roots, list):
                    optimized_roots = [optimized_roots]
                num_ands_optimized = count_num_ands(optimized_roots)
                reduction = (meta['ands_original'] - num_ands_optimized) / max(meta['ands_original'], 1) * 100

                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = Path(output_dir) / meta['file']
                    write_aiger(optimized_roots, str(output_file), num_inputs=meta['inputs'])

                result = {
                    'file': meta['file'],
                    'success': True,
                    'inputs': meta['inputs'],
                    'outputs': meta['outputs'],
                    'ands_original': meta['ands_original'],
                    'ands_optimized': num_ands_optimized,
                    'reduction': reduction,
                    'time': batch_elapsed / max(1, len(batch_meta)),
                }
                results.append(result)
                print(f"  [{i_local + 1}/{len(batch_meta)}] {meta['file']}: {num_ands_optimized} ANDs ({reduction:+.2f}%)")

    elapsed_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 70)
    print("Batch Inference Summary")
    print("=" * 70)

    success_count = sum(1 for r in results if r['success'])
    print(f"\nTotal circuits: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Avg time per circuit: {elapsed_time/len(results):.2f}s")

    # Print optimization statistics for successful optimizations
    if success_count > 0:
        successful_results = [r for r in results if r['success']]
        total_ands_original = sum(r['ands_original'] for r in successful_results)
        total_ands_optimized = sum(r['ands_optimized'] for r in successful_results)
        avg_reduction = (total_ands_original - total_ands_optimized) / total_ands_original * 100

        print(f"\nOptimization Statistics:")
        print(f"  Total ANDs (original): {total_ands_original}")
        print(f"  Total ANDs (optimized): {total_ands_optimized}")
        print(f"  Average reduction: {avg_reduction:.2f}%")

        # Individual statistics
        reductions = [r['reduction'] for r in successful_results]
        print(f"  Best reduction: {max(reductions):.2f}%")
        print(f"  Worst reduction: {min(reductions):.2f}%")
        print(f"  Median reduction: {np.median(reductions):.2f}%")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Inference on subgraphs with SACT')
    parser.add_argument('--subgraph-dir', type=str, required=True,
                       help='Directory containing subgraph .aag files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save optimized circuits')
    parser.add_argument('--ckpt-path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--max-subgraphs', type=int, default=None,
                       help='Maximum number of subgraphs to process')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of circuits per batch (default: 1, sequential)')
    parser.add_argument('--max-inputs', type=int, default=256,
                       help='Maximum number of inputs to support')
    parser.add_argument('--max-outputs', type=int, default=256,
                       help='Maximum number of outputs to support')
    parser.add_argument('--error-threshold', type=float, default=0.1,
                       help='Error threshold for optimization')
    parser.add_argument('--use-memory-dfs', action='store_true',
                       help='Use memory DFS encoding (default: False, uses memoryless DFS)')
    parser.add_argument('--inference-temperature', type=float, default=1.0,
                       help='Temperature for AR decoding (T>1 softens overconfidence; try 1.2 or 1.4 if AR acc is low)')
    parser.add_argument('--decode-top-k', type=int, default=0,
                       help='Top-k decoding in AR inference (0 disables top-k)')
    parser.add_argument('--decode-low-conf-margin', type=float, default=0.0,
                       help='Enable stochastic fallback when top1-top2 margin is below threshold')
    parser.add_argument('--decode-beam-size', type=int, default=1,
                       help='Lightweight lookahead beam size (1 disables beam)')
    parser.add_argument('--decode-beam-until-step', type=int, default=0,
                       help='Enable beam for early steps only (0 disables early-step beam; REF-step beam still enabled when beam size > 1)')
    parser.add_argument('--decode-beam-lookahead-weight', type=float, default=0.15,
                       help='Lookahead weight for next-mask fanout in beam scoring')
    parser.add_argument('--decode-beam-lookahead-steps', type=int, default=1,
                       help='n-step partial rollout depth for beam candidate scoring (1 keeps old behavior)')
    parser.add_argument('--decode-beam-logprob-weight', type=float, default=1.0,
                       help='Weight for log-probability term in beam score')
    parser.add_argument('--decode-beam-delay-delta-weight', type=float, default=0.0,
                       help='Weight for normalized delay improvement term in beam score')
    parser.add_argument('--decode-beam-step-reward-weight', type=float, default=0.0,
                       help='Weight for env.step immediate reward in beam scoring (0 disables)')
    parser.add_argument('--decode-beam-area-growth-penalty', type=float, default=0.0,
                       help='Penalty per one-step AND growth in beam scoring (0 disables)')
    parser.add_argument('--decode-beam-ref-token-penalty', type=float, default=0.0,
                       help='Extra penalty when candidate token is REF_TOKEN (0 disables)')
    parser.add_argument('--disable-temp-sampling', action='store_true',
                       help='Use greedy argmax even when temperature != 1.0')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to JSON file')
    parser.add_argument('--overflow-option', type=str, default='origin',
                       help='Overflow option: origin (return original circuit) or env (return optimized circuit from environment)')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Scalable Circuit Transformer - Inference on Subgraphs")
    print("=" * 70)
    print(f"Subgraph directory: {args.subgraph_dir}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Max inputs/outputs: {args.max_inputs}/{args.max_outputs}")
    print(f"Use memory DFS encoding: {args.use_memory_dfs}")
    print(f"Inference temperature: {args.inference_temperature}")
    print(f"Batch size: {args.batch_size}")
    print(f"Decode top-k: {args.decode_top_k}, low-conf-margin: {args.decode_low_conf_margin}")
    print(
        f"Decode beam: size={args.decode_beam_size}, until_step={args.decode_beam_until_step}, "
        f"lookahead_w={args.decode_beam_lookahead_weight}, n_step={args.decode_beam_lookahead_steps}, "
        f"logprob_w={args.decode_beam_logprob_weight}, delay_delta_w={args.decode_beam_delay_delta_weight}"
    )
    print(
        f"Beam extra score: step_reward_w={args.decode_beam_step_reward_weight}, "
        f"area_growth_penalty={args.decode_beam_area_growth_penalty}, "
        f"ref_penalty={args.decode_beam_ref_token_penalty}"
    )
    print(f"Temperature sampling enabled: {not args.disable_temp_sampling}")

    # Check paths
    if not os.path.exists(args.subgraph_dir):
        print(f"\nError: Subgraph directory not found: {args.subgraph_dir}")
        return 1

    if not os.path.exists(args.ckpt_path + ".index"):
        print(f"\nError: Checkpoint not found: {args.ckpt_path}")
        return 1

    # Initialize model
    print("\n" + "=" * 70)
    print("Initializing SACT Model")
    print("=" * 70)
    print("This may take 1-2 minutes...")

    model = ScalableCircuitTransformer(
        max_inputs=args.max_inputs,
        max_outputs=args.max_outputs,
        embedding_width=512,
        num_layers=12,
        num_attention_heads=8,
        max_seq_length=800,
        ckpt_path=args.ckpt_path,
        error_rate_threshold=args.error_threshold,
        verbose=1,
        use_memory_dfs=args.use_memory_dfs,
        policy_temperature_in_mcts=args.inference_temperature,
        decode_top_k_in_inference=args.decode_top_k,
        sample_when_temp_in_inference=(not args.disable_temp_sampling),
        decode_low_conf_margin=args.decode_low_conf_margin,
        decode_beam_size_in_inference=args.decode_beam_size,
        decode_beam_until_step=args.decode_beam_until_step,
        decode_beam_lookahead_weight=args.decode_beam_lookahead_weight,
        decode_beam_lookahead_steps_in_inference=args.decode_beam_lookahead_steps,
        decode_beam_logprob_weight=args.decode_beam_logprob_weight,
        decode_beam_delay_delta_weight=args.decode_beam_delay_delta_weight,
        decode_beam_step_reward_weight=args.decode_beam_step_reward_weight,
        decode_beam_area_growth_penalty=args.decode_beam_area_growth_penalty,
        decode_beam_ref_token_penalty=args.decode_beam_ref_token_penalty,
    )

    print("✓ Model initialized and weights loaded")

    # Perform batch inference
    results = batch_inference(
        model,
        args.subgraph_dir,
        args.output_dir,
        args.max_subgraphs,
        args.overflow_option,
        args.batch_size
    )

    # Save results to file if requested
    if args.save_results and args.output_dir:
        results_file = Path(args.output_dir) / "inference_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
