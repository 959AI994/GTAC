#!/usr/bin/env python3
"""
Training script for subgraphs using Scalable Circuit Transformer

This script trains the SACT model on circuit optimization datasets,
supporting variable-sized circuits (up to 256 inputs/outputs).

For inference, use inference_subgraphs.py.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.scalable_model import ScalableCircuitTransformer


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train SACT on subgraph datasets')
    parser.add_argument('--train-data-dir', type=str, required=True,
                       help='Directory containing training data (JSON files with circuit pairs)')
    parser.add_argument('--ckpt-path', type=str, default=None,
                       help='Path to save checkpoints')
    parser.add_argument('--load-ckpt', type=str, default=None,
                       help='Path to load checkpoint weights (separate from save path)')
    parser.add_argument('--log-dir', type=str, default='log_sact',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--max-inputs', type=int, default=256,
                       help='Maximum number of inputs to support')
    parser.add_argument('--max-outputs', type=int, default=256,
                       help='Maximum number of outputs to support')
    parser.add_argument('--use-memory-dfs', action='store_true',
                       help='Use memory DFS encoding (default: False, uses memoryless DFS)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate for optimizer')
    parser.add_argument('--validation-split', type=float, default=0.1,
                       help='Fraction of data for validation')
    parser.add_argument('--initial-epoch', type=int, default=0,
                       help='Initial epoch (for resuming training)')
    parser.add_argument('--debug-inf', action='store_true',
                       help='Enable INF loss debug: print sample/position/target when loss=inf')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Scalable Circuit Transformer - Training on Subgraphs")
    print("=" * 70)
    print(f"Training data: {args.train_data_dir}")
    print(f"Checkpoint save path: {args.ckpt_path}")
    print(f"Checkpoint load path: {args.load_ckpt}")
    print(f"Max inputs/outputs: {args.max_inputs}/{args.max_outputs}")
    print(f"Use memory DFS encoding: {args.use_memory_dfs}")

    # Check paths
    if not os.path.exists(args.train_data_dir):
        print(f"\nError: Training data directory not found: {args.train_data_dir}")
        return 1

    # Count training files
    train_files = [f for f in os.listdir(args.train_data_dir) if f.endswith('.json')]
    print(f"\nFound {len(train_files)} training files")

    if len(train_files) == 0:
        print("Error: No JSON training files found in the directory!")
        print("Expected format: JSON files containing [original_aig, num_ands, optimized_aig, opt_num_ands]")
        return 1

    # Initialize model
    print("\n" + "=" * 70)
    print("Initializing SACT Model")
    print("=" * 70)
    print("This may take 1-2 minutes...")

    # Initialize model without loading checkpoint
    model = ScalableCircuitTransformer(
        max_inputs=args.max_inputs,
        max_outputs=args.max_outputs,
        embedding_width=512,
        num_layers=12,
        num_attention_heads=8,
        max_seq_length=400,  # Large enough for most big circuits
        ckpt_path=None,
        verbose=0,
        error_rate_threshold=0.1,
        tt_num_samples=4096,
        tt_seed=42,
        use_memory_dfs=args.use_memory_dfs
    )

    print("✓ Model initialized")
    
    # Load checkpoint if specified (after initialization)
    if args.load_ckpt:
        if os.path.exists(args.load_ckpt + ".index"):
            print(f"Loading weights from: {args.load_ckpt}")
            model.load(args.load_ckpt)
            print(f"✓ Loaded checkpoint from: {args.load_ckpt}")
        else:
            print(f"Warning: Checkpoint not found at {args.load_ckpt}, starting from scratch")

    # Training configuration
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)

    # Set default checkpoint path if not provided
    if args.ckpt_path is None:
        ckpt_path = "/home/gst/repo/Scalable-Approximate-Circuit-Transformer/ckpt_sact"
    else:
        ckpt_path = args.ckpt_path

    # Create directories if needed
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"  Data directory: {args.train_data_dir}")
    print(f"  Number of training files: {len(train_files)}")
    print(f"  Checkpoint save path: {ckpt_path}")
    print(f"  Log directory: {args.log_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Initial epoch: {args.initial_epoch}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Validation split: {args.validation_split}")
    print(f"  Use memory DFS encoding: {args.use_memory_dfs}")
    if args.debug_inf:
        print(f"  Debug INF: ENABLED (will print sample/position when loss=inf)")

    # Start training
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)

    try:
        model.train(
            train_data_dir=args.train_data_dir,
            ckpt_save_path=ckpt_path,
            validation_split=args.validation_split,
            epochs=args.epochs,
            initial_epoch=args.initial_epoch,
            batch_size=args.batch_size,
            profile=True,
            log_dir=args.log_dir,
            learning_rate=args.learning_rate,
            debug_inf=args.debug_inf
        )

        print("\n" + "=" * 70)
        print("✓ Training completed successfully!")
        print("=" * 70)
        print(f"\nCheckpoints saved to: {ckpt_path}")
        print(f"TensorBoard logs saved to: {args.log_dir}")
        print(f"\nTo view training progress:")
        print(f"  tensorboard --logdir={args.log_dir}")
        print(f"\nTo perform inference on subgraphs:")
        print(f"  python inference_subgraphs.py --subgraph-dir <dir> --ckpt-path {ckpt_path}/model-<epoch>")

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ Training failed!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
