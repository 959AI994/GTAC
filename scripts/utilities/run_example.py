import argparse
import os
import random

import numpy as np

import gtac as ct
from gtac.model import CircuitTransformerRefactored

def main():
    parser = argparse.ArgumentParser(description="(Approximate) Circuit Transformer: training + inference example")
    parser.add_argument("--train", action="store_true", help="Run training before inference")
    parser.add_argument("--train_data_dir", type=str, default="./datasets/t/IWLS_FFWs", help="Training dataset directory")
    parser.add_argument("--ckpt_save_path", type=str, default="./ckpt", help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--no_profile", action="store_true", help="Disable TensorBoard profiling/logging during training")
    parser.add_argument("--distributed", action="store_true", help="Enable MirroredStrategy training")
    parser.add_argument("--latest_ckpt_only", action="store_true", help="Only keep latest checkpoint (training)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (shuffle, numpy, TF)")

    parser.add_argument("--load_hf", action="store_true", help="Load pretrained weights from the (anonymized) model hub")
    parser.add_argument("--run_inference", action="store_true", help="Run inference on two embedded AIGER examples")
    parser.add_argument("--run_mcts", action="store_true", help="Run MCTS-enhanced inference on the same examples")
    parser.add_argument("--num_mcts_steps", type=int, default=1, help="Number of MCTS steps")
    parser.add_argument("--num_mcts_playouts_per_step", type=int, default=10, help="Number of MCTS playouts per step")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    circuit_transformer = CircuitTransformerRefactored()

    if args.train:
        if not os.path.exists(args.train_data_dir):
            raise FileNotFoundError(f"Training data directory {args.train_data_dir} does not exist")
        os.makedirs(args.ckpt_save_path, exist_ok=True)

        circuit_transformer.train(
            train_data_dir=args.train_data_dir,
            ckpt_save_path=args.ckpt_save_path,
            validation_split=args.validation_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            profile=(not args.no_profile),
            distributed=args.distributed,
            latest_ckpt_only=args.latest_ckpt_only,
            shuffle_seed=args.seed,
        )

    # If requested, load pretrained weights (this can be used as a pure inference run).
    if args.load_hf:
        circuit_transformer.load_from_hf()

    # Default behavior: if nothing selected, run inference with pretrained weights.
    if not args.train and not args.load_hf and not args.run_inference and not args.run_mcts:
        circuit_transformer.load_from_hf()
        args.run_inference = True

    aig0, info0 = ct.read_aiger(aiger_str="""aag 33 8 0 2 25
2\n4\n6\n8\n10\n12\n14\n16\n58\n67
18 13 16\n20 19 7\n22 21 15\n24 3 9\n26 25 11
28 27 17\n30 3 6\n32 29 31\n34 29 32\n36 23 35
38 7 36\n40 10 29\n42 41 32\n44 13 15\n46 42 45
48 47 21\n50 39 49\n52 4 45\n54 25 53\n56 54 5
58 51 57\n60 45 12\n62 18 61\n64 63 19\n66 48 64
""")
    aig1, info1 = ct.read_aiger(aiger_str="""aag 22 8 0 2 14
2\n4\n6\n8\n10\n12\n14\n16\n24\n44
18 10 12\n20 8 7\n22 21 5\n24 19 23\n26 11 3
28 6 4\n30 26 28\n32 8 5\n34 32 26\n36 35 17
38 37 7\n40 31 39\n42 41 12\n44 43 15
""")
    aigs = [aig0, aig1]

    if args.run_inference:
        optimized_aigs = circuit_transformer.optimize(aigs)
        print("Circuit Transformer:")
        for i, (aig, optimized_aig) in enumerate(zip(aigs, optimized_aigs)):
            print(
                "aig %d #(AND) from %d to %d, equivalence check: %r"
                % (i, ct.count_num_ands(aig), ct.count_num_ands(optimized_aig), ct.cec(aig, optimized_aig))
            )

    if args.run_mcts:
        optimized_aigs_with_mcts = circuit_transformer.optimize(
            aigs=aigs,
            num_mcts_steps=args.num_mcts_steps,
            num_mcts_playouts_per_step=args.num_mcts_playouts_per_step,
        )
        print("Circuit Transformer + Monte-Carlo Tree Search:")
        for i, (aig, optimized_aig) in enumerate(zip(aigs, optimized_aigs_with_mcts)):
            print(
                "aig %d #(AND) from %d to %d, equivalence check: %r"
                % (i, ct.count_num_ands(aig), ct.count_num_ands(optimized_aig), ct.cec(aig, optimized_aig))
            )


if __name__ == "__main__":
    main()  
