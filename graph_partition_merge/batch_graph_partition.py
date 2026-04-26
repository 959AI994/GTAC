#!/usr/bin/env python3
import os
import subprocess
import sys
from collections import deque

# bench_dir = "../EPFL/benchmarks/arithmetic"
# out_root = "partition_out/EPFL/arithmetic"
bench_dir = "../EPFL/benchmarks/random_control"
out_root = "partition_out/EPFL/random_control"
# bench_dir = "../core"
# out_root = "partition_out/core"
TAIL_LINES = 20  # On failure, print this many trailing log lines

# skip_files = {"adder.aig", "bar.aig", "hyp.aig", "log2.aig", "max.aig", "multiplier.aig", "sin.aig", "sqrt.aig", "square.aig"}  # optional skip set
skip_files = {}
# dft.aig is very large (optional to skip)

for filename in sorted(os.listdir(bench_dir)):
    if not filename.endswith(".aig"):
        continue   
    if filename in skip_files:  # skip listed benchmarks
        print(f"Skipping {filename}")
        continue

    name = os.path.splitext(filename)[0]
    input_path = os.path.join(bench_dir, filename)
    output_dir = os.path.join(out_root, name)
    log_path = os.path.join(output_dir, f"{name}.log")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "./als.out",
        "--accCirc", input_path,
        "--mode", "1",
        "--outpPath", output_dir
    ]

    print(f"Running {filename}...")

    # Keep last TAIL_LINES lines in a deque
    tail_buffer = deque(maxlen=TAIL_LINES)
    exception_occurred = False

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in process.stdout:
            log_file.write(line)
            tail_buffer.append(line)

        process.wait()

        if process.returncode != 0:
            exception_occurred = True
            print(f"\nError detected in {filename} (return code {process.returncode}):")
            for line in tail_buffer:
                sys.stdout.write(line)
            if process.returncode < 0:
                print(f"Process terminated by signal {-process.returncode}")

    if exception_occurred:
        print(f"{filename} finished with errors, see full log: {log_path}\n")
    else:
        print(f"{filename} finished successfully, log saved to {log_path}")

print("All benchmarks finished.")
