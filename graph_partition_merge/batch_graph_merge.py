#!/usr/bin/env python3
"""
Run graph merge for each subdirectory under graph_merge/EPFL.
"""
import os
import subprocess
import sys

# Project root (directory containing this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPFL_BASE = os.path.join(SCRIPT_DIR, "graph_merge", "EPFL")

# Subdirectories to skip
# SKIP_FOLDERS = {"arbiter"}
SKIP_FOLDERS = {}


def main():
    os.chdir(SCRIPT_DIR)  # Run with cwd = project root

    if not os.path.isdir(EPFL_BASE):
        print(f"Error: {EPFL_BASE} does not exist")
        sys.exit(1)

    subdirs = sorted(
        d for d in os.listdir(EPFL_BASE)
        if os.path.isdir(os.path.join(EPFL_BASE, d))
    )

    for name in subdirs:
        if name in SKIP_FOLDERS:
            print(f"Skipping {name}")
            continue

        circ_path = os.path.join(EPFL_BASE, name, f"{name}.v")
        if not os.path.isfile(circ_path):
            print(f"Skipping {name}: {name}.v not found")
            continue

        # Paths relative to project root
        rel_circ = os.path.join("graph_merge", "EPFL", name, f"{name}.v")
        rel_outp = os.path.join("graph_merge", "EPFL", name, "merge_out_binary")
        rel_log = os.path.join("graph_merge", "EPFL", name, "merge_binary.log")

        cmd = (
            f"time ./als.out --accCirc {rel_circ} --mode 2 "
            f"--outpPath {rel_outp} --metrType ER --errUppBound 0.1 "
            f"> {rel_log}"
        )

        print(f"Running {name}...")

        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=SCRIPT_DIR,
        )
        proc.wait()

        if proc.returncode != 0:
            print(f"  {name} finished with errors (return code {proc.returncode}), log: {rel_log}")
        else:
            print(f"  {name} finished successfully, log: {rel_log}")

    print("All benchmarks finished.")


if __name__ == "__main__":
    main()
