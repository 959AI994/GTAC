# Graph Partition & Merge

## Usage

1. Environment

    1. Ubuntu 20.04 LTS (reference)
    2. Build toolchain
       1. cmake 3.16.3
       2. make 4.2.1
       3. gcc 10.3.0 and gcc 4.8.0
       4. g++ 10.3.0 and gcc 4.8.0
    3. Libraries
       1. boost 1.75
       2. libreadline
    4. ABC: clone `git@github.com:berkeley-abc/abc.git`, then replace everything under `abc/src` in that tree with `abc/src` from **this** repository.
    5. Espresso: `git@github.com:changmg/espresso.git`

2. Build

   `mkdir build`

   `cd build`

   `cmake ..`

   `make -j <num_cpus>`

   `cd ..`

   This produces `als.out` in the project root (`graph_partition_merge`).

3. Example commands

(1) Graph partition

`./als.out --accCirc demo_big_test/adder.aig --mode 1 --outpPath partition_out/adder > partition_out/adder/adder.log`

`./als.out --accCirc demo_big_test/bar.aig --mode 1 --outpPath partition_out/bar > partition_out/bar/bar.log`

`./als.out --accCirc demo_big_test/ac97_ctrl.aig --mode 1 --outpPath partition_out/ac97_ctrl > partition_out/ac97_ctrl/ac97_ctrl.log`

`./als.out --accCirc demo_big_test/aes_secworks.aig --mode 1 --outpPath partition_out/aes_secworks > partition_out/aes_secworks/aes_secworks.log`

`./als.out --accCirc ../EPFL/benchmarks/arithmetic/xxx.aig --mode 1 --outpPath partition_out/EPFL/arithmetic/xxx > partition_out/EPFL/arithmetic/xxx/xxx.log`

(2) Graph merge

Copy `acc_sop_arbiter_size_11839_depth_87.v` to `arbiter.v` and rename the Verilog `module` to `arbiter`.

`./als.out --accCirc graph_merge/EPFL/random_control/arbiter/arbiter.v --mode 2 --outpPath graph_merge/EPFL/random_control/arbiter/merge_out --metrType ER > graph_merge/EPFL/random_control/arbiter/merge_out/merge.log`

`time ./als.out --accCirc graph_merge/EPFL/random_control/arbiter/arbiter.v --mode 2 --outpPath graph_merge/EPFL/random_control/arbiter/merge_out_greedy_v1 --metrType ER --errUppBound 0.1 > graph_merge/EPFL/random_control/arbiter/merge_greedy_v1.log`

`time ./als.out --accCirc graph_merge/EPFL/random_control/arbiter/arbiter.v --mode 2 --outpPath graph_merge/EPFL/random_control/arbiter/merge_out_binary_v2 --metrType ER --errUppBound 0.1 > graph_merge/EPFL/random_control/arbiter/merge_binary_v2.log`

Command template:

`time ./als.out --accCirc graph_merge/EPFL/xxx/xxx.v --mode 2 --outpPath graph_merge/EPFL/xxx/merge_out_binary --metrType ER --errUppBound 0.1 > graph_merge/EPFL/xxx/merge_binary.log`

EPFL:

`time ./als.out --accCirc graph_merge/EPFL/square/square.v --mode 2 --outpPath graph_merge/EPFL/square/merge_out_binary --metrType ER --errUppBound 0.1 > graph_merge/EPFL/square/merge_binary.log`

`time ./als.out --accCirc graph_merge/EPFL/priority/priority.v --mode 2 --outpPath graph_merge/EPFL/priority/merge_out_binary --metrType ER --errUppBound 0.1 > graph_merge/EPFL/priority/merge_binary.log`


### Graph merge implementation notes

- Main challenge: to keep correct subgraph local inputs (LI) and local outputs (LO) addressable during approximate replacement, those nodes must not be removed in ways that break tracking (e.g. dangling nets), and original IDs must be preserved.
    - `topo` is disabled because dangling nodes would lose `oriId` tracking.
    - Adding fake primary outputs was dropped: after replacing one subgraph, error vs. the exact circuit’s POs becomes inconsistent, and ABC can assert (e.g. in `Abc_NtkCreatePo`).
    - Current approach: when replacing a subgraph, delete each LO node but not its MFFC (`Abc_NtkDeleteObj_rec` replaced by `Abc_NtkDeleteObj`). A finer deletion policy would be possible while keeping LIs, but a conservative policy reduces bugs. After all replacements, run `ExactSimpl` to simplify and remove dangling nodes.

