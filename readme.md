# GTAC: A Generative Transformer for Approximate Circuits

Targeting error-tolerant applications, approximate computing relaxes rigid functional equivalence to significantly improve power,
performance, and area. Traditional approximate logic synthesis (ALS) relies on incremental rewriting, limiting design space exploration. Meanwhile, the inherently probabilistic nature of Transformer-based generative AI makes it a natural fit for generating approximate circuits. Exploiting this, we propose GTAC, an end-to-end framework for arbitrary-scale generative ALS. To overcome the memory bottleneck of generative AI, GTAC partitions a large circuit into tractable subcircuits, applies a generative core to produce approximate candidates for each subcircuit, and finally selects proper candidates to form the final design. Its core generative Transformer utilizes a novel irredundant encoding to compactly encode a circuit, alongside a masking mechanism to exclude designs violating the given error bound. Empowered by a self-evolutionary training strategy, GTAC establishes a new paradigm that demonstrates superior performance: It reduces delay by 30.9% and gate count by 50.5% over exact generative baselines and saves 6.5% area with a 4.3× speedup against traditional ALS methods, alongside a 60× memory reduction for scalability.

> **Note**: This is an **anonymized** repository for double-blind review. Identifying information and links will be restored upon acceptance.

## Install

```bash
pip install -r requirements.txt
```

## Datasets

All circuits are stored in **AIGER** formats (`.aig` / `.aag`) and are used for training/evaluation of circuit optimization and approximation.

### Training Data (error rate = 0.10)

Randomly generated circuits (e.g., 8-input / 2-output random benchmarks). The suffix `0.10` indicates the **allowed output error rate**.

```bash
wget https://huggingface.co/datasets/[ANONYMOUS]/circuit-transformer/resolve/main/8_inputs_2_outputs_random_deepsyn.zip
wget https://huggingface.co/datasets/[ANONYMOUS]/Approximate-Circuit-transformer/resolve/main/random_circuit_0.1_200k.zip
```

We also used the corresponding subgraph datasets of EPFL and Opencores to further train the model, in order to improve GTAC's inference capabilities for different inputs and outputs and large circuits.

```bash
wget https://huggingface.co/datasets/[ANONYMOUS]/Approximate-Circuit-transformer/resolve/main/subckt_0.1.zip
```

### Test Data

Small-scale benchmark set for quick functional testing.

```bash
wget https://huggingface.co/datasets/[ANONYMOUS]/circuit-transformer/resolve/main/test_data.zip
```
If you want to infer subgraphs, please refer to the subgraph data in the `graph_partition_merge/graph_partition_data` directory.

### Approximate Circuit Benchmarks (ALSRAC, error rate = {1%, 5%, 10%})

Approximate circuit benchmarks derived from IWLS, with target error rates  \delta \in 0.01, 0.05, 0.10 .

```bash
wget https://huggingface.co/datasets/[ANONYMOUS]/Approximate-Circuit-Transformer/resolve/main/IWLS_FFWs_app_0.01.zip
wget https://huggingface.co/datasets/[ANONYMOUS]/Approximate-Circuit-Transformer/resolve/main/IWLS_FFWs_app_0.05.zip
wget https://huggingface.co/datasets/[ANONYMOUS]/Approximate-Circuit-Transformer/resolve/main/IWLS_FFWs_app_0.1.zip
```

## Training

```bash
python -m src.train_subgraphs \
  --train-data-dir <train_json_dir> \
  --ckpt-path <checkpoint_output_dir> \
  --epochs 10 \
  --batch-size 8
```

Training directory should contain `.json` samples in:

- `[original_aig, num_ands, optimized_aig, opt_num_ands]`

## Inference

```bash
python -m src.inference_subgraphs \
  --subgraph-dir <aig_or_aag_dir> \
  --ckpt-path <checkpoint_prefix> \
  --output-dir <output_dir> \
  --batch-size 1 \
  --save-results
```

## Graph partition & merge

The `graph_partition_merge/` directory provides a C++ toolchain (build with CMake to produce `als.out`) for **partitioning** large circuits into subgraphs and **merging** them after approximate optimization, plus small Python helpers (`batch_graph_partition.py`, `batch_graph_merge.py`, etc.). It ships with EPFL-oriented benchmark material and related archives under that tree. See `graph_partition_merge/README.md` for build steps and example commands.

## Reproducibility Notes

- The training data shuffle uses a fixed seed by default (configurable via the example script).
- Reported results may vary across GPU / CUDA / driver versions; please include your hardware details when reporting numbers.

## License & Anonymity

- **License**: see `LICENSE`.
- **Anonymity Note**: All links and resources are anonymized for double-blind review. Any identifying information will be released upon acceptance.