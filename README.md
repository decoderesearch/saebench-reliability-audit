# How Reliable Are Sparse Autoencoder Benchmarks?

Code accompanying the paper *How Reliable Are Sparse Autoencoder Benchmarks?*

The repo contains the code for the three audit lenses in the paper:

1. **Reseed noise on a real LLM SAE** (`saebench_audit.runners.reseed`). Runs each
   SAEBench evaluation `n` times on a canonical Gemma Scope SAE with a different
   random seed each time and writes per-seed result JSONs.
2. **Validity on synthetic SAEs** (`saebench_audit.synthetic`). Builds
   hierarchy-aware sparse-probing, TPP, and SCR tasks on top of the SynthSAEBench
   ground-truth dictionary and evaluates SAEs against ground-truth feature
   recovery.
3. **Discriminability across training trajectories** (`saebench_audit.training`,
   `saebench_audit.runners.snapshots`). Trains the cross-architecture and
   sampled-Matryoshka SAE panels described in Appendix C of the paper, snapshots
   them through training, and evaluates every snapshot under SAEBench.

## Layout

```
saebench_audit/
    saes/                 # custom Matryoshka SAE classes (with log-uniform prefix sampling)
    training/             # SAE training scripts (cross-architecture, sampled-Matryoshka)
    synthetic/            # SynthSAEBench task generation + per-task evaluations
    diagnostic/           # diagnostic SAE constructors (best-K, random-init, permuted-decoder, perfect oracle)
    runners/              # SAEBench eval wrappers + reseed and snapshot drivers
tests/                    # tests
```

## Setup

```bash
uv sync
```

The repo depends on `sae-lens`, `sae-bench`, and `sae-probes`. SAE training
needs a CUDA GPU; SAEBench evaluations also need a GPU for any benchmark that
runs the underlying language model.

## Running the experiments

The three audit lenses each have their own entry point under
`saebench_audit/runners/`. See the docstrings of those modules for the canonical
hyperparameters used in the paper.

```bash
# 1. reseed noise on a canonical SAE (Section 3 of the paper)
python -m saebench_audit.runners.reseed \
    --sae-release gemma-scope-2b-pt-res-canonical \
    --sae-id layer_12/width_65k/canonical \
    --seeds 42 123 456 789 2024 \
    --output-dir results/reseed

# 2. synthetic-SAE validity (Section 4)
python -m saebench_audit.synthetic.run_eval --variation v1 --seed 1234

# 2b. To exactly reproduce the paper's Section 4 task feature picks (Figure 1
#     and Table 2 were generated from these), pass --paper-fixture. Without
#     it, fresh task picks are sampled from the data_gen RNGs (the
#     multi-seed rewrite uses different RNG offsets than the version the
#     paper figures were generated with, so feature picks will differ).
python -m saebench_audit.synthetic.run_eval --seed 1234 --paper-fixture v1_seed_1234

# 3. snapshot evals (Section 5) — assumes SAEs already trained, e.g. via
#    saebench_audit.training.train_btk_snapshots / train_sampled_matryoshka.
python -m saebench_audit.runners.snapshots --snapshots-root path/to/snapshots
```

## Tests, formatting, types

```bash
uv run ruff format
uv run ruff check
uv run pyright
uv run pytest
```
