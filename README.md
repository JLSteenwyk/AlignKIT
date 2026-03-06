# RLALIGN — RL for Sequence Alignment Parameter Optimization

Reinforcement learning framework that learns to select alignment tools
(MAFFT, MUSCLE, Clustal Omega) and their parameters on a per-input basis,
optimizing SP and TC scores against reference alignments.

## Architecture

**Contextual bandit** — each episode the agent observes a 20-dimensional
feature vector describing the input sequences (size, length heterogeneity,
pairwise identity, composition, complexity) and selects a single discrete
action specifying the tool and its full parameter configuration.  The reward
is the weighted average of SP and TC scores against a structural reference.

### Action space (380 actions)

| Tool       | Parameters                                    | Count |
|------------|-----------------------------------------------|------:|
| MAFFT      | op (5) × ep (4) × strategy (4) × maxiter (4) |   320 |
| MUSCLE v5  | command (1) × perm (4) × perturb (5)          |    20 |
| Clustal Ω  | iter (5) × full (2) × full-iter (2) × kimura (2) | 40 |

### State features (20-dim)

- Size & scale: num_sequences, avg_length, total_residues
- Length heterogeneity: CV, min/max ratio, range, median ratio
- Pairwise similarity: mean/std/min/max k-mer Jaccard, twilight fraction
- Composition: hydrophobic, charged, polar, small, aromatic, Pro+Gly fractions
- Complexity: normalized AA entropy, low-complexity fraction

## Datasets

Training and evaluation use structural reference alignments from multiple
benchmarks.  A stratified 80/20 split within each source ensures every
dataset is represented in both train and eval.

| Source   | Total MSAs | Description                                  |
|----------|-----------|----------------------------------------------|
| BAliBASE RV11 | 38  | Equidistant families, <20% identity          |
| BAliBASE RV12 | 44  | Equidistant families, 20-40% identity        |
| BAliBASE RV20 | 41  | Families with orphan sequences               |
| BAliBASE RV30 | 30  | Divergent subfamily pairs                    |
| BAliBASE RV40 | 49  | Sequences with large N/C extensions          |
| BAliBASE RV50 | 16  | Internal insertions                          |
| HOMSTRAD | 233       | Structural alignments from homologous families|
| OXBench  | 395       | Wide identity range (Edgar BENCH 1.0)        |
| SABRE    | 423       | Remote homologs, all twilight-zone           |

### Database balancing via farthest-point subsampling

The external benchmarks (HOMSTRAD, OXBench, SABRE) are much larger than the
BAliBASE subsets and would dominate training.  To equalize representation
while preserving diversity, each external benchmark's **training portion** is
subsampled to match the total BAliBASE training count (~172) using
**farthest-point sampling** in the 20-dim state feature space:

1. Z-score normalize all features within the database
2. Seed with the medoid (point closest to centroid)
3. Iteratively select the point with maximum min-distance to selected set
4. Repeat until target count is reached

This guarantees outliers are selected first and the chosen subset fills the
feature space evenly.  The eval split is left untouched for comprehensive
evaluation.

**Balanced training composition** (with `--subsample 172`):

| Source   | Train | Eval | Train % |
|----------|------:|-----:|--------:|
| BAliBASE | 172   | 45   | 25.0%   |
| HOMSTRAD | 172   | 47   | 25.0%   |
| OXBench  | 172   | 79   | 25.0%   |
| SABRE    | 172   | 85   | 25.0%   |
| **Total**| **688**|**257**|         |

## Usage

```bash
# Setup
source venv/bin/activate

# MSA training with balanced databases (recommended)
python main.py msa-train --episodes 100000 --batch-size 8 --subsample 172

# MSA training without subsampling (original behavior)
python main.py msa-train --episodes 100000 --batch-size 8

# Resume from checkpoint
python main.py msa-train --episodes 100000 --resume checkpoint_latest.pt

# Evaluate
python main.py msa-evaluate --checkpoint checkpoint_latest.pt

# Run baselines (MAFFT/MUSCLE/ClustalO defaults)
python main.py msa-baseline

# Visualize training curves
python main.py msa-visualize
```

## Project structure

```
RLALIGN/
├── main.py                  # CLI entry point
├── config_msa.py            # MSA configuration dataclass
├── agent/
│   ├── policy_network.py    # Feed-forward policy (20 -> 128 -> 128 -> 380)
│   └── reinforce.py         # REINFORCE with baseline
├── data/
│   ├── msa_dataset.py       # MSATestCase, MSADataset, build_msa_datasets
│   ├── benchmark_loader.py  # OXBench, SABRE, HOMSTRAD loaders
│   ├── subsample.py         # Farthest-point diversity subsampling
│   └── balibase_parser.py   # BAliBASE MSF parser
├── env/
│   ├── msa_alignment_env.py # Bandit environment (reset/step)
│   ├── msa_action_space.py  # 380-action discrete space
│   └── msa_state_features.py# 20-dim feature extraction
├── scoring/
│   ├── msa_sp_score.py      # Sum-of-pairs score
│   ├── msa_tc_score.py      # Total-column score
│   └── reward.py            # Weighted SP+TC reward
├── training/
│   ├── train_msa.py         # Main training loop
│   ├── checkpointer.py      # Save/load checkpoints
│   └── logger.py            # JSONL metric logger
├── evaluation/
│   ├── evaluate_msa.py      # Full greedy evaluation
│   └── msa_baselines.py     # Default-parameter baselines
└── visualization/
    └── msa_plots.py         # Training curve plots
```

## Key configuration (`config_msa.py`)

| Parameter                  | Default | Description                        |
|----------------------------|---------|------------------------------------|
| `lr`                       | 1e-3    | Learning rate                      |
| `entropy_coeff`            | 0.01    | Entropy regularization weight      |
| `baseline_momentum`        | 0.99    | Exponential moving average for baseline |
| `batch_size`               | 8       | Episodes per policy update         |
| `subprocess_timeout`       | 600     | Max seconds per alignment tool call|
| `sp_weight` / `tc_weight`  | 0.5/0.5 | Reward = sp_weight×SP + tc_weight×TC |
| `subsample_benchmarks_to`  | None    | Subsample external benchmarks (recommended: 172) |
