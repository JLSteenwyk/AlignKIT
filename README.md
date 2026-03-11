# RLALIGN — RL for Sequence Alignment Parameter Optimization

An exploration of reinforcement learning for adaptive gap penalty selection in
protein sequence alignment. The project investigated whether an RL agent can
learn to predict alignment-specific parameters that outperform fixed defaults.

**Status: Concluded.** The approach produces a small but consistent improvement
over MAFFT defaults (+2% reward), but the gain is too modest to justify
further development. This README documents what was built, what was learned,
and why the project was sunset.


## Motivation

Multiple sequence alignment tools like MAFFT use fixed gap penalties (opening
penalty op=1.53, extension penalty ep=0.0) regardless of the input sequences.
An exhaustive oracle analysis on BAliBASE 3.0 showed that per-case parameter
tuning could improve alignment quality — but by how much?

### Oracle Analysis (185 actions x 42 cases)

| Component | Reward | vs Default |
|-----------|:------:|:----------:|
| MAFFT --auto default (op=1.53, ep=0.0) | 0.441 | -- |
| Best fixed config (op=4.0, ep=0.1, --localpair) | 0.506 | +14.7% |
| MAFFT per-case oracle | 0.532 | +20.6% |
| Full oracle (all 185 tool configs) | 0.536 | +21.5% |

Key findings:
- **MAFFT dominates**: oracle-best tool on 34/42 cases (81%)
- **Parameter tuning is the main opportunity**: 85% of oracle headroom comes
  from per-case MAFFT parameter adaptation, not cross-tool selection
- **No universal optimum**: 35 unique best actions across 42 cases

These findings motivated focusing on MAFFT parameter prediction rather than
tool selection.


## What Was Built

### Phase 1: Pairwise Alignment (Needleman-Wunsch)

A custom NW implementation with position-dependent affine gap penalties, where
an RL agent predicts gap_open and gap_extend for K=3 sequence regions
(N-terminal, middle, C-terminal).

- **Agent**: Gaussian continuous policy (12-dim features -> 6-dim actions)
- **Training**: 50K episodes on BAliBASE pairwise pairs
- **Result**: Agent learned gap_open ~ 17 (near optimal 18) but failed to
  find optimal gap_extend (learned ~1.7 vs optimal 0.5). Captured only 8.3%
  of oracle headroom.
- **Conclusion**: 6-dimensional exploration made it hard to disentangle
  parameter effects with limited data.

### Phase 2: MSA with Continuous MAFFT Parameters

Scaled up to multiple sequence alignment by wrapping MAFFT. The agent predicts
two continuous parameters per alignment case:
- **op** (gap opening penalty): range [0.5, 5.0]
- **ep** (gap extension penalty): range [0.0, 1.0]

MAFFT runs with `--localpair --maxiterate 10` (L-INS-i algorithm) using the
agent's predicted parameters.

**Architecture:**
- Input: 28-dim feature vector (sequence lengths, composition, complexity,
  pairwise similarity, Neff, length skewness, gap propensity)
- Policy: Gaussian MLP (28 -> 128 -> 128 -> 2) with sigmoid transform
- Training: REINFORCE with learned value baseline, advantage normalization
- Formulation: Contextual bandit (single decision per alignment case)

**Data:**

| Split | Source | Cases |
|-------|--------|------:|
| Train | BAliBASE RV11/RV12/RV20/RV30 | 153 |
| Train | OXBench + SABRE + HOMSTRAD | 840 |
| Eval  | BAliBASE RV40/RV50 | 65 |
| Eval  | OXBench + SABRE + HOMSTRAD (20%) | 211 |

### Results

**BAliBASE evaluation (RV40 + RV50, n=65):**

| Method | Reward | SP | TC |
|--------|:------:|:--:|:--:|
| MAFFT --auto (default) | 0.446 | 0.698 | 0.193 |
| MAFFT L-INS-i (default params) | 0.448 | 0.703 | 0.194 |
| RL Agent v1 (20-dim, BAliBASE only) | **0.454** | 0.705 | **0.203** |
| RL Agent v2 (28-dim, + benchmarks) | 0.450 | **0.708** | 0.193 |

Best result: **+2.0% reward, +5.4% TC** over MAFFT --auto (v1 agent).

**What the agent learned:**
- op ~ 3.0 (nearly 2x the MAFFT default of 1.53)
- ep ~ 0.4 (vs default of 0.0)
- Both parameters stabilized by ~2000 episodes

**What the agent did NOT learn:**
- Meaningful per-case adaptation. The agent converged to a near-constant
  prediction across all inputs rather than using features to tailor parameters.
- Adding more training data (benchmarks) and richer features (28-dim) did not
  improve BAliBASE results — the extra data diluted the signal without
  enabling better adaptation.


## Why the Project Was Sunset

1. **The improvement is too small.** A 2% gain over defaults, while
   consistent, is not enough to justify the complexity of an RL system.
   A simple grid search over op/ep would likely match or exceed the agent.

2. **The agent learns a better global default, not per-case adaptation.**
   The core thesis — that sequence features can drive case-specific gap
   penalties — was not validated. The policy outputs are nearly constant
   regardless of input.

3. **Limited headroom.** The oracle analysis showed that even perfect
   parameter selection yields only +20% over defaults. With just op and ep,
   the ceiling is lower. The agent captured a small fraction of this.

4. **Data limitations.** 153 BAliBASE training cases is thin for learning
   a feature-to-parameter mapping. Adding external benchmarks helped with
   diversity but introduced distribution mismatch.

5. **The baseline is strong.** MAFFT's defaults are well-tuned. Beating
   a mature tool's 20+ years of parameter engineering with RL requires
   either more expressive actions (position-specific penalties, algorithm
   selection) or a fundamentally different approach.


## What Would Be Needed for Publication

If someone wanted to continue this work:

- **Statistical significance testing** on per-case improvements
- **Ablation studies**: which features matter, does the agent truly adapt
- **Stronger baselines**: grid search, linear regression, parameter advisors
  (DeBlasio & Kececioglu, 2017)
- **Broader action space**: add strategy selection, maxiterate, or
  position-specific penalties
- **Cross-validation** instead of fixed train/eval splits
- **Analysis of learned behavior**: when/why does the agent deviate from
  its mean prediction


## Related Work

- **Parameter Advisors** (DeBlasio & Kececioglu, 2015-2017): Closest prior
  work. Facet accuracy estimator selects from discrete parameter sets for
  the Opal aligner. Different approach (accuracy estimation vs policy gradient)
  but same goal.
- **RL-based aligners** (DPAMSA 2023, RLALIGN 2018): Use RL to perform
  alignment directly (gap insertion decisions). Our approach wraps an
  existing aligner instead.
- **Standard MSA tools** (MAFFT, MUSCLE, Clustal Omega): None adapt gap
  penalties based on input sequence characteristics.


## Usage

```bash
source venv/bin/activate

# Train (BAliBASE + benchmarks, 28-dim features)
CUDA_VISIBLE_DEVICES="" python main.py msa-train --episodes 10000

# Train (BAliBASE only)
CUDA_VISIBLE_DEVICES="" python main.py msa-train --episodes 10000 --no-benchmarks

# Resume training
CUDA_VISIBLE_DEVICES="" python main.py msa-train --resume checkpoint_latest.pt

# Evaluate
CUDA_VISIBLE_DEVICES="" python main.py msa-evaluate

# Pairwise NW training (Phase 1)
CUDA_VISIBLE_DEVICES="" python main.py pw-train --episodes 50000

# Regenerate figures
python results/msa_continuous/generate_figures.py
python results/msa_continuous/workflow_diagram.py
```

Note: `CUDA_VISIBLE_DEVICES=""` is required — PyTorch version is too old for
the RTX 6000 Ada GPUs (sm_89).


## Project Structure

```
RLALIGN/
├── main.py                       # CLI: pw-train, pw-evaluate, msa-train, msa-evaluate
├── config.py                     # Config and MSAConfig dataclasses
├── agent/
│   ├── continuous_policy.py      # Gaussian policy with sigmoid transform
│   └── reinforce_continuous.py   # REINFORCE with value baseline
├── data/
│   ├── balibase_parser.py        # BAliBASE MSF parser
│   ├── msa_dataset.py            # MSATestCase, MSADataset, loaders
│   ├── benchmark_loader.py       # OXBench, SABRE, HOMSTRAD loaders
│   └── pairwise_dataset.py       # Pairwise pair extraction
├── env/
│   ├── msa_env.py                # MSA bandit environment (MAFFT wrapper)
│   ├── msa_features.py           # 28-dim MSA feature extraction
│   ├── pairwise_env.py           # Pairwise NW environment
│   ├── pairwise_features.py      # 12-dim pairwise features
│   └── needleman_wunsch.py       # Vectorized NW with affine gaps
├── scoring/
│   ├── msa_sp_score.py           # MSA sum-of-pairs score
│   ├── msa_tc_score.py           # MSA total-column score
│   ├── sp_score.py               # Pairwise SP score
│   ├── tc_score.py               # Pairwise TC score
│   └── reward.py                 # Weighted SP+TC reward
├── training/
│   ├── train_msa.py              # MSA training loop
│   ├── train_pairwise.py         # Pairwise training loop
│   └── checkpointer.py           # Checkpoint save/load
├── evaluation/
│   ├── evaluate_msa.py           # MSA eval + baselines
│   └── evaluate_pairwise.py      # Pairwise eval + baselines
├── results/
│   ├── oracle_analysis/          # Exhaustive oracle analysis (185 actions x 42 cases)
│   ├── msa_continuous/           # MSA RL results, figures, workflow diagram
│   └── supplementary/            # Oracle analysis figures
├── logs_msa/                     # Training and eval logs (JSONL)
├── checkpoints_msa/              # Model checkpoints
└── archive/                      # Old pilot system code (preserved for reference)
```


## Key Numbers

| Metric | Value |
|--------|-------|
| BAliBASE training cases | 153 |
| External benchmark training cases | 840 |
| BAliBASE eval cases (RV40+RV50) | 65 |
| MAFFT --auto baseline reward | 0.446 |
| Best RL agent reward | 0.454 |
| Improvement over --auto | +2.0% |
| Oracle ceiling (MAFFT params) | 0.532 |
| Headroom captured by agent | ~9% of oracle headroom |
| Training time (10K episodes) | ~10-30 hours (CPU) |
