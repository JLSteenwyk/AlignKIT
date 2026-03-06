# Oracle Analysis: Tool Selection vs Parameter Tuning

## Motivation

Before investing in RL training for multi-tool MSA selection, we conducted an
exhaustive oracle analysis to quantify the performance ceiling and decompose
where gains come from.

## Setup

- **Dataset**: BAliBASE benchmark (6 reference sets: RV11–RV50)
- **Eval cases**: 42 (stratified subsample, up to 8 per reference set)
- **Action space**: 185 total actions
  - MAFFT: 144 configs (4 op × 3 ep × 4 strategy × 3 maxiterate)
  - MUSCLE: 9 configs (1 command × 3 perm × 3 perturb)
  - Clustal Omega: 32 configs (4 iter × 2 full × 2 full_iter × 2 kimura)
- **Reward**: 0.5 × SP + 0.5 × TC (scored against BAliBASE reference alignments)
- **Method**: Exhaustive — all 185 actions evaluated on all 42 cases (7,770 alignments)

## Key Results

### Per-Tool Performance

| Metric | MAFFT | MUSCLE | Clustal Omega |
|--------|-------|--------|---------------|
| Best fixed config (avg over all cases) | 0.506 | 0.499 | 0.486 |
| Per-case oracle (best config per case) | 0.532 | 0.499 | 0.486 |
| Cases where tool is oracle-best | 34 (81%) | 3 (7%) | 5 (12%) |

### Headroom Decomposition

| Component | Reward | Headroom |
|-----------|--------|----------|
| Overall best fixed action (MAFFT #125) | 0.506 | — |
| MAFFT per-case oracle | 0.532 | +0.026 |
| Full oracle (any tool, any config) | 0.536 | +0.030 |

**MAFFT parameter tuning accounts for 86% of total oracle headroom.**
Cross-tool switching adds only 14% additional gain (+0.004 reward).

### Per-Dataset Breakdown

| Dataset | N | MAFFT | MUSCLE | ClstlO | Oracle | BstFix | Headroom | Oracle Tool |
|---------|---|-------|--------|--------|--------|--------|----------|-------------|
| RV11 | 8 | 0.4105 | 0.3617 | 0.2973 | 0.4160 | 0.3668 | +0.049 | M=7 U=1 C=0 |
| RV12 | 8 | 0.7117 | 0.6629 | 0.6447 | 0.7124 | 0.6904 | +0.022 | M=7 U=0 C=1 |
| RV20 | 8 | 0.5371 | 0.5040 | 0.5208 | 0.5384 | 0.5241 | +0.014 | M=7 U=0 C=1 |
| RV30 | 6 | 0.5493 | 0.5390 | 0.5359 | 0.5689 | 0.5317 | +0.037 | M=3 U=2 C=1 |
| RV40 | 8 | 0.4956 | 0.4751 | 0.4784 | 0.4968 | 0.4864 | +0.010 | M=6 U=0 C=2 |
| RV50 | 4 | 0.4482 | 0.4263 | 0.4135 | 0.4482 | 0.4398 | +0.008 | M=4 U=0 C=0 |
| **ALL** | **42** | **0.5316** | **0.4993** | **0.4857** | **0.5361** | **0.5058** | **+0.030** | |

Notes:
- MAFFT/MUSCLE/ClstlO columns show the per-case oracle *within* that tool
- Oracle = best action per case across all tools
- BstFix = single best action for all cases in that dataset
- Headroom = Oracle − BstFix

### Multi-Tool RL Training Attempts

We ran two multi-tool RL training experiments with a hierarchical policy
(tool head + per-tool parameter heads) using REINFORCE with a learned
value baseline:

1. **All benchmarks (688 train / 257 eval)**: 4,400 episodes. Policy
   collapsed to 100% MUSCLE in greedy eval (R=0.567). Tool entropy dropped
   from 1.098 → 0.74 but greedy argmax locked onto MUSCLE at both eval
   checkpoints (ep 2000, 4000).

2. **BAliBASE-only (174 train / 44 eval)**: Killed at ep 400 (no eval
   completed). Earlier training showed same MUSCLE-collapse pattern.

In both cases, the policy found MUSCLE as a safe single-tool default rather
than learning state-dependent tool switching — consistent with the oracle
finding that cross-tool gains are minimal (+0.004).

## Conclusion

**The learnable reward comes from MAFFT parameter tuning, not tool selection.**

- MAFFT is oracle-best on 81% of BAliBASE eval cases
- MAFFT param tuning captures 86% of total headroom
- Cross-tool switching adds only 0.004 reward (14% of headroom)
- Multi-tool RL collapsed to single-tool policies in practice

**Recommendation**: Simplify to a MAFFT-only policy that learns to select
op, ep, strategy, and maxiterate parameters conditioned on input sequence
features. This reduces the action space from 185 → 144, eliminates the
tool selection layer, and focuses learning on where the actual gains are.

## Reproduction

```bash
# Reproduce tables from saved data
python results/oracle_analysis/reproduce_tables.py

# Re-run full oracle analysis (~10 hours on CPU)
CUDA_VISIBLE_DEVICES="" python oracle_analysis.py
```

## Files

- `oracle_results.npz` — raw reward/SP/TC matrices (42 cases × 185 actions)
- `oracle_summary.json` — aggregate statistics
- `reproduce_tables.py` — script to regenerate all tables from saved data
