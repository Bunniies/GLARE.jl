# Preprocessing

Data splitting, normalization, and loading utilities.

## Splitting

Configs are split into four non-overlapping subsets using a generalised
Bresenham / DDA algorithm that preserves Monte Carlo chain order and
maximises separation between consecutive configs in the same split.
`train_ids ∩ bc_ids = ∅` by construction.

```@docs
split_configs
```

## Normalization

Global z-score normalization computed from the training split only.
Per-config normalization is explicitly avoided: it would destroy the
config-to-config fluctuations that are the regression signal.

```@docs
NormStats
compute_normalization
save_normalization
load_normalization
```

## Loading

```@docs
load_gauge
load_corr
load_config
load_split
```
