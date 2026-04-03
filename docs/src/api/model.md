# Model

## Phase 1 — Baseline CNN

A conventional 4D CNN operating on scalar plaquette features.
Used to establish a performance floor before the gauge-equivariant L-CNN (Phase 2).

**Architecture:**
```
Input: (Lt, Ls, Ls, Ls, npls, B)
  └─ PeriodicConv4D → relu  (×n_layers)
  └─ spatial mean over (Ls, Ls, Ls)   →  (Lt, channels_last, B)
  └─ MLP: Linear → relu → Linear      →  (Lt × npol, B)
  └─ reshape                           →  (Lt, npol, B)
```

```@docs
PeriodicConv4D
build_baseline_cnn
```

## Metrics

```@docs
pearson_r
```
