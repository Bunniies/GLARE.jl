---
name: Data scale and IO optimization ideas
description: Measured data sizes and future IO/GPU improvements to implement
type: project
---

**Measured:** A654 plaq_scalar HDF5 with 680 configs = 21 GB (Float64).
Per-config size: Float64[48,24,24,24,6] = 3,981,312 elements × 8 bytes ≈ 30 MB.

**Projected full ensemble (~2000 configs):**
- Float64: ~60 GB
- Float32: ~30 GB

**Two improvements to implement before full-scale training:**

1. **Float32 storage in `build_gauge_dataset`**: cast `plaq_scalar` to Float32 before
   writing to HDF5. Halves disk/RAM usage with no NN-relevant precision loss
   (plaquette values ∈ [-3,3], Float32 gives 7 significant digits).
   Change `_to_spatial_scalar` return type and HDF5 write in `Dataset.jl`.

2. **In-memory preloading in `train_baseline.jl`**: at startup, load the entire
   training set into a single `Float32[Lt,Ls,Ls,Ls,npls,N_train]` array and
   corresponding `Float32[Lt,npol,N_train]` targets. Eliminates per-batch HDF5
   reads, converting disk-bound training to compute-bound.
   Only viable once Float32 storage is in place (~15 GB for 500 train configs).

3. **GPU (Phase 2+)**: worth implementing once L-CNN with SU(3) matrix input is
   in place. Matrix db is ~8× larger per config. Baseline CNN is too small
   (~50K params) for GPU to help — bottleneck is IO not compute.

**How to apply:** implement (1) and (2) before running full-dataset baseline training.
GPU support is a Phase 2 task, not urgent for Phase 1.
