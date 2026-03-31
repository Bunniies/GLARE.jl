# Implementation Plan: Gauge-Equivariant Neural Network for Rest-Eigen Correlator Prediction

**Goal:** Build a gauge-equivariant neural network in Julia/Flux.jl that takes local plaquette or Wilson loop
values as input (config by config) and predicts the rest-eigen correlator output for each gauge configuration.  
**Key references:** arXiv:2602.21617 (ML framework for config-by-config prediction) · arXiv:2304.10438 (Lehner & Wettig, gauge-equivariant pooling/unpooling in L-CNN style)

---

## 1. Problem Framing

The rest-eigen correlator C(t) is a gauge-invariant observable. Its fluctuation across configurations carries
information correlated with the gauge field topology. The goal is to learn a surrogate map:

```
f_θ : {U_μ(x)} → C(t)     per configuration
```

where `{U_μ(x)}` are the SU(3) link variables, and `C(t)` is a vector of (real) correlator values indexed by time slice t.
Gauge invariance of the target means that f_θ must be **gauge invariant**: it must return the same C(t) for any
gauge-equivalent configuration obtained by a local transformation Ω(x).

The approach of arXiv:2602.21617 is the blueprint: train a supervised regression model on a labeled subset of
configurations (where both the gauge field and the expensive observable are known), then predict the observable
cheaply for unlabeled configurations. Variance reduction and bias correction are then applied to the ensemble
average. This is directly applicable here.

---

## 2. Inputs and Outputs in Detail

### 2.1 Input: Local plaquettes / Wilson loops

The natural gauge-equivariant inputs at each lattice site x are:

- **Plaquette matrices** (untraced): `P_μν(x) = U_μ(x) U_ν(x+μ̂) U†_μ(x+ν̂) U†_ν(x)` ∈ SU(3)  
  These transform under gauge transformations as `P_μν(x) → Ω(x) P_μν(x) Ω†(x)` — they live in the **adjoint representation**.  
  For a 4D lattice there are 6 independent plaquette orientations (μ<ν) per site.

- **Larger Wilson loops**: 1×2, 2×2, 1×3, ... loops provide more non-local information.

**Important:** There are two possible design choices for the input features:

| Feature type | Data | Equivariance | Complexity |
|---|---|---|---|
| Gauge-invariant scalars | Re(Tr P_μν(x)) | None needed — already invariant | Simple CNN baseline |
| Untraced plaquettes P_μν(x) ∈ SU(3) | 3×3 complex matrices | Full gauge equivariance required | L-CNN / gauge-equivariant architecture |

The **gauge-equivariant approach** (working with untraced matrices) is strictly more expressive and is the one
most aligned with arXiv:2304.10438.

### 2.2 Output: Rest-eigen correlator

C(t) is a real vector of length N_t (one value per time slice). For a volume L³×T, the prediction is:

```
C_pred(t) ∈ ℝ^{N_t}     (one per configuration)
```

---

## 3. Gauge Equivariance: Core Principles

A function f is **gauge equivariant** if `f(U^Ω) = f(U)` for all local gauge transformations Ω(x).
For the output to be gauge invariant (as C(t) must be), equivariant intermediate representations
must be collapsed to invariants at the final layer.

The **L-CNN building blocks** from arXiv:2304.10438 and related work are:

1. **Parallel transport** of a covariant field Φ(x) (transforms as Φ → ΩΦΩ†):
   ```
   [T_μ Φ](x) = U_μ(x) Φ(x+μ̂) U†_μ(x)
   ```
   This transports the field from x+μ̂ back to x, keeping covariance.

2. **Bilinear product**: Given two covariant fields Φ₁, Φ₂ at the same site:
   ```
   Φ_out(x) = Φ₁(x) · Φ₂(x)    (matrix product)
   ```
   The result is still covariant.

3. **Trainable linear combination** across channels:
   ```
   Φ_out^a(x) = Σ_{μ,b} w_{μ,ab} [T_μ Φ^b](x) + w_{0,ab} Φ^b(x)
   ```

4. **Trace layer** (gauge invariance restoration):
   ```
   s(x) = Re(Tr Φ(x))    ∈ ℝ    (gauge invariant scalar)
   ```

These four operations form a complete equivariant toolkit. The plaquette itself is already a bilinear built
from links: `P_μν = U_μ T_ν U†_μ` in this language.

---

## 4. Architecture Design

### 4.1 Full Architecture (L-CNN style)

```
INPUT: U_μ(x) ∈ SU(3)     [4 links per site, full lattice L³×T]
       ↓
[Feature Extraction]
  Plaquette layer: compute P_μν(x) ∈ SU(3) for all 6 orientations
  → Covariant feature maps: Φ^a(x) ∈ M(3,ℂ), a=1..6     [each transforms as Ω·Φ·Ω†]
       ↓
[Gauge-Equivariant Conv Block × N_layers]
  For each layer:
  - Parallel transport from neighbors: T_μ Φ^a for μ=±1,±2,±3,±4
  - Linear combination with trainable weights w_{μ,ab}
  - Bilinear products between channels: Φ^a(x) · Φ^b(x) → new channels
  - Optional nonlinearity: coefficient modulation via Re(Tr Φ) scalars
       ↓
[Trace Layer]
  s^a(x) = Re(Tr Φ^a(x)) ∈ ℝ     → gauge-invariant scalar fields
       ↓
[Temporal Aggregation]
  For each time slice t: aggregate over spatial volume
    f^a(t) = (1/V₃) Σ_{x: x₄=t} s^a(x)    (spatial mean)
  → Vector f ∈ ℝ^{N_t × N_channels}
       ↓
[MLP / Temporal Decoder]
  Dense layers: ℝ^{N_t × N_channels} → ℝ^{N_t}
       ↓
OUTPUT: C_pred(t) ∈ ℝ^{N_t}
```

### 4.2 Simpler Baseline (Gauge-Invariant Inputs)

Before implementing the full equivariant architecture, build a baseline that uses
only gauge-invariant scalar features:

```
INPUT:  Re(Tr P_μν(x)) ∈ ℝ     [6 channels per site, L³×T lattice]
        ↓
[Standard 4D CNN with periodic boundary conditions]
        ↓
[Temporal pooling → MLP]
        ↓
OUTPUT: C_pred(t)
```

This is faster to implement, easier to debug, and provides a performance target for the equivariant model.

---

## 5. Critical Implementation Points

### 5.1 Extending LatticeGPU.jl for Per-Site Plaquettes

**This is the first blocker.** LatticeGPU.jl currently provides only the global trace of the plaquette.
You need the **untraced plaquette matrix at each site** for the equivariant architecture,
or at minimum its **real trace per site** for the baseline.

What needs to be added:

```julia
# Target function to implement (or wrap from existing internals)
# Returns P[x, μ, ν] ∈ SU(3) for each site x and orientation (μ,ν)
function plaquette_field(U::GaugeField)
    # U[μ][x] = link in direction μ at site x
    # P_μν(x) = U_μ(x) * U_ν(x+μ̂) * U†_μ(x+ν̂) * U†_ν(x)
    ...
end

# For baseline (gauge-invariant): just return real traces
function plaquette_scalar_field(U::GaugeField)
    # returns real array of shape [Lx, Ly, Lz, Lt, 6]
    ...
end
```

**Approach:** Inspect LatticeGPU.jl's source (especially the gauge action / plaquette routines)
and replicate the loop-product logic without taking the trace at the end.
LatticeGPU.jl almost certainly computes the per-site plaquette matrix internally;
the task is to expose that intermediate result.

### 5.2 Gauge-Equivariant Layer in Flux.jl

Implementing the parallel-transport convolution as a Flux layer:

```julia
struct GaugeEquivConv
    W_forward  # learnable weights for forward transport
    W_backward # learnable weights for backward transport
    W_self     # learnable weights for on-site term
end

function (layer::GaugeEquivConv)(U, Phi)
    # Phi[x, a] = covariant field at site x, channel a  (each is a 3×3 complex matrix)
    # For each direction μ and channel pair (a,b):
    #   output[x,a] += W_forward[μ,a,b] * U[μ,x] * Phi[x+μ̂, b] * U†[μ,x]
    #               +  W_backward[μ,a,b] * U†[μ,x-μ̂] * Phi[x-μ̂, b] * U[μ,x-μ̂]
    #               +  W_self[a,b] * Phi[x, b]
    ...
end
```

Key subtlety: the "weights" here are **scalar** coefficients (not matrices themselves).
The SU(3) structure of Φ is preserved exactly by the parallel transport; only the scalar channel-mixing
weights are learned.

### 5.3 Nonlinearity

Standard ReLU/tanh cannot be applied directly to SU(3) matrices.
Instead, use **scalar gating**: compute a gauge-invariant scalar from the field,
apply a nonlinearity to it, and use it as a multiplicative gate:

```julia
# Scalar gate: σ(Re(Tr Φ^a(x))) * Φ^a(x)
gate = activation.(real(tr.(Phi)))   # scalar per site per channel
Phi_out = gate .* Phi
```

This preserves gauge equivariance since the scalar factor commutes with gauge transformations.

### 5.4 Periodicity and Boundary Conditions

The lattice has periodic (or periodic up to a phase for fermions) boundary conditions.
**All nearest-neighbor lookups must wrap with `mod1` indexing**, not standard array indexing.
This is especially important when porting to GPU kernels.

### 5.5 Memory Layout

A 4D lattice with gauge group SU(3) has a lot of data per configuration:
- Links: 4 directions × L³×T × 9 complex numbers per site
- Plaquettes: 6 orientations × L³×T × 9 complex numbers

For a typical L=32, T=64 lattice: ≈ 4×32³×64×9×2 × 8 bytes ≈ 3.7 GB per configuration.

The training loop must be designed to avoid loading multiple full configurations simultaneously.
**Process one config at a time** (or mini-batch of configs, not sites), storing only the
compressed feature representation after the equivariant layers.

### 5.6 GPU Acceleration

LatticeGPU.jl is GPU-aware. The new equivariant layers must also support CUDA arrays.
In Flux.jl this typically means:
- Using `CuArray` for weight tensors
- Ensuring all custom operations use `CUDA.jl`-compatible kernels (or `broadcast`)
- Avoid scalar indexing on GPU arrays (use vectorized / broadcast operations throughout)

---

## 6. Step-by-Step Implementation Plan

### Phase 0: Infrastructure & Data Preparation (Weeks 1–2)

**Step 0.1: Expose per-site plaquette in LatticeGPU.jl**
- Read LatticeGPU.jl source for existing plaquette code
- Write `plaquette_field(U)` returning untraced SU(3) matrices at each site
- Write `plaquette_scalar_field(U)` returning Re(Tr P_μν(x)) for the baseline
- Test: global average of `plaquette_scalar_field` must match existing `plaquette` function output

**Step 0.2: Build dataset**
- For each configuration in your ensemble: load U, compute plaquette fields, load/compute C(t)
- Store as `(features, correlator)` pairs in an efficient format (HDF5 recommended)
- Record ensemble averages and std deviations for normalization

**Step 0.3: Data preprocessing**
- Normalize features: zero mean, unit variance per channel
- Normalize target C(t) similarly (or use log of correlator to enforce positivity)
- Split: 70% train, 15% validation, 15% test — **never mix configurations across splits**

### Phase 1: Baseline CNN (Weeks 2–3)

**Step 1.1: Implement gauge-invariant CNN in Flux.jl**
```julia
model = Chain(
    # 4D periodic conv layers
    PeriodicConv4D(6 => 32, (3,3,3,3)),   # needs custom implementation
    relu,
    PeriodicConv4D(32 => 64, (3,3,3,3)),
    relu,
    # Aggregate over spatial dimensions
    x -> dropdims(mean(x, dims=(1,2,3)), dims=(1,2,3)),  # [Lt, channels]
    # MLP decoder
    Dense(64 * N_t, 256, relu),
    Dense(256, N_t)
)
```
Note: `PeriodicConv4D` needs a custom wrapper around `Conv` with circular padding.

**Step 1.2: Training loop**
```julia
loss(x, y) = Flux.mse(model(x), y)
opt = Flux.Adam(1e-3)
Flux.train!(loss, Flux.params(model), train_data, opt)
```

**Step 1.3: Validate and benchmark**
- Compare predicted vs actual C(t) on test set
- Check correlation coefficient per time slice
- This sets the performance floor for the equivariant model

### Phase 2: Gauge-Equivariant Architecture (Weeks 3–6)

**Step 2.1: Implement SU(3) matrix operations**
```julia
# Core operations needed:
su3_product(A, B)          # 3×3 complex matrix product
su3_adj(A)                 # conjugate transpose  
su3_trace(A)               # complex trace
su3_parallel_transport(U_μ, Phi, x, μ)  # U_μ(x) * Phi(x+μ̂) * U†_μ(x)
```

**Step 2.2: Plaquette initialization layer**
```julia
struct PlaquetteLayer end
function (::PlaquetteLayer)(U)
    # Returns Phi[site, 6 channels], each Phi value is a 3×3 complex matrix
    # Phi^{μν}(x) = U_μ(x) U_ν(x+μ̂) U†_μ(x+ν̂) U†_ν(x)
end
```

**Step 2.3: Gauge-equivariant conv layer**
As described in §5.2. Start with 4 forward directions + 4 backward + self = 9 terms per output channel.

**Step 2.4: Bilinear layer (optional, adds expressivity)**
```julia
struct BilinearLayer
    W  # [n_out, n_in1, n_in2] learnable weights
end
function (layer::BilinearLayer)(Phi)
    # Phi_out^a(x) = Σ_{b,c} W_{a,b,c} Phi^b(x) · Phi^c(x)
end
```

**Step 2.5: Trace + aggregation layer**
```julia
function trace_and_aggregate(Phi, Lt)
    # Phi: covariant field, shape [sites, channels], each entry a 3×3 complex matrix
    scalars = real.(tr.(Phi))          # gauge-invariant real scalar at each site
    # Reshape to [Lx, Ly, Lz, Lt, channels]
    # Average over spatial dims for each time slice
    return mean_over_space(scalars, Lt)  # [Lt, channels]
end
```

**Step 2.6: Full model assembly**
```julia
model = Chain(
    PlaquetteLayer(),                    # U_μ(x) → Phi^{μν}(x) ∈ SU(3)  [6 channels]
    GaugeEquivConv(6 => 16),             # equivariant conv, 16 channels
    ScalarGate(),                         # nonlinearity
    GaugeEquivConv(16 => 32),
    ScalarGate(),
    BilinearLayer(32, 32 => 16),          # bilinear mixing
    GaugeEquivConv(16 => 32),
    TraceAndAggregate(Lt),                # → [Lt, 32]
    Dense(Lt * 32, 128, relu),
    Dense(128, Lt)
)
```

**Step 2.7: Test gauge equivariance explicitly**
```julia
# Generate random Omega(x) ∈ SU(3) at each site
# Compute U_transformed = gauge_transform(U, Omega)
# Assert: model(U_transformed) ≈ model(U) (to numerical precision)
```
This test should pass by construction if all layers are implemented correctly.

### Phase 3: Training & Evaluation (Weeks 6–8)

**Step 3.1: Loss function**
```julia
function loss(U_batch, C_batch)
    C_pred = model(U_batch)
    mse = mean((C_pred .- C_batch).^2)
    # Optional: add relative error weighting (C(t) decays, late times are noisy)
    # weights = 1 ./ var(C_batch, dims=1)
    return mse
end
```

**Step 3.2: Learning rate schedule**
- Start with Adam (lr=1e-3)
- Cosine annealing or ReduceLROnPlateau for convergence
- Batch size: 8–32 configurations (memory dependent)

**Step 3.3: Regularization**
- L2 weight decay on MLP layers
- Optionally add dropout in the MLP decoder (not in equivariant layers)

**Step 3.4: Evaluation metrics**
For each time slice t:
- Pearson correlation r(t) between predicted and actual C(t) across test configs
- Relative MSE: `||C_pred - C_true||² / ||C_true||²`
- Ensemble average: `<C_pred(t)>` vs `<C_true(t)>` — must agree within statistical errors
- Variance reduction factor: key metric from arXiv:2602.21617 framework

**Step 3.5: Bias correction**
Following arXiv:2602.21617, the ML estimator introduces a bias.
Apply the control-variate / multi-ensemble reweighting correction:
```
C_corrected(t) = C_pred(t) + (1/N_labeled) Σ_{configs in labeled set} [C_true - C_pred](t)
```
This makes the estimator unbiased.

### Phase 4: Extensions and Optimization (Weeks 8+)

**Step 4.1: Richer input features**
- Add 1×2 (rectangle) Wilson loops as additional input channels
- Add longer Wilson lines (Wilson flow smeared links) to reduce UV noise

**Step 4.2: Pooling / multiscale architecture**
Implement gauge-equivariant pooling as in arXiv:2304.10438:
- Coarsen the lattice by a factor of 2 in spatial directions
- Build a U-Net style hierarchy: equivariant conv at multiple scales
- This captures both short-range and long-range gauge correlations

**Step 4.3: Ensemble generalization**
Test whether the model trained on one ensemble (β, L) generalizes to:
- Different gauge coupling β (renormalization group flow)
- Different volumes L (finite-volume effects)
- Different smearing levels

---

## 7. LatticeGPU.jl Integration Notes

The key modification needed is to expose the **per-site, untraced plaquette** from the existing
plaquette computation. Looking at typical lattice code structure, this involves:

```julia
# Pseudocode: what the internals likely look like
for x in lattice_sites
    for (mu, nu) in plaquette_orientations
        P = U[mu,x] * U[nu, x+mu] * U[mu, x+nu]' * U[nu, x]'
        plaquette_sum += real(tr(P))   # <-- this is what's currently done
        # You need: plaquette_field[x, mu, nu] = P  (before the trace)
    end
end
```

The `U[mu, x+mu]` step requires correct periodic boundary indexing — this already works in LatticeGPU.jl
for the trace version, so the new code should reuse that indexing logic exactly.

**Practical approach:** Contact the LatticeGPU.jl developers (A. Ramos group, IFIC Valencia) or
open a PR/issue requesting a `plaquette_matrix_field` function. Alternatively, write a wrapper
that computes it independently using LatticeGPU.jl's link access API.

---

## 8. On Using Claude Code

Yes, **Claude Code would be very useful here**, specifically for:

**Where it helps most:**
- Writing boilerplate Julia code for custom Flux layers (equivariant conv, trace layer, etc.)
- Debugging tensor shape issues interactively (4D lattice indexing is error-prone)
- Writing and running gauge-equivariance unit tests
- Setting up the HDF5 dataset pipeline
- Iterating quickly on architecture variants
- Writing the training loop with proper logging/checkpointing

**Where to be careful:**
- Claude Code cannot access the internal GitLab of LatticeGPU.jl without credentials;
  provide the relevant source files or API documentation in context
- For GPU debugging (CUDA.jl issues), Claude Code works best if you can run tests
  interactively and share error messages

**Suggested workflow with Claude Code:**
1. Provide the LatticeGPU.jl interface (type definitions + function signatures) as context
2. Ask it to implement `plaquette_field` using that interface
3. Iteratively build and test each layer: `PlaquetteLayer` → `GaugeEquivConv` → tests
4. Once layers pass the equivariance test, assemble the full model
5. Use it to help write the training loop and evaluation metrics

---

## 9. Dependencies and Environment

```julia
# Julia packages needed (add to Project.toml)
[deps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
HDF5 = "f67ccb44-e63f-a3bf-81d4-5cdba7a07e88"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
# LatticeGPU = from igit.ific.uv.es/alramos/latticegpu.jl
```

---

## 10. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| LatticeGPU.jl doesn't expose per-site plaquette | High | Write independent wrapper; contact developers |
| Equivariant layers slow on GPU | Medium | Profile; consider fused CUDA kernels; use batched matmul |
| Not enough labeled configurations | Medium | Use variance-reduction framing from 2602.21617; even 10–20% labeled is enough |
| Model doesn't generalize across gauge topologies | Medium | Augment training with topology-diverse configs; check by sector |
| Gauge equivariance breaks in practice due to floating-point | Low | Enforce to double precision in tests; monitor drift |
| Julia/Flux ecosystem gaps (custom AD rules needed) | Medium | May need `ChainRulesCore.rrule` for SU(3) matrix ops |

---

## 11. Suggested Reading Order

1. arXiv:2602.21617 — the ML regression framework (your application model)
2. arXiv:2304.10438 — Lehner & Wettig, gauge-equivariant pooling (your architecture model)
3. arXiv:2003.06413 — Favoni et al., Lattice Gauge Equivariant CNNs (L-CNN, foundational architecture paper)
4. arXiv:2602.23840 — Pfahler et al. (2026), novel gauge-equivariant architecture for Dirac preconditioner
5. arXiv:2501.16955 — CASK, gauge-covariant Transformer (for future extensions)
