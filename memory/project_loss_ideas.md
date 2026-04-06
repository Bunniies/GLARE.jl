---
name: Loss function ideas for Phase 3
description: Time-weighted MSE and r(t)-maximizing loss as Phase 3 training improvements
type: project
---

Two loss function ideas to revisit in Phase 3 training tuning:

1. **Time-weighted MSE**: `L = Σ_t w(t) · MSE(t)` where `w(t)` steers network
   capacity toward timeslices where the gauge field is actually predictive.
   After z-score normalization the unweighted MSE is already 1/Var-weighted in
   the original space, so `w(t)` would be an additional bias toward certain t.

2. **r(t)-maximizing loss**: `L = Σ_t (1 - r(t)²)` — directly minimizes the
   variance reduction factor. This is the quantity that enters the cost-reduction
   estimator and is the true scientific objective.

**Why:** Current normalized MSE and r-maximization are equivalent at the optimum
(same minimizer), but a direct r(t) loss or time-weighted loss could improve
convergence when network capacity is spread thin across Lt=48 outputs.

**How to apply:** Implement as alternative loss options in train_baseline.jl /
train_lcnn.jl during Phase 3 hyperparameter tuning, after the architecture is
fixed and the full dataset is available.
