using GLARE
using Flux
using HDF5
using Statistics
using Printf
using Random

# ---------------------------------------------------------------------------
# Configuration — edit these paths and hyperparameters
# ---------------------------------------------------------------------------

gauge_h5 = get(ENV, "GLARE_TEST_GAUGE_H5",
               "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_gauge_scalar.h5")
corr_h5  = get(ENV, "GLARE_TEST_CORR_H5",
               "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_corr.h5")

isfile(gauge_h5) || error("Gauge HDF5 not found: $gauge_h5")
isfile(corr_h5)  || error("Correlator HDF5 not found: $corr_h5")

# Lattice dimensions — read from HDF5 metadata to stay in sync with the database
Lt, Ls, npls = HDF5.h5open(gauge_h5, "r") do fid
    vol  = read(fid["metadata"]["vol"])   # Int64[4]
    ps   = read(fid["configs"][first(keys(fid["configs"]))]["plaq_scalar"])
    vol[1], vol[2], size(ps, 5)
end

npol = 3          # polarizations: g1-g1, g2-g2, g3-g3

POLARIZATIONS = ["g1-g1", "g2-g2", "g3-g3"]

# Hyperparameters
CHANNELS   = [16, 16]
MLP_HIDDEN = 128
LR         = 1e-3
EPOCHS     = 50
BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# Data split and normalization
# ---------------------------------------------------------------------------

println("Splitting configurations...")
train_ids, val_ids, test_ids, bc_ids =
    split_configs(gauge_h5; train=0.60, val=0.15, test=0.15, bias_corr=0.10)

@printf("Split: %d train / %d val / %d test / %d bc\n",
        length(train_ids), length(val_ids), length(test_ids), length(bc_ids))

println("Computing normalization statistics from train set...")
stats = compute_normalization(gauge_h5, corr_h5, train_ids;
                              polarizations=POLARIZATIONS)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

"""
    load_one(cid) -> (feat5d::Array{Float32,5}, corr2d::Array{Float32,2})

Load one configuration: plaq_scalar in spatial layout `(Lt, Ls, Ls, Ls, npls)`
and stack all three polarizations (source-averaged) into `(Lt, npol)`.
Both are normalised.
"""
function load_one(cid::String)
    feat5d = load_gauge(gauge_h5, cid; stats=stats)      # (Lt, Ls, Ls, Ls, npls)

    corr_pols = [load_corr(corr_h5, cid; stats=stats, polarization=pol)
                 for pol in POLARIZATIONS]               # each (T, nsrcs)

    # source-average each polarization, stack → (Lt, npol)
    corr2d = hcat([vec(mean(c, dims=2)) for c in corr_pols]...)  # (Lt, npol)

    return Float32.(feat5d), Float32.(corr2d)
end

"""
    load_batch(ids) -> (feat::Array{Float32,6}, corr::Array{Float32,3})

Load a batch of configs and stack them into:
- `feat`: `(Lt, Ls, Ls, Ls, npls, B)` — Flux Conv4D format
- `corr`: `(Lt, npol, B)`
"""
function load_batch(ids::Vector{String})
    pairs = [load_one(cid) for cid in ids]
    feat  = cat([p[1] for p in pairs]...; dims=6)  # (..., B)
    corr  = cat([reshape(p[2], Lt, npol, 1) for p in pairs]...; dims=3)
    return feat, corr
end

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

println("Building model...")
model = build_baseline_cnn(;
    Lt         = Lt,
    npls       = npls,
    npol       = npol,
    channels   = CHANNELS,
    mlp_hidden = MLP_HIDDEN)

opt_state = Flux.setup(Adam(LR), model)

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

mse_loss(ŷ, y) = mean((ŷ .- y) .^ 2)

function batch_loss(model, feats, corrs)
    ŷ = model(feats)
    return mse_loss(ŷ, corrs)
end

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

function evaluate(model, ids; batch_size=BATCH_SIZE)
    total_loss = 0.0
    n_batches  = 0
    all_pred   = Array{Float32}(undef, Lt, npol, 0)
    all_true   = Array{Float32}(undef, Lt, npol, 0)

    for start in 1:batch_size:length(ids)
        batch_ids = ids[start:min(start + batch_size - 1, end)]
        feats, corrs = load_batch(batch_ids)
        pred = model(feats)
        total_loss += mse_loss(pred, corrs)
        n_batches  += 1
        all_pred = cat(all_pred, pred;  dims=3)
        all_true = cat(all_true, corrs; dims=3)
    end

    r_per_t = pearson_r(Array{Float32}(all_pred), Array{Float32}(all_true))
    return total_loss / n_batches, r_per_t
end

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

println("Pre-loading validation set...")
val_feats, val_corrs = load_batch(val_ids)

println("\n--- Training Phase-1 Baseline CNN ---")
println("Epochs: $EPOCHS  |  Batch: $BATCH_SIZE  |  LR: $LR")
println("Channels: $CHANNELS  |  MLP hidden: $MLP_HIDDEN\n")

for epoch in 1:EPOCHS
    # Shuffle training order each epoch
    perm = randperm(length(train_ids))
    epoch_loss = 0.0
    n_batches  = 0

    for start in 1:BATCH_SIZE:length(train_ids)
        batch_ids = train_ids[perm[start:min(start + BATCH_SIZE - 1, end)]]
        feats, corrs = load_batch(batch_ids)

        loss_val, grads = Flux.withgradient(model) do m
            batch_loss(m, feats, corrs)
        end

        Flux.update!(opt_state, model, grads[1])
        epoch_loss += loss_val
        n_batches  += 1
    end

    train_loss = epoch_loss / n_batches
    val_loss, r_per_t = evaluate(model, val_ids)
    r_mean  = mean(r_per_t)
    r_midt  = mean(r_per_t[div(Lt, 4):3*div(Lt, 4)])  # mid-t window

    @printf("Epoch %3d/%d  train_loss=%.5f  val_loss=%.5f  r̄=%.3f  r̄(mid-t)=%.3f\n",
            epoch, EPOCHS, train_loss, val_loss, r_mean, r_midt)
end

# ---------------------------------------------------------------------------
# Final evaluation on test set
# ---------------------------------------------------------------------------

println("\n--- Test set evaluation ---")
test_loss, r_per_t = evaluate(model, test_ids)
@printf("Test MSE loss : %.5f\n", test_loss)
@printf("Pearson r mean: %.3f\n", mean(r_per_t))
@printf("Pearson r mid-t (t=%d..%d): %.3f\n",
        div(Lt, 4), 3*div(Lt, 4), mean(r_per_t[div(Lt, 4):3*div(Lt, 4)]))

println("\nr(t) per time slice:")
for t in 1:Lt
    @printf("  t=%2d  r=%.3f\n", t, r_per_t[t])
end
