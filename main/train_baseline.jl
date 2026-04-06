using GLARE
using Flux
using HDF5
using Statistics
using Printf
using Random
using PyPlot

# ---------------------------------------------------------------------------
# Configuration — edit these paths and hyperparameters
# ---------------------------------------------------------------------------
ENV["GLARE_TEST_GAUGE_H5"] = "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_gauge_scalar_1_680.h5"
ENV["GLARE_TEST_CORR_H5"]  = "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_corr.h5"

gauge_h5 = get(ENV, "GLARE_TEST_GAUGE_H5", "")
corr_h5  = get(ENV, "GLARE_TEST_CORR_H5", "")

isfile(gauge_h5) || error("Gauge HDF5 not found: $gauge_h5")
isfile(corr_h5)  || error("Correlator HDF5 not found: $corr_h5")

# Output directory for logs and plots
LOG_DIR  = get(ENV, "GLARE_LOG_DIR",
               "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/training/")
mkpath(LOG_DIR)

# Lattice dimensions — read from HDF5 metadata to stay in sync with the database
Lt, Ls, npls = HDF5.h5open(gauge_h5, "r") do fid
    vol  = read(fid["metadata"]["vol"])   # Int64[4]
    ps   = read(fid["configs"][first(keys(fid["configs"]))]["plaq_scalar"])
    vol[1], vol[2], size(ps, 5)
end

npol = 3
POLARIZATIONS = ["g1-g1", "g2-g2", "g3-g3"]

# Hyperparameters
CHANNELS   = [16, 16]
MLP_HIDDEN = 128
LR         = 1e-3
EPOCHS     = 30
BATCH_SIZE = 32

# Set true to preload all splits into RAM before training (eliminates
# per-batch HDF5 reads). Set false to keep the per-batch HDF5 path,
# which is useful when RAM is limited or for quick diagnostic runs.
USE_PRELOAD = false

# ---------------------------------------------------------------------------
# Data split and normalization
# ---------------------------------------------------------------------------

println("Splitting configurations...")
train_ids, val_ids, test_ids, bc_ids =
    split_configs(gauge_h5; train=0.70, val=0.15, test=0.15, bias_corr=0.0)

@printf("Split: %d train / %d val / %d test / %d bc\n",
        length(train_ids), length(val_ids), length(test_ids), length(bc_ids))

println("Computing normalization statistics from train set...")
stats = compute_normalization(gauge_h5, corr_h5, train_ids;
                              polarizations=POLARIZATIONS)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

# --- per-batch HDF5 path (used when USE_PRELOAD = false) ---

function load_one(cid::String)
    feat5d = load_gauge(gauge_h5, cid; stats=stats)
    corr_pols = [load_corr(corr_h5, cid; stats=stats, polarization=pol)
                 for pol in POLARIZATIONS]
    corr2d = hcat([vec(mean(c, dims=2)) for c in corr_pols]...)  # (Lt, npol)
    return Float32.(feat5d), Float32.(corr2d)
end

function load_batch(ids::Vector{String})
    pairs = [load_one(cid) for cid in ids]
    feat  = cat([p[1] for p in pairs]...; dims=6)
    corr  = cat([reshape(p[2], Lt, npol, 1) for p in pairs]...; dims=3)
    return feat, corr
end

# --- in-memory path (used when USE_PRELOAD = true) ---

function load_batch(ids::Vector{String}, cache::PreloadedDataset)
    pairs = [cache[cid] for cid in ids]
    feat  = cat([p[1] for p in pairs]...; dims=6)
    corr  = cat([reshape(p[2], Lt, npol, 1) for p in pairs]...; dims=3)
    return feat, corr
end

# --- unified helper: dispatches based on whether a cache is supplied ---

get_batch(ids, cache=nothing) =
    isnothing(cache) ? load_batch(ids) : load_batch(ids, cache)

# ---------------------------------------------------------------------------
# Optional dataset preloading
# ---------------------------------------------------------------------------

if USE_PRELOAD
    println("Preloading datasets into RAM...")
    train_cache = preload_dataset(gauge_h5, corr_h5, train_ids, stats;
                                  polarizations=POLARIZATIONS)
    val_cache   = preload_dataset(gauge_h5, corr_h5, val_ids,   stats;
                                  polarizations=POLARIZATIONS)
    test_cache  = preload_dataset(gauge_h5, corr_h5, test_ids,  stats;
                                  polarizations=POLARIZATIONS)
    @printf("  Preloaded %d train / %d val / %d test configs\n",
            length(train_ids), length(val_ids), length(test_ids))
else
    train_cache = nothing
    val_cache   = nothing
    test_cache  = nothing
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

# --- Gradient sanity check ---
let
    _feats, _corrs = get_batch(train_ids[1:2], train_cache)
    _, _grads = Flux.withgradient(model) do m
        mean((m(_feats) .- _corrs).^2)
    end
    g = _grads[1]
    using LinearAlgebra
    @printf("Gradient norms — conv1.weight: %.2e  conv2.weight: %.2e  dense1.weight: %.2e\n",
            norm(g.layers[1].conv.weight),
            norm(g.layers[2].conv.weight),
            norm(g.layers[5].weight))
end

##

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

mse_loss(ŷ, y) = mean((ŷ .- y) .^ 2)

function batch_loss(model, feats, corrs)
    ŷ = model(feats)
    return mse_loss(ŷ, corrs)
end

# Does the batched loss correlate with r? Quick sanity check:
_feats_big, _corrs_big = get_batch(train_ids[1:32], train_cache)
@printf("32-sample batch loss: %.5f\n", batch_loss(model, _feats_big, _corrs_big))

# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

function evaluate(model, ids, cache=nothing; batch_size=BATCH_SIZE)
    total_loss = 0.0
    n_batches  = 0
    all_pred   = Array{Float32}(undef, Lt, npol, 0)
    all_true   = Array{Float32}(undef, Lt, npol, 0)

    for start in 1:batch_size:length(ids)
        batch_ids = ids[start:min(start + batch_size - 1, end)]
        feats, corrs = get_batch(batch_ids, cache)
        pred = model(feats)
        total_loss += mse_loss(pred, corrs)
        n_batches  += 1
        all_pred = cat(all_pred, pred;  dims=3)
        all_true = cat(all_true, corrs; dims=3)
    end

    r_per_t = pearson_r(Array{Float32}(all_pred), Array{Float32}(all_true))
    return total_loss / n_batches, r_per_t, all_pred, all_true
end

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

println("\n--- Training Phase-1 Baseline CNN ---")
println("Epochs: $EPOCHS  |  Batch: $BATCH_SIZE  |  LR: $LR")
println("Channels: $CHANNELS  |  MLP hidden: $MLP_HIDDEN\n")

log_path = joinpath(LOG_DIR, "training_log.csv")
open(log_path, "w") do io
    println(io, "epoch,train_loss,val_loss,r_mean,r_midt")
end

train_losses = Float64[]
val_losses   = Float64[]

for epoch in 1:EPOCHS
    perm = randperm(length(train_ids))
    epoch_loss = 0.0
    n_batches  = 0

    for start in 1:BATCH_SIZE:length(train_ids)
        batch_ids = train_ids[perm[start:min(start + BATCH_SIZE - 1, end)]]
        feats, corrs = get_batch(batch_ids, train_cache)

        loss_val, grads = Flux.withgradient(model) do m
            batch_loss(m, feats, corrs)
        end

        Flux.update!(opt_state, model, grads[1])
        epoch_loss += loss_val
        n_batches  += 1
    end

    train_loss = epoch_loss / n_batches
    val_loss, r_per_t, _, _ = evaluate(model, val_ids, val_cache)
    r_mean = mean(r_per_t)
    r_midt = mean(r_per_t[div(Lt, 4):3*div(Lt, 4)])

    push!(train_losses, train_loss)
    push!(val_losses,   val_loss)

    @printf("Epoch %3d/%d  train_loss=%.5f  val_loss=%.5f  r̄=%.3f  r̄(mid-t)=%.3f\n",
            epoch, EPOCHS, train_loss, val_loss, r_mean, r_midt)

    open(log_path, "a") do io
        @printf(io, "%d,%.6f,%.6f,%.6f,%.6f\n",
                epoch, train_loss, val_loss, r_mean, r_midt)
    end
end

# ---------------------------------------------------------------------------
# Loss curve plot
# ---------------------------------------------------------------------------
##


fig, ax = subplots(figsize=(7, 4))
ax.plot(1:EPOCHS, train_losses, label="train")
ax.plot(1:EPOCHS, val_losses,   label="val")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss (normalised)")
ax.set_title("Baseline CNN — loss curve")
ax.legend()
fig.tight_layout()
savefig(joinpath(LOG_DIR, "loss_curve.pdf"))
close(fig)

# ---------------------------------------------------------------------------
# Final evaluation on test set
# ---------------------------------------------------------------------------

##

println("\n--- Test set evaluation ---")
test_loss, r_per_t, test_pred, test_true = evaluate(model, test_ids, test_cache)
@printf("Test MSE loss : %.5f\n", test_loss)
@printf("Pearson r mean: %.3f\n", mean(r_per_t))
@printf("Pearson r mid-t (t=%d..%d): %.3f\n",
        div(Lt, 4), 3*div(Lt, 4), mean(r_per_t[div(Lt, 4):3*div(Lt, 4)]))

println("\nr(t) per time slice:")
for t in 1:Lt
    @printf("  t=%2d  r=%.3f\n", t, r_per_t[t])
end
##

# ---------------------------------------------------------------------------
# r(t) plot
# ---------------------------------------------------------------------------

fig, ax = subplots(figsize=(8, 4))
ax.plot(1:Lt, r_per_t, marker="o", markersize=3, linewidth=1)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("t")
ax.set_ylabel("Pearson r(t)")
ax.set_title("Baseline CNN — Pearson r per time slice (test set)")
ax.set_xlim(1, Lt)
fig.tight_layout()
savefig(joinpath(LOG_DIR, "pearson_r.pdf"))
close(fig)

# ---------------------------------------------------------------------------
# Predicted vs true correlator (test set, mean over configs, per polarization)
# ---------------------------------------------------------------------------

# Denormalise: ŷ_phys(t) = ŷ_norm(t) * σ(t) + μ(t)
corr_mean = Float32.(stats.corr_mean)
corr_std  = Float32.(stats.corr_std)

pred_phys = test_pred .* corr_std .+ corr_mean   # (Lt, npol, N_test)
true_phys = test_true .* corr_std .+ corr_mean

# Config-averaged correlator per polarization
pred_mean = mean(pred_phys, dims=3)[:, :, 1]   # (Lt, npol)
true_mean = mean(true_phys, dims=3)[:, :, 1]

fig, axes = subplots(1, npol; figsize=(5*npol, 4), sharey=false)
for (ipol, pol) in enumerate(POLARIZATIONS)
    ax = axes[ipol]
    ax.semilogy(1:Lt, abs.(true_mean[:, ipol]), label="exact",     linewidth=1.5)
    ax.semilogy(1:Lt, abs.(pred_mean[:, ipol]), label="NN pred",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("t")
    ax.set_title(pol)
    ax.legend(fontsize=8)
end
axes[1].set_ylabel("C(t)")
fig.suptitle("Baseline CNN — predicted vs exact correlator (test set, config average)")
fig.tight_layout()
savefig(joinpath(LOG_DIR, "correlator_comparison.pdf"))
close(fig)

##
# Per-config scatter at mid-t (one point per config, one panel per polarization)
t_mid = div(Lt, 2)
fig, axes = subplots(1, npol; figsize=(5*npol, 4))
for (ipol, pol) in enumerate(POLARIZATIONS)
    ax = axes[ipol]
    x = vec(true_phys[t_mid, ipol, :])
    y = vec(pred_phys[t_mid, ipol, :])
    ax.scatter(x, y, s=10, alpha=0.6)
    lo, hi = min(minimum(x), minimum(y)), max(maximum(x), maximum(y))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax.set_xlabel("C_exact(t=$t_mid)")
    ax.set_ylabel("C_pred(t=$t_mid)")
    ax.set_title("$(pol)  r=$(round(r_per_t[t_mid], digits=3))")
end
fig.suptitle("Baseline CNN — scatter at t=$t_mid (test set)")
fig.tight_layout()
savefig(joinpath(LOG_DIR, "scatter_midt.pdf"))
close(fig)

println("\nLogs and plots written to: $LOG_DIR")
