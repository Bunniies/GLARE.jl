using GLARE
using Flux
using HDF5
using JLD2
using Statistics
using Printf
using Random
using LinearAlgebra
# ENV["MPLBACKEND"] = "Agg"   # headless rendering — no display required
using PyPlot

# ---------------------------------------------------------------------------
# Device selection — GPU if available, CPU otherwise.
# Requires CUDA.jl to be installed separately.
# ---------------------------------------------------------------------------
const device = Flux.gpu_device()

# ---------------------------------------------------------------------------
# Configuration — edit these paths and hyperparameters
# ---------------------------------------------------------------------------
gauge_h5 = get(ENV, "GLARE_GAUGE_H5",
    "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_gauge_scalar.h5")
corr_h5  = get(ENV, "GLARE_CORR_H5",
    "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_corr.h5")

isfile(gauge_h5) || error("Gauge HDF5 not found: $gauge_h5")
isfile(corr_h5)  || error("Correlator HDF5 not found: $corr_h5")

# Output directory for logs, plots, and checkpoints
LOG_DIR = get(ENV, "GLARE_LOG_DIR",
    "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/training_baseline/")
mkpath(LOG_DIR)

CHECKPOINT_PATH = joinpath(LOG_DIR, "baseline_best.jld2")    # best val-loss checkpoint
FINAL_PATH      = joinpath(LOG_DIR, "baseline_final.jld2")   # model at last epoch
CONFIG_PATH     = joinpath(LOG_DIR, "baseline_config.toml")  # human-readable run config

# Lattice dimensions — read from HDF5 metadata to stay in sync with the database
Lt, Ls, npls = HDF5.h5open(gauge_h5, "r") do fid
    vol  = read(fid["metadata"]["vol"])   # Int64[4]: lp.iL = (Lx, Ly, Lz, Lt) — t at index 4
    ps   = read(fid["configs"][first(keys(fid["configs"]))]["plaq_scalar"])
    vol[4], vol[1], size(ps, 5)           # Lt = vol[4], Ls = vol[1]
end

npol = 3
POLARIZATIONS = ["g1-g1", "g2-g2", "g3-g3"]

# Hyperparameters
CHANNELS     = [16, 16]
MLP_HIDDEN   = 128
DROPOUT      = 0.3    # dropout between Dense layers — prevents memorisation of training configs
LR           = 3e-4   # reduced from 1e-3; overfitting was fast at 1e-3 with r_loss
WEIGHT_DECAY = 1e-2   # increased from 1e-4; stronger L2 needed with r_loss + small batch
EPOCHS       = 200
BATCH_SIZE   = 32

# Set true (or set ENV GLARE_USE_PRELOAD=true) to preload all splits into RAM
# before training (eliminates per-batch HDF5 reads). Set false for quick
# diagnostic runs or when RAM is limited.
USE_PRELOAD = true # parse(Bool, get(ENV, "GLARE_USE_PRELOAD", "false"))

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
    mlp_hidden = MLP_HIDDEN,
    dropout    = DROPOUT) |> device

# opt_state must be set up AFTER the model is on the target device.
opt_state = Flux.setup(Adam(LR, (0.9, 0.999), WEIGHT_DECAY), model)

ps, _   = Flux.destructure(model)
n_params = length(ps)
@printf("Baseline CNN: npls=%d  channels=%s  mlp_hidden=%d\n",
        npls, string(CHANNELS), MLP_HIDDEN)
@printf("Parameters: %d\n", n_params)

# ---------------------------------------------------------------------------
# Write run config to TOML for reproducibility and post-hoc checks
# ---------------------------------------------------------------------------
open(CONFIG_PATH, "w") do io
    println(io, "# Baseline CNN training run configuration")
    println(io, "# Generated automatically — do not edit by hand\n")

    println(io, "[lattice]")
    @printf(io, "Lt   = %d\n", Lt)
    @printf(io, "Ls   = %d\n", Ls)
    @printf(io, "npls = %d    # plaquette planes\n", npls)
    println(io)

    println(io, "[data]")
    @printf(io, "gauge_h5      = \"%s\"\n", gauge_h5)
    @printf(io, "corr_h5       = \"%s\"\n", corr_h5)
    @printf(io, "polarizations = %s\n", string(POLARIZATIONS))
    @printf(io, "n_train       = %d\n", length(train_ids))
    @printf(io, "n_val         = %d\n", length(val_ids))
    @printf(io, "n_test        = %d\n", length(test_ids))
    @printf(io, "n_bc          = %d\n", length(bc_ids))
    @printf(io, "preload       = %s\n", string(USE_PRELOAD))
    println(io)

    println(io, "[model]")
    @printf(io, "channels   = %s\n", string(CHANNELS))
    @printf(io, "mlp_hidden = %d\n", MLP_HIDDEN)
    @printf(io, "n_params   = %d\n", n_params)
    println(io)

    println(io, "[training]")
    @printf(io, "epochs       = %d\n", EPOCHS)
    @printf(io, "batch_size   = %d\n", BATCH_SIZE)
    @printf(io, "lr           = %.2e\n", LR)
    @printf(io, "weight_decay = %.2e\n", WEIGHT_DECAY)
    @printf(io, "optimizer    = \"Adam\"\n")
    @printf(io, "device       = \"%s\"\n", string(device))
    println(io)

    println(io, "[output]")
    @printf(io, "log_dir          = \"%s\"\n", LOG_DIR)
    @printf(io, "checkpoint_best  = \"%s\"\n", CHECKPOINT_PATH)
    @printf(io, "checkpoint_final = \"%s\"\n", FINAL_PATH)
    @printf(io, "training_log     = \"%s\"\n", joinpath(LOG_DIR, "baseline_training_log.csv"))
end
println("Run config written to: $CONFIG_PATH")

# ---------------------------------------------------------------------------
# Gradient sanity check
# ---------------------------------------------------------------------------
let
    _feats, _corrs = get_batch(train_ids[1:2], train_cache)
    _feats = _feats |> device
    _corrs = _corrs |> device
    _, _grads = Flux.withgradient(model) do m
        mean((m(_feats) .- _corrs).^2)
    end
    g = _grads[1]
    @printf("Gradient norms — conv1.weight: %.2e  conv2.weight: %.2e  dense1.weight: %.2e\n",
            norm(g.layers[1].conv.weight),
            norm(g.layers[2].conv.weight),
            norm(g.layers[5].weight))
end

# ---------------------------------------------------------------------------
# Loss and evaluation
# ---------------------------------------------------------------------------

mse_loss(ŷ, y) = mean((ŷ .- y) .^ 2)
r_loss(ŷ, y)   = pearson_r_loss(ŷ, y)   # -mean r(t,pol) over batch

function evaluate(model, ids, cache=nothing; batch_size=BATCH_SIZE)
    Flux.testmode!(model)
    total_loss = 0.0
    n_batches  = 0
    all_pred   = Array{Float32}(undef, Lt, npol, 0)
    all_true   = Array{Float32}(undef, Lt, npol, 0)

    for start in 1:batch_size:length(ids)
        batch_ids    = ids[start:min(start + batch_size - 1, end)]
        feats, corrs = get_batch(batch_ids, cache)
        pred         = model(feats |> device)
        pred_cpu     = Flux.cpu(pred)
        total_loss  += Float64(mse_loss(pred_cpu, corrs))
        n_batches   += 1
        all_pred = cat(all_pred, pred_cpu; dims=3)
        all_true = cat(all_true, corrs;    dims=3)
    end

    Flux.trainmode!(model)
    r_per_t = pearson_r(all_pred, all_true)
    return total_loss / n_batches, r_per_t, all_pred, all_true
end

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

println("\n--- Training Phase-1 Baseline CNN ---")
@printf("Epochs: %d  |  Batch: %d  |  LR: %.0e\n", EPOCHS, BATCH_SIZE, LR)
@printf("Channels: %s  |  MLP hidden: %d  |  Device: %s\n\n",
        string(CHANNELS), MLP_HIDDEN, string(device))

log_path = joinpath(LOG_DIR, "baseline_training_log.csv")
open(log_path, "w") do io
    println(io, "epoch,train_rloss,val_mse,r_mean,r_midt")
end

train_rlosses = Float64[]
val_losses    = Float64[]
best_r_mean   = -Inf
r_midt        = NaN

for epoch in 1:EPOCHS
    perm = randperm(length(train_ids))
    epoch_rloss = 0.0
    n_batches   = 0

    for start in 1:BATCH_SIZE:length(train_ids)
        batch_ids    = train_ids[perm[start:min(start + BATCH_SIZE - 1, end)]]
        feats, corrs = get_batch(batch_ids, train_cache)
        feats        = feats  |> device
        corrs        = corrs  |> device

        loss_val, grads = Flux.withgradient(model) do m
            r_loss(m(feats), corrs)
        end

        Flux.update!(opt_state, model, grads[1])
        epoch_rloss += Float64(loss_val)
        n_batches   += 1
    end

    train_rloss = epoch_rloss / n_batches
    val_loss, r_per_t, _, _ = evaluate(model, val_ids, val_cache)
    r_mean = mean(r_per_t)
    r_midt = mean(r_per_t[div(Lt, 4):3*div(Lt, 4)])

    push!(train_rlosses, train_rloss)
    push!(val_losses,    val_loss)

    # Cosine annealing: LR decays from LR → 0 over EPOCHS
    new_lr = LR * (1 + cos(π * epoch / EPOCHS)) / 2
    Flux.Optimisers.adjust!(opt_state, new_lr)

    improved = r_mean > best_r_mean
    if improved
        best_r_mean = r_mean
        jldsave(CHECKPOINT_PATH; model=Flux.cpu(model), epoch=epoch,
                val_loss=val_loss, r_midt=r_midt)
    end

    @printf("Epoch %3d/%d  r_loss=%.4f  val_mse=%.4f  r̄=%.3f  r̄(mid-t)=%.3f%s\n",
            epoch, EPOCHS, train_rloss, val_loss, r_mean, r_midt,
            improved ? "  ✓ saved" : "")

    open(log_path, "a") do io
        @printf(io, "%d,%.6f,%.6f,%.6f,%.6f\n",
                epoch, train_rloss, val_loss, r_mean, r_midt)
    end
end

# Save the final model regardless of whether it is the best.
jldsave(FINAL_PATH; model=Flux.cpu(model), epoch=EPOCHS,
        val_mse=val_losses[end], r_midt=r_midt)

# ---------------------------------------------------------------------------
# Loss curve plot
# ---------------------------------------------------------------------------

fig, ax = subplots(figsize=(7, 4))
ax.plot(1:EPOCHS, train_rlosses, label="train r-loss (-mean r)")
ax.plot(1:EPOCHS, val_losses,    label="val MSE", linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Baseline CNN — loss curve")
ax.legend()
fig.tight_layout()
savefig(joinpath(LOG_DIR, "baseline_loss_curve.pdf"))
close(fig)

# ---------------------------------------------------------------------------
# Final evaluation on test set
# ---------------------------------------------------------------------------

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
savefig(joinpath(LOG_DIR, "baseline_pearson_r.pdf"))
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
savefig(joinpath(LOG_DIR, "baseline_correlator_comparison.pdf"))
close(fig)

# ---------------------------------------------------------------------------
# Per-config scatter at mid-t
# ---------------------------------------------------------------------------

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
savefig(joinpath(LOG_DIR, "baseline_scatter_midt.pdf"))
close(fig)

println("\nLogs and plots written to: $LOG_DIR")
