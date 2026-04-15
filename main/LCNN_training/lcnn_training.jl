using GLARE
using Flux
using HDF5
using JLD2
using Statistics
using Printf
using Random
using LinearAlgebra
using PyPlot

# ---------------------------------------------------------------------------
# Device selection — GPU if available, CPU otherwise.
# Requires CUDA.jl (or Metal.jl on Apple Silicon) to be installed separately.
# ---------------------------------------------------------------------------
const device = Flux.gpu_device()

# ---------------------------------------------------------------------------
# Configuration — edit paths and hyperparameters
# ---------------------------------------------------------------------------

links_h5 = "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_gauge_links_1_200.h5"
corr_h5  = "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_corr.h5"

isfile(links_h5) || error("Gauge link HDF5 not found: $links_h5")
isfile(corr_h5)  || error("Correlator HDF5 not found: $corr_h5")

LOG_DIR = get(ENV, "GLARE_LOG_DIR",
              "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/training_lcnn/")
mkpath(LOG_DIR)

CHECKPOINT_PATH = joinpath(LOG_DIR, "lcnn_best.jld2")   # best val-loss checkpoint
FINAL_PATH      = joinpath(LOG_DIR, "lcnn_final.jld2")  # model at last epoch
CONFIG_PATH     = joinpath(LOG_DIR, "lcnn_config.toml") # human-readable run config

# Lattice dimensions — read from HDF5 metadata
Lt, Ls, ndim = HDF5.h5open(links_h5, "r") do fid
    vol = read(fid["metadata"]["vol"])   # Int64[4]: lp.iL = (Lx, Ly, Lz, Lt) — t at index 4
    gl  = read(fid["configs"][first(keys(fid["configs"]))]["gauge_links"])
    vol[4], vol[1], size(gl, 6)          # Lt = vol[4], Ls = vol[1], ndim = last dim
end

npol          = 3
POLARIZATIONS = ["g1-g1", "g2-g2", "g3-g3"]

# Hyperparameters
# C_in = 6: plaquette matrices P_μν(x) as the initial gauge-covariant field W₀.
# Each plaquette transforms as V(x)·P·V†(x) — both gauge indices at the same
# site x — satisfying the L-CNN covariance requirement. Raw links do NOT satisfy
# this (they transform with V†(x+μ̂) on the right, a different site).
C_IN       = ndim * (ndim - 1) ÷ 2   # = 6 for 4D lattice (one per plane)
CHANNELS   = [2, 2]         # L-CB block output channels (conservative for memory)
MLP_HIDDEN = 64
LR           = 3e-3
WEIGHT_DECAY = 1e-4
EPOCHS     = 10
BATCH_SIZE = 1              # small: each config ~ 190 MB reconstructed links
                            # batch=4 ~ 760 MB field tensors before Zygote tape

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

println("Splitting configurations...")
train_ids, val_ids, test_ids, bc_ids =
    split_configs(links_h5; train=0.70, val=0.15, test=0.15, bias_corr=0.0)

@printf("Split: %d train / %d val / %d test / %d bc\n",
        length(train_ids), length(val_ids), length(test_ids), length(bc_ids))

# Gauge links are not normalized — only correlator stats are needed.
println("Computing normalization statistics (correlator only)...")
stats = compute_corr_normalization(corr_h5, train_ids; polarizations=POLARIZATIONS)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
# Data loading — full volume training enabled by gradient checkpointing in LCNN.
# random_spatial_crop and TRAIN_CROP_S kept as fallback (set TRAIN_CROP_S < Ls to re-enable).

TRAIN_CROP_S = Ls   # full volume — gradient checkpointing removes the memory constraint.
                    # set to e.g. 16 to re-enable cropping if checkpointing is not available.
function random_spatial_crop(U::AbstractArray, crop_s::Int)
    # U: (3, 3, Lt, Ls, Ls, Ls, ndim)
    Ls_u = size(U, 4)
    crop_s == Ls_u && return U
    ox = rand(1:(Ls_u - crop_s + 1))
    oy = rand(1:(Ls_u - crop_s + 1))
    oz = rand(1:(Ls_u - crop_s + 1))
    return U[:, :, :, ox:ox+crop_s-1, oy:oy+crop_s-1, oz:oz+crop_s-1, :]
end
# Reconstruct full SU(3) links for one config.
# load_links returns ComplexF32[6, Lt, Ls, Ls, Ls, ndim].
# su3_reconstruct returns ComplexF32[3, 3, Lt, Ls, Ls, Ls, ndim].
function load_one(cid::String; crop_s::Int=Ls)
    raw  = load_links(links_h5, cid)             # (6, Lt, Ls, Ls, Ls, ndim)
    U    = su3_reconstruct(raw)                  # (3, 3, Lt, Ls, Ls, Ls, ndim)
    U    = random_spatial_crop(U, crop_s)

    # Source-averaged normalised correlator: (Lt, npol)
    corr2d = Matrix{Float32}(undef, Lt, npol)
    h5open(corr_h5, "r") do fid
        for (ipol, pol) in enumerate(POLARIZATIONS)
            co   = read(fid["configs"][cid][pol]["correlator"])   # (Lt, nsrcs)
            cbar = vec(mean(co, dims=2))
            for t in 1:Lt
                corr2d[t, ipol] = Float32((cbar[t] - stats.corr_mean[t]) / stats.corr_std[t])
            end
        end
    end
    return U, corr2d
end
Ut, _ = load_one("1") ;
@show eltype(Ut) # should be ComplexF32 not ComplexF64
size(Ut)

# Batch loader: returns
#   U_batch :: ComplexF32[3, 3, Lt, Ls, Ls, Ls, ndim, B]   gauge links (for transport)
#   corr    :: Float32[Lt, npol, B]
# W₀ = plaquette_matrices(U_batch) — computed at call site, C_in = 6.
# crop_s defaults to Ls (full volume). Pass crop_s=TRAIN_CROP_S explicitly if cropping needed.
function load_batch(ids::Vector{String}; crop_s::Int=Ls)
    pairs = [load_one(cid; crop_s=crop_s) for cid in ids]
    cs    = size(pairs[1][1], 4)   # actual spatial size after crop
    U_batch = cat([reshape(p[1], 3, 3, Lt, cs, cs, cs, ndim, 1) for p in pairs]...; dims=8)
    corr    = cat([reshape(p[2], Lt, npol, 1) for p in pairs]...;  dims=3)
    return U_batch, corr
end

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

println("Building L-CNN model...")
model = build_lcnn(;
    Lt         = Lt,
    C_in       = C_IN,
    ndim       = ndim,
    channels   = CHANNELS,
    npol       = npol,
    mlp_hidden = MLP_HIDDEN)

# Move model to device. opt_state must be set up AFTER so that optimizer
# moments live on the same device as the parameters.
# All parameters are already Float32/ComplexF32 from build_lcnn — no f32 cast needed.

# model     = model |> device # it crashes locally with rosetta
opt_state = Flux.setup(Adam(LR, (0.9, 0.999), WEIGHT_DECAY), model)

n_params = sum(length, Flux.params(model))
@printf("L-CNN: C_in=%d  channels=%s  mlp_hidden=%d  ndim=%d\n",
        C_IN, string(CHANNELS), MLP_HIDDEN, ndim)
@printf("Parameters: %d\n", n_params)

# ---------------------------------------------------------------------------
# Write run config to TOML for reproducibility and post-hoc checks
# ---------------------------------------------------------------------------
open(CONFIG_PATH, "w") do io
    println(io, "# L-CNN training run configuration")
    println(io, "# Generated automatically — do not edit by hand\n")

    println(io, "[lattice]")
    @printf(io, "Lt     = %d\n", Lt)
    @printf(io, "Ls     = %d\n", Ls)
    @printf(io, "ndim   = %d\n", ndim)
    @printf(io, "C_in   = %d    # plaquette planes (ndim*(ndim-1)/2)\n", C_IN)
    println(io)

    println(io, "[data]")
    @printf(io, "links_h5     = \"%s\"\n", links_h5)
    @printf(io, "corr_h5      = \"%s\"\n", corr_h5)
    @printf(io, "polarizations = %s\n", string(POLARIZATIONS))
    @printf(io, "n_train      = %d\n", length(train_ids))
    @printf(io, "n_val        = %d\n", length(val_ids))
    @printf(io, "n_test       = %d\n", length(test_ids))
    @printf(io, "n_bc         = %d\n", length(bc_ids))
    @printf(io, "crop_s       = %d    # spatial crop (Ls=%d = full volume, checkpointing enabled)\n", TRAIN_CROP_S, Ls)
    println(io)

    println(io, "[model]")
    @printf(io, "channels   = %s\n", string(CHANNELS))
    @printf(io, "mlp_hidden = %d\n", MLP_HIDDEN)
    @printf(io, "n_params   = %d\n", n_params)
    println(io)

    println(io, "[training]")
    @printf(io, "epochs     = %d\n", EPOCHS)
    @printf(io, "batch_size = %d\n", BATCH_SIZE)
    @printf(io, "lr           = %.2e\n", LR)
    @printf(io, "weight_decay = %.2e\n", WEIGHT_DECAY)
    @printf(io, "optimizer    = \"Adam\"\n")
    @printf(io, "device     = \"%s\"\n", string(device))
    println(io)

    println(io, "[output]")
    @printf(io, "log_dir         = \"%s\"\n", LOG_DIR)
    @printf(io, "checkpoint_best = \"%s\"\n", CHECKPOINT_PATH)
    @printf(io, "checkpoint_final = \"%s\"\n", FINAL_PATH)
    @printf(io, "training_log    = \"%s\"\n", joinpath(LOG_DIR, "lcnn_training_log.csv"))
end
println("Run config written to: $CONFIG_PATH")

##
# ---------------------------------------------------------------------------
# Gradient sanity check
# ---------------------------------------------------------------------------
# Verify gradients flow through all blocks before training.
let
    # Use a real gauge config so norms are physical (SU(3), not random complex).
    _U, _corr = load_batch(train_ids[1:1]; crop_s=TRAIN_CROP_S)
    _U    = _U    |> device
    _corr = _corr |> device

    _, _grads = Flux.withgradient(model) do m
        W = plaquette_matrices(_U)
        for blk in m.blocks
            W = blk(W, _U)
        end
        x = m.pool(W)
        mean(x.^2)   # just check grads flow; skip MLP (different Lt)
    end
    g = _grads[1]
    for (k, blk) in enumerate(g.blocks)
        gconv  = blk === nothing ? nothing : blk.conv
        gbilin = blk === nothing ? nothing : blk.bilin
        ω_norm = (gconv  === nothing || gconv.omega  === nothing) ? NaN : norm(gconv.omega)
        α_norm = (gbilin === nothing || gbilin.α     === nothing) ? NaN : norm(gbilin.α)
        @printf("Block %d  ω grad norm: %.2e   α grad norm: %.2e\n", k, ω_norm, α_norm)
    end
end

##
# ---------------------------------------------------------------------------
# Loss and evaluation
# ---------------------------------------------------------------------------

mse_loss(ŷ, y) = mean((ŷ .- y).^2)

function evaluate(model, ids; batch_size=BATCH_SIZE)
    total_loss = 0.0
    n_batches  = 0
    all_pred   = Array{Float32}(undef, Lt, npol, 0)
    all_true   = Array{Float32}(undef, Lt, npol, 0)

    for start in 1:batch_size:length(ids)
        batch_ids     = ids[start:min(start + batch_size - 1, end)]
        U_batch, corr = load_batch(batch_ids)
        U_d           = U_batch |> device
        pred          = model(plaquette_matrices(U_d), U_d)
        pred_cpu      = Flux.cpu(pred)
        total_loss   += Float64(mse_loss(pred_cpu, corr))
        n_batches    += 1
        all_pred      = cat(all_pred, pred_cpu; dims=3)
        all_true      = cat(all_true, corr;     dims=3)
    end

    r_per_t = pearson_r(all_pred, all_true)
    return total_loss / n_batches, r_per_t, all_pred, all_true
end

# Quick pre-training baseline
# _U0, _c0 = load_batch(train_ids[1:BATCH_SIZE]);
# @printf("Pre-training batch loss: %.5f  (expect ≈ 1.0 for normalised targets)\n",
        # mse_loss(model(_U0, _U0), _c0))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

println("\n--- Training L-CNN ---")
@printf("Epochs: %d  |  Batch: %d  |  LR: %.0e\n", EPOCHS, BATCH_SIZE, LR)
@printf("Channels: %s  |  MLP hidden: %d\n\n", string(CHANNELS), MLP_HIDDEN)

log_path = joinpath(LOG_DIR, "lcnn_training_log.csv")
open(log_path, "w") do io
    println(io, "epoch,train_loss,val_loss,r_mean,r_midt")
end

train_losses  = Float64[]
val_losses    = Float64[]
best_val_loss = Inf
r_midt        = NaN

for epoch in 1:EPOCHS
    perm       = randperm(length(train_ids))
    epoch_loss = 0.0
    n_batches  = 0

    for start in 1:BATCH_SIZE:length(train_ids)
        batch_ids     = train_ids[perm[start:min(start + BATCH_SIZE - 1, end)]]
        U_batch, corr = load_batch(batch_ids; crop_s=TRAIN_CROP_S)
        U_batch       = U_batch |> device
        corr          = corr    |> device

        loss_val, grads = Flux.withgradient(model) do m
            mse_loss(m(plaquette_matrices(U_batch), U_batch), corr)
        end

        Flux.update!(opt_state, model, grads[1])
        epoch_loss += Float64(loss_val)
        n_batches  += 1
    end

    train_loss         = epoch_loss / n_batches
    val_loss, r_per_t, _, _ = evaluate(model, val_ids)
    r_mean = mean(r_per_t)
    r_midt = mean(r_per_t[div(Lt, 4):3*div(Lt, 4)])

    push!(train_losses, train_loss)
    push!(val_losses,   val_loss)

    improved = val_loss < best_val_loss
    if improved
        best_val_loss = val_loss
        jldsave(CHECKPOINT_PATH; model=Flux.cpu(model), epoch=epoch,
                val_loss=val_loss, r_midt=r_midt)
    end

    @printf("Epoch %3d/%d  train=%.5f  val=%.5f  r̄=%.3f  r̄(mid-t)=%.3f%s\n",
            epoch, EPOCHS, train_loss, val_loss, r_mean, r_midt,
            improved ? "  ✓ saved" : "")

    open(log_path, "a") do io
        @printf(io, "%d,%.6f,%.6f,%.6f,%.6f\n",
                epoch, train_loss, val_loss, r_mean, r_midt)
    end
end

# Save the final model regardless of whether it is the best.
jldsave(FINAL_PATH; model=Flux.cpu(model), epoch=EPOCHS,
        val_loss=val_losses[end], r_midt=r_midt)

# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

##

fig, ax = subplots(figsize=(7, 4))
ax.plot(1:EPOCHS, train_losses, label="train")
ax.plot(1:EPOCHS, val_losses,   label="val")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss (normalised)")
ax.set_title("L-CNN — loss curve")
ax.legend()
fig.tight_layout()
savefig(joinpath(LOG_DIR, "lcnn_loss_curve.pdf"))
close(fig)

# ---------------------------------------------------------------------------
# Test set evaluation
# ---------------------------------------------------------------------------

println("\n--- Test set evaluation ---")
test_loss, r_per_t, test_pred, test_true = evaluate(model, test_ids)
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
ax.set_title("L-CNN — Pearson r per time slice (test set)")
ax.set_xlim(1, Lt)
fig.tight_layout()
savefig(joinpath(LOG_DIR, "lcnn_pearson_r.pdf"))
close(fig)

# ---------------------------------------------------------------------------
# Predicted vs exact correlator (test set, config average)
# ---------------------------------------------------------------------------

corr_mean = Float32.(stats.corr_mean)
corr_std  = Float32.(stats.corr_std)

pred_phys = test_pred .* corr_std .+ corr_mean   # denormalise: (Lt, npol, N_test)
true_phys = test_true .* corr_std .+ corr_mean

pred_mean = mean(pred_phys, dims=3)[:, :, 1]
true_mean = mean(true_phys, dims=3)[:, :, 1]

fig, axes = subplots(1, npol; figsize=(5*npol, 4), sharey=false)
for (ipol, pol) in enumerate(POLARIZATIONS)
    ax = axes[ipol]
    ax.semilogy(1:Lt, abs.(true_mean[:, ipol]), label="exact",   linewidth=1.5)
    ax.semilogy(1:Lt, abs.(pred_mean[:, ipol]), label="NN pred", linewidth=1.5, linestyle="--")
    ax.set_xlabel("t")
    ax.set_title(pol)
    ax.legend(fontsize=8)
end
axes[1].set_ylabel("C(t)")
fig.suptitle("L-CNN — predicted vs exact correlator (test set, config average)")
fig.tight_layout()
savefig(joinpath(LOG_DIR, "lcnn_correlator_comparison.pdf"))
close(fig)

# ---------------------------------------------------------------------------
# Per-config scatter at mid-t
# ---------------------------------------------------------------------------

t_mid = div(Lt, 2)
fig, axes = subplots(1, npol; figsize=(5*npol, 4))
for (ipol, pol) in enumerate(POLARIZATIONS)
    ax = axes[ipol]
    x  = vec(true_phys[t_mid, ipol, :])
    y  = vec(pred_phys[t_mid, ipol, :])
    ax.scatter(x, y, s=10, alpha=0.6)
    lo, hi = min(minimum(x), minimum(y)), max(maximum(x), maximum(y))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax.set_xlabel("C_exact(t=$t_mid)")
    ax.set_ylabel("C_pred(t=$t_mid)")
    ax.set_title("$(pol)  r=$(round(r_per_t[t_mid], digits=3))")
end
fig.suptitle("L-CNN — scatter at t=$t_mid (test set)")
fig.tight_layout()
savefig(joinpath(LOG_DIR, "lcnn_scatter_midt.pdf"))
close(fig)

println("\nLogs and plots written to: $LOG_DIR")
