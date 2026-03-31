using HDF5
using Statistics
using PyPlot
using GLARE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

h5path  = get(ENV, "GLARE_TEST_H5",
              "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_test.hf")
outdir  = get(ENV, "GLARE_PLOT_DIR", joinpath(dirname(h5path), "plots"))
mkpath(outdir)

isfile(h5path) || error("HDF5 file not found: $h5path")

PLANE_LABELS = ["(4,3)", "(4,2)", "(4,1)", "(3,2)", "(3,1)", "(2,1)"]

# ---------------------------------------------------------------------------
# Split and compute normalization (train only)
# ---------------------------------------------------------------------------

train_ids, val_ids, test_ids, bc_ids =
    split_configs(h5path; train=0.60, val=0.15, test=0.15, bias_corr=0.10)

println("Split: $(length(train_ids)) train / $(length(val_ids)) val / ",
        "$(length(test_ids)) test / $(length(bc_ids)) bias_corr")

stats = compute_normalization(h5path, train_ids)

# ---------------------------------------------------------------------------
# 1. Feature normalization: mean and std per plane (should be ~0 and ~1 on train)
# ---------------------------------------------------------------------------

# Accumulate per-plane values across train / val / test after normalization
function collect_feat_stats(h5path, ids, stats)
    # Returns (npls,) vectors of per-plane means and stds
    sums  = zeros(Float64, length(stats.feat_mean))
    sum2s = zeros(Float64, length(stats.feat_mean))
    ns    = zeros(Int, length(stats.feat_mean))
    for cid in ids
        feat, _ = load_config(h5path, cid; stats=stats, field=:scalar)
        for ipl in axes(feat, 2)
            v = @view feat[:, ipl, :]
            sums[ipl]  += sum(v)
            sum2s[ipl] += sum(v .^ 2)
            ns[ipl]    += length(v)
        end
    end
    means = sums  ./ ns
    stds  = sqrt.(max.(sum2s ./ ns .- means .^ 2, 0.0))
    return means, stds
end

tr_means, tr_stds = collect_feat_stats(h5path, train_ids, stats)
va_means, va_stds = collect_feat_stats(h5path, val_ids,   stats)
te_means, te_stds = collect_feat_stats(h5path, test_ids,  stats)

npls = length(tr_means)
x    = 1:npls

fig, axs = subplots(1, 2, figsize=(11, 4))
fig.suptitle("Feature normalization check (per plane)", fontsize=13)

ax = axs[1]
ax.axhline(0.0, color="k", lw=0.8, ls="--")
ax.errorbar(x .- 0.1, tr_means, fmt="o", color="steelblue",  label="train", capsize=3)
ax.errorbar(x,        va_means, fmt="s", color="darkorange", label="val",   capsize=3)
ax.errorbar(x .+ 0.1, te_means, fmt="^", color="seagreen",   label="test",  capsize=3)
ax.set_xticks(collect(x)); ax.set_xticklabels(PLANE_LABELS[1:npls], rotation=30)
ax.set_xlabel("Plane (μ,ν)"); ax.set_ylabel("Mean of normalised features")
ax.set_title("Mean ≈ 0"); ax.legend()

ax = axs[2]
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.errorbar(x .- 0.1, tr_stds, fmt="o", color="steelblue",  label="train", capsize=3)
ax.errorbar(x,        va_stds, fmt="s", color="darkorange", label="val",   capsize=3)
ax.errorbar(x .+ 0.1, te_stds, fmt="^", color="seagreen",   label="test",  capsize=3)
ax.set_xticks(collect(x)); ax.set_xticklabels(PLANE_LABELS[1:npls], rotation=30)
ax.set_xlabel("Plane (μ,ν)"); ax.set_ylabel("Std of normalised features")
ax.set_title("Std ≈ 1"); ax.legend()

tight_layout()
savefig(joinpath(outdir, "feat_normalization.pdf"))
println("Saved feat_normalization.pdf")
close(fig)

# ---------------------------------------------------------------------------
# 2. Raw plaquette distributions: histogram per plane (before normalization)
# ---------------------------------------------------------------------------

fig, axs = subplots(2, 3, figsize=(13, 8))
fig.suptitle("Raw plaq_scalar distributions (train set)", fontsize=13)
axs_flat = axs[:]

for ipl in 1:npls
    vals = Float64[]
    for cid in train_ids
        feat, _ = load_config(h5path, cid; field=:scalar)
        append!(vals, vec(feat[:, ipl, :]))
    end
    ax = axs_flat[ipl]
    ax.hist(vals, bins=80, color="steelblue", alpha=0.75, density=true)
    ax.axvline(stats.feat_mean[ipl], color="red",   lw=1.5, label="mean")
    ax.axvline(stats.feat_mean[ipl] + stats.feat_std[ipl], color="orange", lw=1, ls="--", label="±std")
    ax.axvline(stats.feat_mean[ipl] - stats.feat_std[ipl], color="orange", lw=1, ls="--")
    ax.set_title("Plane $(PLANE_LABELS[ipl])")
    ax.set_xlabel("Re(Tr P)"); ax.set_ylabel("density")
    ipl == 1 && ax.legend(fontsize=8)
end

tight_layout()
savefig(joinpath(outdir, "feat_distributions.pdf"))
println("Saved feat_distributions.pdf")
close(fig)

# ---------------------------------------------------------------------------
# 3. Correlator normalization: mean and std per time slice
# ---------------------------------------------------------------------------

function collect_corr_stats(h5path, ids, stats)
    T     = length(stats.corr_mean)
    sums  = zeros(Float64, T)
    sum2s = zeros(Float64, T)
    n     = 0
    for cid in ids
        _, corr = load_config(h5path, cid; stats=stats)
        for t in 1:T
            v = @view corr[t, :]
            sums[t]  += sum(v)
            sum2s[t] += sum(v .^ 2)
        end
        n += size(corr, 2)
    end
    ncfg   = length(ids)
    means  = sums  ./ (n * ncfg)
    stds   = sqrt.(max.(sum2s ./ (n * ncfg) .- means .^ 2, 0.0))
    return means, stds
end

T = length(stats.corr_mean)
t = 1:T

tr_cm, tr_cs = collect_corr_stats(h5path, train_ids, stats)
va_cm, va_cs = collect_corr_stats(h5path, val_ids,   stats)
te_cm, te_cs = collect_corr_stats(h5path, test_ids,  stats)

fig, axs = subplots(1, 2, figsize=(11, 4))
fig.suptitle("Correlator normalization check (per time slice)", fontsize=13)

ax = axs[1]
ax.axhline(0.0, color="k", lw=0.8, ls="--")
ax.plot(t, tr_cm, "o-", color="steelblue",  ms=3, label="train")
ax.plot(t, va_cm, "s-", color="darkorange", ms=3, label="val")
ax.plot(t, te_cm, "^-", color="seagreen",   ms=3, label="test")
ax.set_xlabel("t"); ax.set_ylabel("Mean of normalised C(t)")
ax.set_title("Mean ≈ 0"); ax.legend()

ax = axs[2]
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.plot(t, tr_cs, "o-", color="steelblue",  ms=3, label="train")
ax.plot(t, va_cs, "s-", color="darkorange", ms=3, label="val")
ax.plot(t, te_cs, "^-", color="seagreen",   ms=3, label="test")
ax.set_xlabel("t"); ax.set_ylabel("Std of normalised C(t)")
ax.set_title("Std ≈ 1"); ax.legend()

tight_layout()
savefig(joinpath(outdir, "corr_normalization.pdf"))
println("Saved corr_normalization.pdf")
close(fig)

# ---------------------------------------------------------------------------
# 4. Raw correlator: mean ± std across configs (train, src-averaged)
# ---------------------------------------------------------------------------

corr_all = []
for cid in train_ids
    _, corr = load_config(h5path, cid)
    push!(corr_all, mean(corr, dims=2)[:])   # average over sources → Vector{T}
end
corr_mat  = hcat(corr_all...)'               # [ncfg, T]
corr_mean_t = vec(mean(corr_mat, dims=1))
corr_std_t  = vec(std(corr_mat,  dims=1))

fig, ax = subplots(figsize=(8, 4))
ax.errorbar(t, corr_mean_t, yerr=corr_std_t, fmt="o-", color="steelblue",
            ms=3, capsize=2, label="mean ± std over train configs")
ax.set_xlabel("t"); ax.set_ylabel("C(t) [src-averaged]")
ax.set_title("Raw correlator: ensemble mean ± fluctuation")
ax.set_yscale("log")
ax.legend()
tight_layout()
savefig(joinpath(outdir, "raw_correlator.pdf"))
println("Saved raw_correlator.pdf")
close(fig)

# ---------------------------------------------------------------------------
# 5. Input–output correlation: Pearson r between spatial-mean plaquette
#    per plane and C(t) per time slice — heatmap [npls × T]
# ---------------------------------------------------------------------------
# NOTE: Pearson r with N configs has N-2 degrees of freedom.
# With N < ~20 the estimate is essentially noise (r ≈ ±1 by chance).
# Results below are only interpretable for large ensembles.
if length(train_ids) < 20
    @warn "Only $(length(train_ids)) training configs — Pearson r is unreliable. " *
          "Run with a larger HDF5 dataset for meaningful correlation estimates."
end

# Build flat arrays: one value per config
# spatial-mean plaquette per plane: mean over all sites (b,r)
feat_plane = zeros(Float64, length(train_ids), npls)   # [ncfg, npls]
corr_t     = zeros(Float64, length(train_ids), T)      # [ncfg, T]

for (i, cid) in enumerate(train_ids)
    feat, corr = load_config(h5path, cid)
    for ipl in 1:npls
        feat_plane[i, ipl] = mean(feat[:, ipl, :])
    end
    corr_t[i, :] = mean(corr, dims=2)[:]   # src-averaged
end

# Pearson r matrix [npls × T]
pearson = zeros(Float64, npls, T)
for ipl in 1:npls, tt in 1:T
    x = feat_plane[:, ipl]
    y = corr_t[:, tt]
    pearson[ipl, tt] = cor(x, y)
end

fig, ax = subplots(figsize=(12, 4))
im = ax.imshow(pearson, aspect="auto", cmap="RdBu_r",
               vmin=-1, vmax=1,
               extent=[0.5, T+0.5, npls+0.5, 0.5])
colorbar(im, ax=ax, label="Pearson r")
ax.set_yticks(collect(1:npls))
ax.set_yticklabels(PLANE_LABELS[1:npls])
ax.set_xlabel("t"); ax.set_ylabel("Plane (μ,ν)")
ax.set_title("Pearson correlation: spatial-mean Re(Tr P_μν) vs C(t)  [train set]")
tight_layout()
savefig(joinpath(outdir, "pearson_heatmap.pdf"))
println("Saved pearson_heatmap.pdf")
close(fig)

# ---------------------------------------------------------------------------
# 6. Scatter plot: strongest correlated (plane, t) pair
# ---------------------------------------------------------------------------

idx = argmax(abs.(pearson))
best_ipl, best_t = idx[1], idx[2]
r_best = pearson[best_ipl, best_t]

fig, ax = subplots(figsize=(5, 5))
ax.scatter(feat_plane[:, best_ipl], corr_t[:, best_t],
           s=20, alpha=0.7, color="steelblue")
ax.set_xlabel("⟨Re(Tr P)⟩ plane $(PLANE_LABELS[best_ipl])")
ax.set_ylabel("⟨C(t=$(best_t))⟩ over sources")
ax.set_title("Strongest correlation: r = $(round(r_best, digits=3))")
tight_layout()
savefig(joinpath(outdir, "scatter_best.pdf"))
println("Saved scatter_best.pdf")
close(fig)

# ---------------------------------------------------------------------------
# 7. Pearson r profile: max over planes, as a function of t
# ---------------------------------------------------------------------------

max_r_per_t    = vec(maximum(abs.(pearson), dims=1))
argmax_per_t   = [argmax(abs.(pearson[:, tt])) for tt in 1:T]

fig, ax = subplots(figsize=(8, 4))
ax.plot(t, max_r_per_t, "o-", color="steelblue", ms=4)
ax.set_xlabel("t")
ax.set_ylabel("|r|  (max over planes)")
ax.set_title("Strongest input–output Pearson correlation per time slice")
ax.set_ylim(0, 1)
tight_layout()
savefig(joinpath(outdir, "pearson_profile.pdf"))
println("Saved pearson_profile.pdf")
close(fig)

println("\nAll plots saved to: $outdir")
println("Best (plane, t) pair: plane=$(PLANE_LABELS[best_ipl]), t=$best_t, r=$(round(r_best, digits=4))")
