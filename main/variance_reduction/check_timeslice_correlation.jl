using HDF5
using Statistics
using Printf
using PyPlot

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

corr_h5  = get(ENV, "GLARE_CORR_H5",
    "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_corr.h5")
# PLOT_DIR = get(ENV, "GLARE_PLOT_DIR",
    # "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/variance_reduction/")

isfile(corr_h5) || error("Correlator HDF5 not found: $corr_h5")
# mkpath(PLOT_DIR)

POLARIZATIONS = ["g1-g1", "g2-g2", "g3-g3"]
npol          = length(POLARIZATIONS)

# Physical time range: periodic box with T=48 → meaningful range is t=1..T/2=24.
# Early (precise): t = 1..T_EARLY_MAX.  Late (noisy target): t = T_EARLY_MAX+1..T_HALF.
# Also note: C(t) = C(T-t) by periodicity, so t > T/2 is redundant — but the values
# at t ∈ [T-T_EARLY_MAX..T-1] mirror the early region and are equally precise;
# we include them as extra input channels in the symmetrized correlator below.
T_EARLY_MAX = 10    # early / precise region:  t = 1..T_EARLY_MAX
                    # late  / noisy   region:  t = T_EARLY_MAX+1..T_HALF

# Fixed t_early values to probe in the line plots
T_EARLY_PROBE = [1, 3, 5, 7, 10]

# ---------------------------------------------------------------------------
# Load all source-averaged correlators
# C_pol[ipol] :: Matrix{Float64}(Lt, N) — one column per config
# ---------------------------------------------------------------------------

println("Loading correlators from: $corr_h5")

config_ids = String[]
Lt         = 0

h5open(corr_h5, "r") do fid
    global config_ids = sort(collect(keys(fid["configs"])), by=x -> parse(Int, x))
    raw      = read(fid["configs"][config_ids[1]][POLARIZATIONS[1]]["correlator"])
    global Lt = size(raw, 1)
end

T_HALF = Lt ÷ 2   # = 24 for A654
N      = length(config_ids)
@printf("Configs: %d   Lt: %d   T_HALF: %d\n", N, Lt, T_HALF)
@printf("Early region: t = 1..%d   Late region: t = %d..%d\n\n",
        T_EARLY_MAX, T_EARLY_MAX+1, T_HALF)

C_pol = [Matrix{Float64}(undef, Lt, N) for _ in 1:npol]

h5open(corr_h5, "r") do fid
    for (icfg, cid) in enumerate(config_ids)
        for (ipol, pol) in enumerate(POLARIZATIONS)
            raw = read(fid["configs"][cid][pol]["correlator"])   # Float64[Lt, nsrcs]
            C_pol[ipol][:, icfg] = vec(mean(raw, dims=2))
        end
    end
end
println("Loaded.")

# ---------------------------------------------------------------------------
# Symmetrized correlator: C_sym(t) = (C(t) + C(T-t)) / 2  for t = 1..T_HALF
# Uses both halves of the periodic box — halves the variance at each t.
# Index convention: C_sym_pol[ipol][t, icfg] for t = 1..T_HALF.
# Note: t=T_HALF (=24) maps to itself since C(24)=C(48-24)=C(24) is unique.
# ---------------------------------------------------------------------------

C_sym_pol = [
    (C_pol[ipol][1:T_HALF, :] .+ C_pol[ipol][[1; Lt-1:-1:T_HALF+1], :]) ./ 2
    for ipol in 1:npol
]
# Index check: for t=1..T_HALF, the mirror is T-t+1? Actually for 1-indexed arrays:
# C(t) is at row t.  C(T-t) for t=1 is C(47) = row 47.  In Julia: row = Lt - t + 1.
# Re-do correctly:
C_sym_pol = [
    begin
        Cf   = C_pol[ipol][1:T_HALF, :]                        # rows 1..24
        mirror_idx = [t == T_HALF ? T_HALF : Lt - t + 1 for t in 1:T_HALF]
        Cbk  = C_pol[ipol][mirror_idx, :]                      # rows 48,47,...,25,24
        (Cf .+ Cbk) ./ 2
    end
    for ipol in 1:npol
]

# ---------------------------------------------------------------------------
# Pearson correlation matrices on the physical range [1..T_HALF]
# R[t1, t2] = Pearson r of C(t1) vs C(t2) across all N configs
# ---------------------------------------------------------------------------

# Raw correlator (first T_HALF rows)
R_pol     = [cor(C_pol[ipol][1:T_HALF, :]') for ipol in 1:npol]   # each: (T_HALF, T_HALF)
R_avg     = mean(R_pol)

# Symmetrized correlator
R_sym_pol = [cor(C_sym_pol[ipol]') for ipol in 1:npol]
R_sym_avg = mean(R_sym_pol)

# ---------------------------------------------------------------------------
# Print key statistics
# ---------------------------------------------------------------------------

println("--- r(t_early, t_late) pol-averaged, RAW ---")
@printf("  %-8s", "")
for t2 in T_EARLY_MAX+1:T_HALF
    @printf("  t=%2d", t2)
end
println()
for t1 in T_EARLY_PROBE
    @printf("  t_e=%2d  ", t1)
    for t2 in T_EARLY_MAX+1:T_HALF
        @printf("  %+.2f", R_avg[t1, t2])
    end
    println()
end

println("\n--- r(t_early, t_late) pol-averaged, SYMMETRIZED ---")
@printf("  %-8s", "")
for t2 in T_EARLY_MAX+1:T_HALF
    @printf("  t=%2d", t2)
end
println()
for t1 in T_EARLY_PROBE
    @printf("  t_e=%2d  ", t1)
    for t2 in T_EARLY_MAX+1:T_HALF
        @printf("  %+.2f", R_sym_avg[t1, t2])
    end
    println()
end

# ---------------------------------------------------------------------------
# Plot 1: (T_HALF × T_HALF) heatmap with early/late boundary marked
# Left panel: raw C(t).  Right panel: symmetrized C_sym(t).
# ---------------------------------------------------------------------------

ts  = 1:T_HALF
sep = T_EARLY_MAX + 0.5   # boundary between early and late regions

fig, axes = subplots(1, 2; figsize=(11, 5))

for (ax, R, title) in zip(axes,
                           [R_avg, R_sym_avg],
                           ["Raw C(t)", "Symmetrized (C(t)+C(T-t))/2"])
    im = ax.imshow(R; origin="lower", vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto",
                   extent=[0.5, T_HALF+0.5, 0.5, T_HALF+0.5])
    # Dashed lines marking early/late boundary
    ax.axvline(sep; color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(sep; color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("t₂")
    ax.set_ylabel("t₁")
    ax.set_title(title; fontsize=10)
    ax.set_xticks(vcat(1:5:T_HALF))
    ax.set_yticks(vcat(1:5:T_HALF))
    fig.colorbar(im; ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    # Annotate regions
    ax.text(T_EARLY_MAX/2, T_HALF-1, "early"; ha="center", fontsize=8, color="black")
    ax.text(T_HALF - (T_HALF - T_EARLY_MAX)/2, T_HALF-1, "late"; ha="center", fontsize=8, color="black")
end

fig.suptitle("Pearson r(t₁,t₂), t ∈ [1,$(T_HALF)] — A654 ($(N) configs)", fontsize=11)
fig.tight_layout()
display(fig)
# savefig(joinpath(PLOT_DIR, "corr_matrix_heatmap.pdf"))
close(fig)
println("\nSaved: corr_matrix_heatmap.pdf")
##
# ---------------------------------------------------------------------------
# Plot 2: r(t_early, t_late) vs t_late for fixed t_early — raw vs symmetrized
# ---------------------------------------------------------------------------

late_ts = T_EARLY_MAX+1:T_HALF

fig, axes = subplots(1, 2; figsize=(12, 4), sharey=true)

for (ax, R, title) in zip(axes, [R_avg, R_sym_avg], ["Raw", "Symmetrized"])
    for t0 in T_EARLY_PROBE
        ax.plot(late_ts, R[t0, late_ts]; label="t_early = $t0", linewidth=1.4)
    end
    ax.axhline(0; color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("t_late")
    ax.set_ylabel("Pearson r")
    ax.set_title("$(title) — r(t_early, t_late)"; fontsize=10)
    ax.set_xlim(T_EARLY_MAX+1, T_HALF)
    ax.legend(fontsize=8)
end

fig.suptitle("Pearson r — pol-averaged — A654 ($(N) configs)", fontsize=11)
fig.tight_layout()
savefig(joinpath(PLOT_DIR, "corr_vs_tlate.pdf"))
close(fig)
println("Saved: corr_vs_tlate.pdf")

# ---------------------------------------------------------------------------
# Plot 3: r(t_early, t_late=T_HALF) vs t_early — how far does the correlation
# reach? Shows the best input timeslices for predicting the mid-T correlator.
# ---------------------------------------------------------------------------

fig, ax = subplots(figsize=(7, 4))
for (R, label, ls) in [(R_avg, "raw", "-"), (R_sym_avg, "symmetrized", "--")]
    ax.plot(ts, R[:, T_HALF]; label=label, linewidth=1.4, linestyle=ls)
end
ax.axvline(T_EARLY_MAX + 0.5; color="gray", linewidth=0.8, linestyle=":", label="early/late boundary")
ax.axhline(0; color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("t_early")
ax.set_ylabel("Pearson r")
ax.set_title("r(t_early, t_late=T/2=$(T_HALF)) — A654"; fontsize=10)
ax.set_xlim(1, T_HALF)
ax.legend(fontsize=8)
fig.tight_layout()
savefig(joinpath(PLOT_DIR, "corr_vs_tearly_at_thalf.pdf"))
close(fig)
println("Saved: corr_vs_tearly_at_thalf.pdf")

println("\nAll plots written to: $PLOT_DIR")
