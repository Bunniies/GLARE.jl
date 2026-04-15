using GLARE
using Flux
using HDF5
using JLD2
using Statistics
using Printf

# ---------------------------------------------------------------------------
# Paths — edit before running
# ---------------------------------------------------------------------------

links_h5       = "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_gauge_links_1_200.h5"
corr_h5        = "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_corr.h5"
checkpoint_path = "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/training_lcnn/lcnn_best.jld2"

const device = Flux.gpu_device()

# ---------------------------------------------------------------------------
# Lattice dimensions (from HDF5 metadata)
# ---------------------------------------------------------------------------

Lt, Ls, ndim = HDF5.h5open(links_h5, "r") do fid
    vol = read(fid["metadata"]["vol"])   # lp.iL = (Lx, Ly, Lz, Lt), t at index 4
    gl  = read(fid["configs"][first(keys(fid["configs"]))]["gauge_links"])
    vol[4], vol[1], size(gl, 6)
end

npol          = 3
POLARIZATIONS = ["g1-g1", "g2-g2", "g3-g3"]

# ---------------------------------------------------------------------------
# Reproduce the same split used during training
# ---------------------------------------------------------------------------

println("Splitting configurations...")
train_ids, val_ids, test_ids, bc_ids =
    split_configs(links_h5; train=0.70, val=0.15, test=0.15, bias_corr=0.0)

@printf("Split: %d train / %d val / %d test / %d bc\n",
        length(train_ids), length(val_ids), length(test_ids), length(bc_ids))

# Normalization stats from training split (correlator only — links not normalised)
println("Computing normalization statistics...")
stats = compute_corr_normalization(corr_h5, train_ids; polarizations=POLARIZATIONS)

# ---------------------------------------------------------------------------
# Data helpers  (full volume — no gradient, no crop needed)
# ---------------------------------------------------------------------------

function load_one(cid::String)
    raw = load_links(links_h5, cid)    # ComplexF32[6, Lt, Ls, Ls, Ls, ndim]
    U   = su3_reconstruct(raw)         # ComplexF32[3, 3, Lt, Ls, Ls, Ls, ndim]

    corr2d_phys = Matrix{Float64}(undef, Lt, npol)   # physical (unnormalised)
    h5open(corr_h5, "r") do fid
        for (ipol, pol) in enumerate(POLARIZATIONS)
            co   = read(fid["configs"][cid][pol]["correlator"])   # (Lt, nsrcs)
            cbar = vec(mean(co, dims=2))
            for t in 1:Lt
                corr2d_phys[t, ipol] = cbar[t]
            end
        end
    end
    return U, corr2d_phys
end

# Returns:
#   U_batch   :: ComplexF32[3, 3, Lt, Ls, Ls, Ls, ndim, B]
#   corr_phys :: Float64[Lt, npol, B]   physical units (for bias correction)
function load_batch(ids::Vector{String})
    triples = [load_one(cid) for cid in ids]
    U_batch   = cat([reshape(t[1], 3, 3, Lt, Ls, Ls, Ls, ndim, 1) for t in triples]...; dims=8)
    corr_phys = cat([reshape(t[2], Lt, npol, 1) for t in triples]...; dims=3)
    return U_batch, corr_phys
end

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

println("Loading model from: $checkpoint_path")
model = JLD2.load(checkpoint_path)["model"] |> device
println("Model loaded.")

# ---------------------------------------------------------------------------
# Predict for a set of config ids — returns physical-unit arrays
# ---------------------------------------------------------------------------

# pred_phys :: Float64[Lt, npol, N]  — NN prediction in physical units
# true_phys :: Float64[Lt, npol, N]  — exact correlator in physical units
function predict_all(ids::Vector{String}; batch_size::Int=1)
    pred_list = Vector{Array{Float64,3}}()
    true_list = Vector{Array{Float64,3}}()

    for start in 1:batch_size:length(ids)
        batch_ids = ids[start:min(start + batch_size - 1, end)]
        U_batch, corr_phys = load_batch(batch_ids)

        U_d    = U_batch |> device
        pred_n = Flux.cpu(model(plaquette_matrices(U_d), U_d))   # Float32[Lt, npol, B]

        # Unnormalise: pred_phys[t,p] = pred_norm[t,p] * std[t] + mean[t]
        pred_phys = similar(corr_phys)
        for t in 1:Lt
            pred_phys[t, :, :] .= Float64.(pred_n[t, :, :]) .* stats.corr_std[t] .+ stats.corr_mean[t]
        end

        push!(pred_list, pred_phys)
        push!(true_list, corr_phys)
    end

    return cat(pred_list...; dims=3), cat(true_list...; dims=3)
end

# ---------------------------------------------------------------------------
# Run inference on test and bc splits
# ---------------------------------------------------------------------------

println("\nRunning inference on test set ($(length(test_ids)) configs)...")
pred_test, true_test = predict_all(test_ids)

println("Running inference on bc set ($(length(bc_ids)) configs)...")
pred_bc, true_bc = predict_all(bc_ids)

# ---------------------------------------------------------------------------
# Cost-reduction estimator
# arXiv:2602.21617:
#   C_corrected(t) = <C_approx_test(t)> + (<C_exact_bc(t)> - <C_approx_bc(t)>)
#
# All averages are over configs in the respective split (MC average).
# Each quantity is Float64[Lt, npol] after averaging over the config axis.
# ---------------------------------------------------------------------------

C_approx_test = mean(pred_test; dims=3)[:, :, 1]   # <NN(U)>  on test
C_exact_bc    = mean(true_bc;   dims=3)[:, :, 1]   # <C_exact> on bc
C_approx_bc   = mean(pred_bc;   dims=3)[:, :, 1]   # <NN(U)>  on bc

C_corrected   = C_approx_test .+ (C_exact_bc .- C_approx_bc)
C_exact_test  = mean(true_test; dims=3)[:, :, 1]   # reference (not used in estimator)

# ---------------------------------------------------------------------------
# Pearson r per time slice (in normalised units, for diagnostics)
# ---------------------------------------------------------------------------

r_per_t = pearson_r(Float32.(pred_test), Float32.(true_test))

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

println("\n--- Inference summary ---")
@printf("Test configs : %d\n", length(test_ids))
@printf("BC   configs : %d\n", length(bc_ids))
@printf("r̄         : %.4f\n", mean(r_per_t))
@printf("r̄(mid-t)  : %.4f\n", mean(r_per_t[div(Lt,4):3*div(Lt,4)]))

println("\nt   r(t)       C_exact_test   C_corrected    C_approx_test")
println("-"^70)
for t in 1:Lt
    r_t     = r_per_t[t]
    c_ex    = mean(C_exact_test[t, :])
    c_cor   = mean(C_corrected[t, :])
    c_approx = mean(C_approx_test[t, :])
    @printf("%3d  %+.4f    %+.6e   %+.6e   %+.6e\n", t, r_t, c_ex, c_cor, c_approx)
end

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

out_path = joinpath(dirname(checkpoint_path), "lcnn_inference.jld2")
jldsave(out_path;
    test_ids      = test_ids,
    bc_ids        = bc_ids,
    pred_test     = pred_test,        # Float64[Lt, npol, N_test]  NN predictions (physical)
    true_test     = true_test,        # Float64[Lt, npol, N_test]  exact correlators (physical)
    pred_bc       = pred_bc,          # Float64[Lt, npol, N_bc]
    true_bc       = true_bc,          # Float64[Lt, npol, N_bc]
    C_corrected   = C_corrected,      # Float64[Lt, npol]  bias-corrected estimator
    C_exact_test  = C_exact_test,     # Float64[Lt, npol]  naive average on test (reference)
    C_approx_test = C_approx_test,    # Float64[Lt, npol]
    r_per_t       = r_per_t,          # Float64[Lt]
    corr_mean     = stats.corr_mean,
    corr_std      = stats.corr_std,
)
println("\nResults saved to: $out_path")
