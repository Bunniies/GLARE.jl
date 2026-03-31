using Test
using HDF5
using LatticeGPU
using GLARE

@testset "Preprocessing: normalization and data loading" begin

    ENV["GLARE_TEST_H5"] = "/Users/alessandroconigli/Lattice/data/HVP/LMA/hdf5/A654_all_t_sources/A654_test.hf"
    h5path = get(ENV, "GLARE_TEST_H5", "")

    if isempty(h5path) || !isfile(h5path)
        @warn "Skipping preprocessing test: set ENV[\"GLARE_TEST_H5\"] to a " *
              "dataset HDF5 file produced by build_dataset."
        return
    end

    # ---------------------------------------------------------------------------
    # split_configs: 4-way interleaved, MC order preserved
    # ---------------------------------------------------------------------------
    train_ids, val_ids, test_ids, bc_ids =
        split_configs(h5path; train=0.60, val=0.15, test=0.15, bias_corr=0.10)

    all_ids = vcat(train_ids, val_ids, test_ids, bc_ids)
    h5_ids  = h5open(fid -> sort(keys(fid["configs"]), by=x->parse(Int,x)), h5path, "r")

    @test length(all_ids) == length(h5_ids)
    @test Set(all_ids) == Set(h5_ids)                              # no config lost
    @test isempty(intersect(Set(train_ids), Set(val_ids)))
    @test isempty(intersect(Set(train_ids), Set(test_ids)))
    @test isempty(intersect(Set(train_ids), Set(bc_ids)))
    @test isempty(intersect(Set(val_ids),   Set(test_ids)))
    @test isempty(intersect(Set(val_ids),   Set(bc_ids)))
    @test isempty(intersect(Set(test_ids),  Set(bc_ids)))

    # splits are in MC (numerical) order
    to_int = x -> parse.(Int, x)
    @test to_int(train_ids) == sort(to_int(train_ids))
    @test to_int(val_ids)   == sort(to_int(val_ids))
    @test to_int(test_ids)  == sort(to_int(test_ids))
    @test to_int(bc_ids)    == sort(to_int(bc_ids))

    # deterministic: same fractions → same result
    train2, _, _, _ = split_configs(h5path; train=0.60, val=0.15, test=0.15, bias_corr=0.10)
    @test train_ids == train2

    # ---------------------------------------------------------------------------
    # compute_normalization
    # ---------------------------------------------------------------------------
    stats = compute_normalization(h5path, train_ids)

    @test stats isa NormStats
    @test all(isfinite, stats.feat_mean)
    @test all(isfinite, stats.feat_std)
    @test all(stats.feat_std .> 0)
    @test all(isfinite, stats.corr_mean)
    @test all(isfinite, stats.corr_std)
    @test all(stats.corr_std .> 0)

    # ---------------------------------------------------------------------------
    # save / load normalization
    # ---------------------------------------------------------------------------
    save_normalization(h5path, stats)
    stats2 = load_normalization(h5path)

    @test stats2.feat_mean ≈ stats.feat_mean
    @test stats2.feat_std  ≈ stats.feat_std
    @test stats2.corr_mean ≈ stats.corr_mean
    @test stats2.corr_std  ≈ stats.corr_std

    # ---------------------------------------------------------------------------
    # load_config: raw (no normalization)
    # ---------------------------------------------------------------------------
    cid = first(train_ids)

    feat_raw, corr_raw = load_config(h5path, cid)
    @test feat_raw  isa Array{Float64, 3}
    @test corr_raw  isa Matrix{Float64}

    feat_mat, _ = load_config(h5path, cid; field=:matrix)
    @test ndims(feat_mat) == 4
    @test size(feat_mat, 1) == 6

    feat_both, _ = load_config(h5path, cid; field=:both)
    @test feat_both isa NamedTuple
    @test haskey(feat_both, :scalar)
    @test haskey(feat_both, :matrix)

    # ---------------------------------------------------------------------------
    # load_config: with normalization applied
    # ---------------------------------------------------------------------------
    feat_norm, corr_norm = load_config(h5path, cid; stats=stats)

    # normalized features should have approximately zero mean over training set
    # (exact only if we average all training configs; here just check finite + changed)
    @test all(isfinite, feat_norm)
    @test all(isfinite, corr_norm)
    @test !(feat_norm ≈ feat_raw)
    @test !(corr_norm ≈ corr_raw)

    # ---------------------------------------------------------------------------
    # load_split
    # ---------------------------------------------------------------------------
    feats, corrs = load_split(h5path, train_ids; stats=stats)
    @test length(feats)  == length(train_ids)
    @test length(corrs)  == length(train_ids)
    @test corrs[1] isa Matrix{Float64}

    @info "Preprocessing: $(length(train_ids)) train / $(length(val_ids)) val / " *
          "$(length(test_ids)) test / $(length(bc_ids)) bias_corr configs"

end
