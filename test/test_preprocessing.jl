using Test
using HDF5
using LatticeGPU
using GLARE

@testset "Preprocessing: normalization and data loading" begin

    # GLARE_TEST_GAUGE_H5 = scalar gauge HDF5 produced by build_gauge_dataset
    # GLARE_TEST_CORR_H5  = correlator HDF5 produced by build_corr_dataset
    gauge_h5 = get(ENV, "GLARE_TEST_GAUGE_H5", "")
    corr_h5  = get(ENV, "GLARE_TEST_CORR_H5",  "")

    if isempty(gauge_h5) || isempty(corr_h5) ||
       !isfile(gauge_h5) || !isfile(corr_h5)
        @warn "Skipping preprocessing test: set ENV[\"GLARE_TEST_GAUGE_H5\"] and " *
              "ENV[\"GLARE_TEST_CORR_H5\"] to HDF5 files produced by " *
              "build_gauge_dataset and build_corr_dataset."
        return
    end

    POLARIZATIONS = ["g1-g1", "g2-g2", "g3-g3"]

    # -----------------------------------------------------------------------
    # split_configs: 4-way interleaved, MC order preserved
    # (pass gauge_h5 — both files share the same config id keys)
    # -----------------------------------------------------------------------
    train_ids, val_ids, test_ids, bc_ids =
        split_configs(gauge_h5; train=0.60, val=0.15, test=0.15, bias_corr=0.10)

    all_ids = vcat(train_ids, val_ids, test_ids, bc_ids)
    h5_ids  = h5open(fid -> sort(keys(fid["configs"]), by=x->parse(Int,x)), gauge_h5, "r")

    @test length(all_ids) == length(h5_ids)
    @test Set(all_ids) == Set(h5_ids)
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
    train2, _, _, _ = split_configs(gauge_h5; train=0.60, val=0.15, test=0.15, bias_corr=0.10)
    @test train_ids == train2

    # -----------------------------------------------------------------------
    # compute_normalization (two-database)
    # -----------------------------------------------------------------------
    stats = compute_normalization(gauge_h5, corr_h5, train_ids;
                                  polarizations=POLARIZATIONS)

    @test stats isa NormStats
    @test all(isfinite, stats.feat_mean)
    @test all(isfinite, stats.feat_std)
    @test all(stats.feat_std .> 0)
    @test all(isfinite, stats.corr_mean)
    @test all(isfinite, stats.corr_std)
    @test all(stats.corr_std .> 0)

    # -----------------------------------------------------------------------
    # save / load normalization (save into gauge_h5)
    # -----------------------------------------------------------------------
    save_normalization(gauge_h5, stats)
    stats2 = load_normalization(gauge_h5)

    @test stats2.feat_mean ≈ stats.feat_mean
    @test stats2.feat_std  ≈ stats.feat_std
    @test stats2.corr_mean ≈ stats.corr_mean
    @test stats2.corr_std  ≈ stats.corr_std

    # -----------------------------------------------------------------------
    # load_gauge
    # -----------------------------------------------------------------------
    cid = first(train_ids)

    feat_raw = load_gauge(gauge_h5, cid)
    @test feat_raw isa Array{Float64, 5}   # (iL[1], iL[2], iL[3], iL[4], npls)
    @test ndims(feat_raw) == 5

    feat_norm = load_gauge(gauge_h5, cid; stats=stats)
    @test all(isfinite, feat_norm)
    @test !(feat_norm ≈ feat_raw)

    # field=:matrix requires the matrix database — skip if not available
    matrix_h5 = get(ENV, "GLARE_TEST_MATRIX_H5", "")
    if !isempty(matrix_h5) && isfile(matrix_h5)
        feat_mat = load_gauge(matrix_h5, cid; field=:matrix)
        @test ndims(feat_mat) == 6
        @test size(feat_mat, 1) == 6
    end

    # -----------------------------------------------------------------------
    # load_corr
    # -----------------------------------------------------------------------
    corr_raw = load_corr(corr_h5, cid; polarization="g1-g1")
    @test corr_raw isa Matrix{Float64}

    corr_norm = load_corr(corr_h5, cid; stats=stats, polarization="g1-g1")
    @test all(isfinite, corr_norm)
    @test !(corr_norm ≈ corr_raw)

    # all three polarizations accessible
    for pol in POLARIZATIONS
        c = load_corr(corr_h5, cid; polarization=pol)
        @test c isa Matrix{Float64}
        @test size(c, 1) > 0
    end

    # -----------------------------------------------------------------------
    # load_config (combined)
    # -----------------------------------------------------------------------
    feat, corr = load_config(gauge_h5, corr_h5, cid; polarization="g2-g2")
    @test feat isa Array{Float64, 5}
    @test corr isa Matrix{Float64}

    feat_n, corr_n = load_config(gauge_h5, corr_h5, cid;
                                  stats=stats, polarization="g3-g3")
    @test all(isfinite, feat_n)
    @test all(isfinite, corr_n)

    # -----------------------------------------------------------------------
    # load_split
    # -----------------------------------------------------------------------
    feats, corrs = load_split(gauge_h5, corr_h5, train_ids;
                               stats=stats, polarization="g1-g1")
    @test length(feats) == length(train_ids)
    @test length(corrs) == length(train_ids)
    @test corrs[1] isa Matrix{Float64}

    @info "Preprocessing: $(length(train_ids)) train / $(length(val_ids)) val / " *
          "$(length(test_ids)) test / $(length(bc_ids)) bias_corr configs"

end
