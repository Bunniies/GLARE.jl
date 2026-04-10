using Test
using HDF5
using LinearAlgebra
using LatticeGPU
using NNlib
using GLARE
using Statistics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _random_su3(T=ComplexF64)
    A = randn(T, 3, 3)
    Q, _ = qr(A)
    U    = Matrix(Q)
    U  ./= det(U)^(1/3)
    return U
end

function _pack2rows(U::Matrix{<:Complex})
    return [U[1,1], U[1,2], U[1,3], U[2,1], U[2,2], U[2,3]]
end

# ---------------------------------------------------------------------------
# Lattice parameters (CLS A654)
# ---------------------------------------------------------------------------

const VOL   = (24, 24, 24, 48)   # (Lx, Ly, Lz, Lt): iL[4]=48=Lt, iL[1..3]=24=Ls
const SVOL  = (4, 4, 4, 8)
const LP    = SpaceParm{4}(VOL, SVOL, BC_PERIODIC, (0,0,0,0,0,0))

# ---------------------------------------------------------------------------

@testset "GLARE Phase 2" begin

    # -----------------------------------------------------------------------
    # build_gauge_link_dataset
    # -----------------------------------------------------------------------

    @testset "build_gauge_link_dataset" begin
        ENV["GLARE_TEST_CONF"] = "/Users/alessandroconigli/Lattice/data/cls/"

        conf_file     = get(ENV, "GLARE_TEST_CONF", "")
        ensemble_path = isempty(conf_file) ? "" : dirname(conf_file)

        if isempty(ensemble_path) || !isdir(ensemble_path)
            @warn "Skipping build_gauge_link_dataset: set ENV[\"GLARE_TEST_CONF\"] " *
                  "(path to a single gauge config file)."
            @test true broken=true
        else
            links_h5 = tempname() * "_gauge_links.h5"

            build_gauge_link_dataset(ensemble_path, LP, links_h5;
                                     config_range=1:1, verbose=true)
            @test isfile(links_h5)

            h5open(links_h5, "r") do fid
                @test haskey(fid, "metadata")
                @test haskey(fid, "configs")

                meta = fid["metadata"]
                @test read(meta["vol"])  == collect(Int64, LP.iL)
                @test read(meta["svol"]) == collect(Int64, LP.blk)

                cfg_ids = keys(fid["configs"])
                @test length(cfg_ids) == 1

                cid = first(cfg_ids)
                grp = fid["configs"][cid]

                @test  haskey(grp, "gauge_links")
                @test !haskey(grp, "plaq_scalar")
                @test !haskey(grp, "plaq_matrix")

                gl = read(grp["gauge_links"])
                @test gl isa Array{ComplexF32, 6}
                # shape: (6, Lt, Ls, Ls, Ls, ndim=4) = (6, iL[4], iL[1], iL[2], iL[3], ndim)
                @test size(gl) == (6, LP.iL[4], LP.iL[1], LP.iL[2], LP.iL[3], 4)

                # Reconstruct full SU(3) for a sample and verify unitarity + det=1
                # (atol=1e-6: Float32 precision after cross-product reconstruction)
                gl_flat = reshape(ComplexF64.(gl), 6, :)   # upcast for numerical check
                Ms      = su3_reconstruct(gl_flat)         # (3, 3, nvol*ndim)
                nsample = min(100, size(Ms, 3))
                for k in 1:nsample
                    M = Ms[:, :, k]
                    @test M * M' ≈ I(3)  atol=1e-6
                    @test det(M) ≈ 1.0   atol=1e-6
                end

                @info "gauge_links shape=$(size(gl)), SU(3) check on $(nsample) links passed"
            end

            rm(links_h5)
        end
    end

    # -----------------------------------------------------------------------
    # merge_dataset for gauge links
    # -----------------------------------------------------------------------

    @testset "merge_dataset (gauge links)" begin

        conf_file     = get(ENV, "GLARE_TEST_CONF", "")
        ensemble_path = isempty(conf_file) ? "" : dirname(conf_file)

        if isempty(ensemble_path) || !isdir(ensemble_path)
            @warn "Skipping merge_dataset (links): set ENV[\"GLARE_TEST_CONF\"]."
            @test true broken=true
        else
            shard1 = tempname() * "_shard1.h5"
            shard2 = tempname() * "_shard2.h5"
            merged = tempname() * "_merged.h5"

            gauge_files = filter(f -> !isdir(joinpath(ensemble_path, f)) &&
                                      occursin(r"n\d+$", f),
                                 readdir(ensemble_path))
            if length(gauge_files) < 2
                @warn "Fewer than 2 configs available — skipping merge test."
                @test true broken=true
            else
                build_gauge_link_dataset(ensemble_path, LP, shard1;
                                         config_range=1:1, verbose=false)
                build_gauge_link_dataset(ensemble_path, LP, shard2;
                                         config_range=2:2, verbose=false)

                merge_dataset([shard1, shard2], merged; verbose=false)
                @test isfile(merged)

                h5open(merged, "r") do fid
                    @test length(keys(fid["configs"])) == 2
                    for cid in keys(fid["configs"])
                        gl = read(fid["configs"][cid]["gauge_links"])
                        @test size(gl, 1) == 6    # row-storage dim
                        @test size(gl, 6) == 4    # ndim
                    end
                end

                rm(shard1); rm(shard2); rm(merged)
            end
        end
    end

    # -----------------------------------------------------------------------
    # On-the-fly plaquette from stored links (pure algebraic, no ENV needed)
    # -----------------------------------------------------------------------

    @testset "plaquette from links (on-the-fly, algebraic)" begin

        # Pack 4 random SU(3) links into (6, ndim) storage, reconstruct,
        # compute all 6 plaquettes P_μν = U_μ U_ν U_μ† U_ν† and verify SU(3).
        ndim = 4
        Us   = [_random_su3() for _ in 1:ndim]

        x = zeros(ComplexF64, 6, ndim)
        for iμ in 1:ndim
            x[:, iμ] = _pack2rows(Us[iμ])
        end

        Ms = su3_reconstruct(reshape(x, 6, 1, ndim))   # (3, 3, 1, ndim)

        # Round-trip fidelity
        for iμ in 1:ndim
            @test Ms[:, :, 1, iμ] ≈ Us[iμ]  atol=1e-12
        end

        # All 6 plaquettes must be SU(3)
        planes = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
        for (μ, ν) in planes
            P = Ms[:,:,1,μ] * Ms[:,:,1,ν] * Ms[:,:,1,μ]' * Ms[:,:,1,ν]'
            @test P * P' ≈ I(3)  atol=1e-11
            @test det(P) ≈ 1.0   atol=1e-11
        end
    end

    # -----------------------------------------------------------------------
    # Phase 2 stubs — not yet implemented
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # ScalarGate
    # -----------------------------------------------------------------------

    @testset "ScalarGate (σ(Re(Tr Φ)) * Φ)" begin

        Lt, Ls, C, B = 4, 3, 2, 5
        Φ = randn(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, C, B)
        gate = ScalarGate()
        out  = gate(Φ)

        # Shape preserved
        @test size(out) == size(Φ)

        # Verify pointwise: out[i,j,t,x,y,z,c,b] = σ(Re(Tr Φ[t,x,y,z,c,b])) * Φ[i,j,...]
        for t in 1:Lt, c in 1:C, b in 1:B
            tr_val  = real(Φ[1,1,t,1,1,1,c,b] + Φ[2,2,t,1,1,1,c,b] + Φ[3,3,t,1,1,1,c,b])
            g       = sigmoid(tr_val)
            @test out[:, :, t, 1, 1, 1, c, b] ≈ g .* Φ[:, :, t, 1, 1, 1, c, b]  atol=1e-12
        end

        # Gauge covariance: Φ → V Φ V† leaves Re(Tr Φ) invariant → same gate value
        V  = _random_su3()
        Φ2 = similar(Φ)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, c in 1:C, b in 1:B
            Φ2[:, :, t, x, y, z, c, b] = V * Φ[:, :, t, x, y, z, c, b] * V'
        end
        out2 = gate(Φ2)

        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, c in 1:C, b in 1:B
            expected = V * out[:, :, t, x, y, z, c, b] * V'
            @test out2[:, :, t, x, y, z, c, b] ≈ expected  atol=1e-12
        end
    end

    # -----------------------------------------------------------------------
    # TracePool
    # -----------------------------------------------------------------------

    @testset "TracePool (Re(Tr Φ) → spatial mean)" begin

        Lt, Ls, C, B = 4, 3, 2, 5
        Φ   = randn(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, C, B)
        out = TracePool()(Φ)

        # Output shape
        @test size(out) == (Lt, C, B)
        @test eltype(out) <: Real

        # Verify manually: mean of Re(Tr Φ) over spatial volume
        for t in 1:Lt, c in 1:C, b in 1:B
            tr_vals = [real(Φ[1,1,t,x,y,z,c,b] + Φ[2,2,t,x,y,z,c,b] + Φ[3,3,t,x,y,z,c,b])
                       for x in 1:Ls, y in 1:Ls, z in 1:Ls]
            @test out[t, c, b] ≈ mean(tr_vals)  atol=1e-12
        end

        # Gauge invariance: Re(Tr Φ) is invariant under Φ → V Φ V†
        V  = _random_su3()
        Φ2 = similar(Φ)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, c in 1:C, b in 1:B
            Φ2[:, :, t, x, y, z, c, b] = V * Φ[:, :, t, x, y, z, c, b] * V'
        end
        @test TracePool()(Φ2) ≈ out  atol=1e-12
    end

    @testset "GaugeEquivConv (parallel transport)" begin

        Lt, Ls, C_in, C_out, B = 4, 3, 2, 3, 2
        ndim = 4

        # Random SU(3) gauge links: (3, 3, Lt, Ls, Ls, Ls, ndim, B)
        U = zeros(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, ndim, B)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, mu in 1:ndim, b in 1:B
            U[:, :, t, x, y, z, mu, b] = _random_su3()
        end

        W = randn(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, C_in, B)
        layer = GaugeEquivConv(C_in, C_out; ndim=ndim)
        out   = layer(W, U)

        # Output shape
        @test size(out) == (3, 3, Lt, Ls, Ls, Ls, C_out, B)

        # Manual check at site (t=1, x=1, y=1, z=1), batch b=1:
        # Verify forward transport contribution from (j=1, mu=1).
        # PT_fwd[x] = U_1(x) * W_1(x+t) * U_1(x)'
        # x+t wraps: (1,1,1,1) + mu=1 → t index goes to 2 (dim 3 of the 7D subfield).
        j_test, mu_test, b_test = 1, 1, 1
        t0, x0, y0, z0 = 1, 1, 1, 1
        # mu=1 is the x-direction (LatticeGPU convention 1=x,2=y,3=z,4=t).
        x1 = mod1(x0 + 1, Ls)   # forward x-neighbour

        U_mu = U[:, :, :, :, :, :, mu_test, b_test]   # (3,3,Lt,Ls,Ls,Ls)
        W_j  = W[:, :, :, :, :, :, j_test,  b_test]   # same

        pt_fwd_manual = U_mu[:,:,t0,x0,y0,z0] *
                        W_j[:,:,t0,x1,y0,z0]  *
                        U_mu[:,:,t0,x0,y0,z0]'

        x_back = mod1(x0 - 1, Ls)   # backward x-neighbour
        pt_bwd_manual = U_mu[:,:,t0,x_back,y0,z0]' *
                        W_j[:,:,t0,x_back,y0,z0]   *
                        U_mu[:,:,t0,x_back,y0,z0]

        # Accumulate all (j, mu) contributions to out[:,:,t0,x0,y0,z0,:,b_test]
        # coord_of_dir: LatticeGPU dir mu → index into [t,x,y,z] coordinate list
        #   dir 1=x → index 2,  dir 2=y → index 3,  dir 3=z → index 4,  dir 4=t → index 1
        expected = zeros(ComplexF64, 3, 3, C_out)
        for j in 1:C_in, mu in 1:ndim
            ci     = (2, 3, 4, 1)[mu]
            L_dir  = (Ls, Ls, Ls, Lt)[mu]
            U_mu_j  = U[:, :, :, :, :, :, mu, b_test]
            W_jj    = W[:, :, :, :, :, :, j,  b_test]
            coords_fwd = [t0, x0, y0, z0]
            coords_fwd[ci] = mod1(coords_fwd[ci] + 1, L_dir)
            tf, xf, yf, zf = coords_fwd
            coords_bwd = [t0, x0, y0, z0]
            coords_bwd[ci] = mod1(coords_bwd[ci] - 1, L_dir)
            tb, xb, yb, zb = coords_bwd
            pt_f = U_mu_j[:,:,t0,x0,y0,z0] * W_jj[:,:,tf,xf,yf,zf] * U_mu_j[:,:,t0,x0,y0,z0]'
            pt_b = U_mu_j[:,:,tb,xb,yb,zb]' * W_jj[:,:,tb,xb,yb,zb] * U_mu_j[:,:,tb,xb,yb,zb]
            for i in 1:C_out
                expected[:, :, i] .+= layer.omega[i, j, mu, 1] .* pt_f
                expected[:, :, i] .+= layer.omega[i, j, mu, 2] .* pt_b
            end
        end

        for i in 1:C_out
            @test out[:, :, t0, x0, y0, z0, i, b_test] ≈ expected[:, :, i]  atol=1e-11
        end

        # Gauge covariance: W(x) → V(x)W(x)V†(x), U_mu(x) → V(x)U_mu(x)V†(x+mu)
        # ⟹ out(x) → V(x) out(x) V†(x)
        V = zeros(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, B)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, b in 1:B
            V[:, :, t, x, y, z, b] = _random_su3()
        end

        W2 = zeros(ComplexF64, size(W)...)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, c in 1:C_in, b in 1:B
            v = V[:, :, t, x, y, z, b]
            W2[:, :, t, x, y, z, c, b] = v * W[:, :, t, x, y, z, c, b] * v'
        end

        U2 = zeros(ComplexF64, size(U)...)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, mu in 1:ndim, b in 1:B
            ci     = (2, 3, 4, 1)[mu]   # LatticeGPU dir (1=x,2=y,3=z,4=t) → coord index in [t,x,y,z]
            L_mu   = (Ls, Ls, Ls, Lt)[mu]
            coords_fwd = [t, x, y, z]
            coords_fwd[ci] = mod1(coords_fwd[ci] + 1, L_mu)
            tf, xf, yf, zf = coords_fwd
            v_x    = V[:, :, t,  x,  y,  z,  b]
            v_xpmu = V[:, :, tf, xf, yf, zf, b]
            U2[:, :, t, x, y, z, mu, b] = v_x * U[:, :, t, x, y, z, mu, b] * v_xpmu'
        end

        out2 = layer(W2, U2)

        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, i in 1:C_out, b in 1:B
            v = V[:, :, t, x, y, z, b]
            @test out2[:, :, t, x, y, z, i, b] ≈ v * out[:, :, t, x, y, z, i, b] * v'  atol=1e-10
        end
    end

    # -----------------------------------------------------------------------
    # BilinearLayer
    # -----------------------------------------------------------------------

    @testset "BilinearLayer (Σ_{j,k} α[i,j,k] W_j · W'_k)" begin

        Lt, Ls, C_in1, C_in2, C_out, B = 4, 3, 2, 3, 4, 5
        W  = randn(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, C_in1, B)
        W′ = randn(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, C_in2, B)
        layer = BilinearLayer(C_in1, C_in2, C_out)

        out = layer(W, W′)

        # Output shape
        @test size(out) == (3, 3, Lt, Ls, Ls, Ls, C_out, B)

        # Manual check at one site: out[:,:,t,x,y,z,i,b] = Σ_{j,k} α[i,j,k] * W[:,:,j,...] * W'[:,:,k,...]
        t, x, y, z, b = 1, 1, 1, 1, 1
        for i in 1:C_out
            expected = zeros(ComplexF64, 3, 3)
            for j in 1:C_in1, k in 1:C_in2
                expected .+= layer.α[i, j, k] .* (W[:, :, t, x, y, z, j, b] * W′[:, :, t, x, y, z, k, b])
            end
            @test out[:, :, t, x, y, z, i, b] ≈ expected  atol=1e-12
        end

        # Gauge covariance: W → V W V†, W' → V W' V† ⟹ out → V out V†
        # Proof: (VW_j V†)(VW'_k V†) = V(W_j W'_k)V†
        V  = _random_su3()
        W2  = similar(W);  W′2 = similar(W′)
        for t in 1:Lt, xi in 1:Ls, yi in 1:Ls, zi in 1:Ls, c in 1:C_in1, bi in 1:B
            W2[:, :, t, xi, yi, zi, c, bi] = V * W[:, :, t, xi, yi, zi, c, bi] * V'
        end
        for t in 1:Lt, xi in 1:Ls, yi in 1:Ls, zi in 1:Ls, c in 1:C_in2, bi in 1:B
            W′2[:, :, t, xi, yi, zi, c, bi] = V * W′[:, :, t, xi, yi, zi, c, bi] * V'
        end
        out2 = layer(W2, W′2)
        for t in 1:Lt, xi in 1:Ls, yi in 1:Ls, zi in 1:Ls, i in 1:C_out, bi in 1:B
            @test out2[:, :, t, xi, yi, zi, i, bi] ≈ V * out[:, :, t, xi, yi, zi, i, bi] * V'  atol=1e-11
        end
    end

    # -----------------------------------------------------------------------
    # LCBBlock
    # -----------------------------------------------------------------------

    @testset "LCBBlock (L-Conv → L-Bilin → ScalarGate)" begin

        Lt, Ls, C_in, C_conv, C_out, B = 4, 3, 2, 3, 4, 2
        ndim = 4

        U = zeros(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, ndim, B)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, mu in 1:ndim, b in 1:B
            U[:, :, t, x, y, z, mu, b] = _random_su3()
        end
        W = randn(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, C_in, B)

        block = LCBBlock(C_in, C_conv, C_out; ndim=ndim)
        out   = block(W, U)

        # Output shape
        @test size(out) == (3, 3, Lt, Ls, Ls, Ls, C_out, B)

        # Verify it matches manual Conv → Bilin → Gate
        W_conv  = block.conv(W, U)
        W_bilin = block.bilin(W, W_conv)
        W_gate  = block.gate(W_bilin)
        @test out ≈ W_gate  atol=1e-14

        # Gauge covariance under site-dependent V(x):
        #   W → V W V†, U_μ(x) → V(x) U_μ(x) V†(x+μ̂)  ⟹  out → V out V†
        V = zeros(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, B)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, b in 1:B
            V[:, :, t, x, y, z, b] = _random_su3()
        end

        W2 = zeros(ComplexF64, size(W)...)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, c in 1:C_in, b in 1:B
            v = V[:, :, t, x, y, z, b]
            W2[:, :, t, x, y, z, c, b] = v * W[:, :, t, x, y, z, c, b] * v'
        end

        U2 = zeros(ComplexF64, size(U)...)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, mu in 1:ndim, b in 1:B
            ci     = (2, 3, 4, 1)[mu]   # LatticeGPU dir (1=x,2=y,3=z,4=t) → coord index in [t,x,y,z]
            L_mu   = (Ls, Ls, Ls, Lt)[mu]
            coords_fwd = [t, x, y, z]
            coords_fwd[ci] = mod1(coords_fwd[ci] + 1, L_mu)
            tf, xf, yf, zf = coords_fwd
            v_x    = V[:, :, t,  x,  y,  z,  b]
            v_xpmu = V[:, :, tf, xf, yf, zf, b]
            U2[:, :, t, x, y, z, mu, b] = v_x * U[:, :, t, x, y, z, mu, b] * v_xpmu'
        end

        out2 = block(W2, U2)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, i in 1:C_out, b in 1:B
            v = V[:, :, t, x, y, z, b]
            @test out2[:, :, t, x, y, z, i, b] ≈ v * out[:, :, t, x, y, z, i, b] * v'  atol=1e-10
        end

        # Stackability: two blocks in sequence
        block2 = LCBBlock(C_out, C_conv, C_out; ndim=ndim)
        out_stacked = block2(block(W, U), U)
        @test size(out_stacked) == (3, 3, Lt, Ls, Ls, Ls, C_out, B)

        # Stacked gauge covariance: V(x) on both blocks
        out2_stacked = block2(block(W2, U2), U2)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, i in 1:C_out, b in 1:B
            v = V[:, :, t, x, y, z, b]
            @test out2_stacked[:, :, t, x, y, z, i, b] ≈ v * out_stacked[:, :, t, x, y, z, i, b] * v'  atol=1e-9
        end
    end

    # -----------------------------------------------------------------------
    # build_lcnn / LCNN
    # -----------------------------------------------------------------------

    @testset "build_lcnn (shape, construction, forward pass)" begin

        Lt, Ls, C_in, B = 4, 3, 2, 2
        ndim = 4
        npol = 3
        channels = [3, 4]
        mlp_hidden = 32

        model = build_lcnn(; Lt=Lt, C_in=C_in, ndim=ndim,
                             channels=channels, npol=npol, mlp_hidden=mlp_hidden)

        @test model isa LCNN
        @test length(model.blocks) == 2
        @test model.Lt == Lt
        @test model.npol == npol

        # Block channel dims
        @test size(model.blocks[1].conv.omega) == (channels[1], C_in, ndim, 2)
        @test size(model.blocks[1].bilin.α) == (channels[1], C_in, channels[1])
        @test size(model.blocks[2].conv.omega) == (channels[2], channels[1], ndim, 2)
        @test size(model.blocks[2].bilin.α) == (channels[2], channels[1], channels[2])

        # Forward pass
        U = zeros(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, ndim, B)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, mu in 1:ndim, b in 1:B
            U[:, :, t, x, y, z, mu, b] = _random_su3()
        end
        W = randn(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, C_in, B)

        out = model(W, U)
        @test size(out) == (Lt, npol, B)
        @test eltype(out) <: Real

        # Single-block model
        model1 = build_lcnn(; Lt=Lt, C_in=C_in, ndim=ndim,
                              channels=[3], npol=npol, mlp_hidden=16)
        @test length(model1.blocks) == 1
        out1 = model1(W, U)
        @test size(out1) == (Lt, npol, B)
    end

    # -----------------------------------------------------------------------
    # Full model gauge invariance: build_lcnn output unchanged under V(x)
    # -----------------------------------------------------------------------

    @testset "gauge invariance (build_lcnn output under V(x))" begin

        Lt, Ls, C_in, B = 4, 3, 2, 2
        ndim = 4
        npol = 3

        model = build_lcnn(; Lt=Lt, C_in=C_in, ndim=ndim,
                             channels=[3, 3], npol=npol, mlp_hidden=16)

        U = zeros(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, ndim, B)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, mu in 1:ndim, b in 1:B
            U[:, :, t, x, y, z, mu, b] = _random_su3()
        end
        W = randn(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, C_in, B)

        out = model(W, U)
        @test size(out) == (Lt, npol, B)

        # Site-dependent gauge transformation
        V = zeros(ComplexF64, 3, 3, Lt, Ls, Ls, Ls, B)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, b in 1:B
            V[:, :, t, x, y, z, b] = _random_su3()
        end

        W2 = zeros(ComplexF64, size(W)...)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, c in 1:C_in, b in 1:B
            v = V[:, :, t, x, y, z, b]
            W2[:, :, t, x, y, z, c, b] = v * W[:, :, t, x, y, z, c, b] * v'
        end

        U2 = zeros(ComplexF64, size(U)...)
        for t in 1:Lt, x in 1:Ls, y in 1:Ls, z in 1:Ls, mu in 1:ndim, b in 1:B
            ci     = (2, 3, 4, 1)[mu]   # LatticeGPU dir (1=x,2=y,3=z,4=t) → coord index in [t,x,y,z]
            L_mu   = (Ls, Ls, Ls, Lt)[mu]
            coords_fwd = [t, x, y, z]
            coords_fwd[ci] = mod1(coords_fwd[ci] + 1, L_mu)
            tf, xf, yf, zf = coords_fwd
            v_x    = V[:, :, t,  x,  y,  z,  b]
            v_xpmu = V[:, :, tf, xf, yf, zf, b]
            U2[:, :, t, x, y, z, mu, b] = v_x * U[:, :, t, x, y, z, mu, b] * v_xpmu'
        end

        out2 = model(W2, U2)
        @test out2 ≈ out  atol=1e-9
    end

    # -----------------------------------------------------------------------
    # plaquette_matrices vs plaquette_field (numerical cross-validation)
    # -----------------------------------------------------------------------

    @testset "plaquette_matrices vs plaquette_field (cross-validation)" begin

        conf_file = get(ENV, "GLARE_TEST_CONF", "")
        conf_file = joinpath(conf_file, "A654r000n1")

        if isempty(conf_file) || !isfile(conf_file)
            @warn "Skipping plaquette cross-validation: set ENV[\"GLARE_TEST_CONF\"]."
            @test true broken=true
        else
            # Load config in LatticeGPU format: (bsz, ndim, rsz) SU3{Float64}
            U_lgu = GLARE.import_cern64(conf_file, 0, LP)

            # Reference: plaquette_field in LatticeGPU blocked layout (bsz, npls, rsz)
            # Plane ordering: (4,1),(4,2),(4,3),(3,1),(3,2),(2,1) — t,x,y,z channels.
            P_lgu = plaquette_field(U_lgu, LP)

            # Pack plaquette_field result into (6, Lt, Ls, Ls, Ls, 6) row-storage,
            # reconstruct full 3×3 matrices, add batch dim.
            # Temporal coordinate (coord4, period iL[4]=Lt) placed first → dim3=Lt.
            P_packed = Array{ComplexF64, 6}(undef, 6, LP.iL[4], LP.iL[1], LP.iL[2], LP.iL[3], 6)
            for r in 1:LP.rsz, b in 1:LP.bsz
                coord = point_coord((b, r), LP)
                i1, i2, i3, i4 = coord[1], coord[2], coord[3], coord[4]
                for ipl in 1:6
                    p = P_lgu[b, ipl, r]
                    P_packed[1,i4,i1,i2,i3,ipl] = p.u11
                    P_packed[2,i4,i1,i2,i3,ipl] = p.u12
                    P_packed[3,i4,i1,i2,i3,ipl] = p.u13
                    P_packed[4,i4,i1,i2,i3,ipl] = p.u21
                    P_packed[5,i4,i1,i2,i3,ipl] = p.u22
                    P_packed[6,i4,i1,i2,i3,ipl] = p.u23
                end
            end
            P_ref = reshape(su3_reconstruct(P_packed),
                            3, 3, LP.iL[4], LP.iL[1], LP.iL[2], LP.iL[3], 6, 1)

            # Pack gauge links into (6, Lt, Ls, Ls, Ls, 4), reconstruct, add batch dim.
            # Same convention: temporal coord (coord4) first.
            U_packed = Array{ComplexF64, 6}(undef, 6, LP.iL[4], LP.iL[1], LP.iL[2], LP.iL[3], 4)
            for r in 1:LP.rsz, b in 1:LP.bsz
                coord = point_coord((b, r), LP)
                i1, i2, i3, i4 = coord[1], coord[2], coord[3], coord[4]
                for iμ in 1:4
                    u = U_lgu[b, iμ, r]
                    U_packed[1,i4,i1,i2,i3,iμ] = u.u11
                    U_packed[2,i4,i1,i2,i3,iμ] = u.u12
                    U_packed[3,i4,i1,i2,i3,iμ] = u.u13
                    U_packed[4,i4,i1,i2,i3,iμ] = u.u21
                    U_packed[5,i4,i1,i2,i3,iμ] = u.u22
                    U_packed[6,i4,i1,i2,i3,iμ] = u.u23
                end
            end
            U_plain = reshape(su3_reconstruct(U_packed),
                              3, 3, LP.iL[4], LP.iL[1], LP.iL[2], LP.iL[3], 4, 1)

            # Compute via LCNN's plaquette_matrices and compare.
            # Only check the first time slice to keep memory manageable (~500 MB total).
            P_lcnn = plaquette_matrices(U_plain)   # (3,3,Lt,Ls,Ls,Ls,6,1)

            @test P_lcnn[:, :, 1, :, :, :, :, 1] ≈ P_ref[:, :, 1, :, :, :, :, 1]  atol=1e-10
            @info "plaquette_matrices vs plaquette_field cross-validation passed " *
                  "(max diff t=1 slice: $(maximum(abs.(P_lcnn[:,:,1,:,:,:,:,1] .- P_ref[:,:,1,:,:,:,:,1]))))"
        end
    end

end


