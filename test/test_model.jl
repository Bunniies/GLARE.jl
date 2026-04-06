using Test
using LinearAlgebra
using GLARE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pack first two rows of a 3×3 matrix into the (6,) storage format.
function _pack2rows(U::Matrix{<:Complex})
    return [U[1,1], U[1,2], U[1,3], U[2,1], U[2,2], U[2,3]]
end

# Random SU(3) matrix via QR decomposition.
function _random_su3(T=ComplexF64)
    A = randn(T, 3, 3)
    Q, _ = qr(A)
    U    = Matrix(Q)
    U  ./= det(U)^(1/3)   # enforce det = 1
    return U
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "su3_reconstruct" begin

    @testset "identity matrix" begin
        I3 = ComplexF64[1 0 0; 0 1 0; 0 0 1]
        x  = reshape(_pack2rows(I3), 6, 1)   # (6, 1)
        out = su3_reconstruct(x)             # (3, 3, 1)
        @test size(out) == (3, 3, 1)
        @test out[:, :, 1] ≈ I3
    end

    @testset "single random SU(3)" begin
        U  = _random_su3()
        x  = reshape(_pack2rows(U), 6, 1)   # (6, 1)
        M  = su3_reconstruct(x)[:, :, 1]   # (3, 3)

        # Round-trip: reconstructed matrix matches original
        @test M ≈ U  atol=1e-12

        # Unitarity: U * U† ≈ I
        @test M * M' ≈ I(3)  atol=1e-12

        # Determinant ≈ 1
        @test det(M)  ≈ 1.0  atol=1e-12
    end

    @testset "batch shape (6, Lx, Ly, Lz, Lt, npls)" begin
        Lx, Ly, Lz, Lt, npls = 2, 2, 2, 4, 6

        x = zeros(ComplexF64, 6, Lx, Ly, Lz, Lt, npls)
        Us = Array{Matrix{ComplexF64}}(undef, Lx, Ly, Lz, Lt, npls)

        for i in 1:Lx, j in 1:Ly, k in 1:Lz, l in 1:Lt, p in 1:npls
            U = _random_su3()
            Us[i, j, k, l, p] = U
            x[:, i, j, k, l, p] = _pack2rows(U)
        end

        out = su3_reconstruct(x)
        @test size(out) == (3, 3, Lx, Ly, Lz, Lt, npls)

        # Verify a sample of sites
        for i in 1:Lx, p in 1:npls
            M = out[:, :, i, 1, 1, 1, p]
            @test M ≈ Us[i, 1, 1, 1, p]  atol=1e-12
            @test M * M'   ≈ I(3)         atol=1e-10
            @test det(M)   ≈ 1.0          atol=1e-10
        end
    end

    @testset "Float32 complex input" begin
        U  = _random_su3(ComplexF32)
        x  = reshape(ComplexF32.(_pack2rows(U)), 6, 1)
        M  = su3_reconstruct(x)[:, :, 1]
        @test eltype(M) == ComplexF32
        @test M * M'  ≈ I(3)  atol=1f-5
        @test det(M)  ≈ 1.0f0 atol=1f-5
    end

end
