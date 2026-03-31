module Plaquette

using LatticeGPU

export plaquette_field, plaquette_scalar_field

"""
    plaquette_field(U, lp::SpaceParm{N,M,BC_PERIODIC,D}) -> Array{SU3{T}, 3}

Compute the untraced Wilson plaquette matrix at every lattice site and plane.

For each site (b,r) and plane ipl with directions (id1, id2) from lp.plidx:

    P_μν(x) = U_μ(x) · U_ν(x+μ̂) · U†_μ(x+ν̂) · U†_ν(x)

which in LatticeGPU notation is:

    U[b,id1,r] * U[bu1,id2,ru1] / (U[b,id2,r] * U[bu2,id1,ru2])

where `/` is right-multiply by conjugate transpose (A/B = A·B†).

Output shape: (bsz, npls, rsz) — matches LatticeGPU's tensor_field layout.
Plane ordering follows lp.plidx: (N,N-1), (N,N-2), ..., (2,1).

Note: the twist phase (for twisted BC) is not applied to the matrix — it cancels
in gauge-invariant quantities and is irrelevant for zero-twist ensembles (e.g. CLS).
"""
function plaquette_field(U::Array{SU3{T}, 3},
                         lp::SpaceParm{N,M,BC_PERIODIC,D}) where {T,N,M,D}

    out = Array{SU3{T}, 3}(undef, lp.bsz, lp.npls, lp.rsz)

    for r in 1:lp.rsz
        for b in 1:lp.bsz
            ipl = 0
            for id1 in N:-1:1
                bu1, ru1 = up((b, r), id1, lp)
                for id2 in 1:id1-1
                    bu2, ru2 = up((b, r), id2, lp)
                    ipl += 1
                    @inbounds out[b, ipl, r] = U[b, id1, r] * U[bu1, id2, ru1] /
                                               (U[b, id2, r] * U[bu2, id1, ru2])
                end
            end
        end
    end

    return out
end

"""
    plaquette_scalar_field(U, lp::SpaceParm{N,M,BC_PERIODIC,D}) -> Array{Float64, 3}

Compute Re(Tr P_μν(x)) at every lattice site and plane — the 6-channel
gauge-invariant scalar input for the baseline CNN.

Output shape: (bsz, npls, rsz) — matches LatticeGPU's tensor_field layout.

Consistency check (zero-twist ensembles):

    sum(plaquette_scalar_field(U, lp)) / (prod(lp.iL) * lp.npls)

should match the output of LatticeGPU's `plaquette(U, lp, gp, ymws)`.
"""
function plaquette_scalar_field(U::Array{SU3{T}, 3},
                                lp::SpaceParm{N,M,BC_PERIODIC,D}) where {T,N,M,D}

    out = Array{Float64, 3}(undef, lp.bsz, lp.npls, lp.rsz)

    for r in 1:lp.rsz
        for b in 1:lp.bsz
            ipl = 0
            for id1 in N:-1:1
                bu1, ru1 = up((b, r), id1, lp)
                for id2 in 1:id1-1
                    bu2, ru2 = up((b, r), id2, lp)
                    ipl += 1
                    P = U[b, id1, r] * U[bu1, id2, ru1] /
                        (U[b, id2, r] * U[bu2, id1, ru2])
                    @inbounds out[b, ipl, r] = real(tr(P))
                end
            end
        end
    end

    return out
end

end # module Plaquette
