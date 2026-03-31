module IO

using CUDA, LatticeGPU

export import_cern64, set_reader

"""
    import_cern64(fname::String, ibc::Int, lp::SpaceParm; log=true)

Import a double-precision SU(3) gauge configuration in CERN format.

The CERN binary format stores:
- 4 × Int32  : lattice extents [L1, L2, L3, L4]
- 1 × Float64: average plaquette
- Links in checkerboard (odd-site) order: for each odd site, 4 directions × 2 matrices
  (forward link at the site and backward link at the downward neighbour).

Direction convention on file: (t, x, y, z) → remapped to LatticeGPU order (x, y, z, t)
via `dtr = [4, 1, 2, 3]`.

Returns a CPU Array{SU3{Float64}, 3} of shape (bsz, ndim, rsz).
"""
function import_cern64(fname::String, _ibc::Int, lp::SpaceParm; log::Bool=true)

    fp = open(fname, "r")
    iL = Vector{Int32}(undef, 4)
    read!(fp, iL)
    avgpl = Vector{Float64}(undef, 1)
    read!(fp, avgpl)
    if log
        println("# [import_cern64] Read from conf file: ", iL, " (plaq: ", avgpl, ")")
    end

    # File stores directions as (t, x, y, z); remap to LatticeGPU order (x, y, z, t)
    dtr = [4, 1, 2, 3]
    assign(V, ic) = SU3{Float64}(V[1,ic], V[2,ic], V[3,ic],
                                  V[4,ic], V[5,ic], V[6,ic])

    Ucpu = Array{SU3{Float64}, 3}(undef, lp.bsz, lp.ndim, lp.rsz)
    V = Array{ComplexF64, 2}(undef, 9, 2)

    for i4 in 1:lp.iL[4]
        for i1 in 1:lp.iL[1]
            for i2 in 1:lp.iL[2]
                for i3 in 1:lp.iL[3]
                    if (mod(i1 + i2 + i3 + i4 - 4, 2) == 1)
                        b, r = point_index(CartesianIndex(i1, i2, i3, i4), lp)
                        for id in 1:lp.ndim
                            read!(fp, V)
                            Ucpu[b, dtr[id], r] = assign(V, 1)

                            bd, rd = dw((b, r), dtr[id], lp)
                            Ucpu[bd, dtr[id], rd] = assign(V, 2)
                        end
                    end
                end
            end
        end
    end
    close(fp)

    return Ucpu
end

"""
    set_reader(fmt::String, lp::SpaceParm) -> Function

Return a reader function `f(path::String) -> U` for the given configuration format and
lattice parameters. The boundary condition encoded in `lp` determines the `ibc` flag
passed to the underlying reader.

Supported formats:
- `"cern"`   : CERN double-precision binary (calls `import_cern64`)
- `"lex64"`  : lexicographic 64-bit binary   (calls LatticeGPU `import_lex64`)
- `"native"` : LatticeGPU native format      (calls LatticeGPU `read_cnfg`)
- `"bsfqcd"` : BSFQCD format                 (calls LatticeGPU `import_bsfqcd`)
"""
function set_reader(fmt::String, lp::SpaceParm{N,M,B,D}) where {N,M,B,D}

    if fmt == "cern"
        if B == BC_PERIODIC
            return s -> import_cern64(s, 3, lp)
        elseif B == BC_OPEN
            return s -> import_cern64(s, 0, lp)
        elseif (B == BC_SF_AFWB) || (B == BC_SF_ORBI)
            return s -> import_cern64(s, 1, lp)
        else
            error("set_reader: unsupported boundary condition for CERN format.")
        end

    elseif fmt == "lex64"
        return s -> getindex(import_lex64(s, lp), 1)

    elseif fmt == "native"
        return s -> read_cnfg(s)

    elseif fmt == "bsfqcd"
        return s -> getindex(import_bsfqcd(s, lp), 1)

    else
        error("set_reader: configuration format \"$fmt\" is not supported. " *
              "Choose one of: \"cern\", \"lex64\", \"native\", \"bsfqcd\".")
    end
end

end # module IO
