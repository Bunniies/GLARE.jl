# test/ — Test Reference

## Running tests

```julia
using Pkg; Pkg.test("GLARE")
```

## Environment variables required

| Test file | ENV vars needed |
|---|---|
| `test_reader.jl` | `GLARE_TEST_CONF` — path to a single gauge config file |
| `test_plaquette.jl` | `GLARE_TEST_CONF` |
| `test_correlator.jl` | `GLARE_TEST_CORR` — LMA root dir (integer subdirs) |
| `test_dataset.jl` | `GLARE_TEST_CONF`, `GLARE_TEST_CORR` |
| `test_preprocessing.jl` | `GLARE_TEST_GAUGE_H5` (scalar db), `GLARE_TEST_CORR_H5`; optionally `GLARE_TEST_MATRIX_H5` |

All tests skip gracefully with a `@warn` if the required ENV vars are not set.

## ENV var values (CLS A654)

```
GLARE_TEST_CONF       = .../cls/A654r000n1
GLARE_TEST_CORR       = .../HVP/LMA/A654_all_t_sources/dat
GLARE_TEST_GAUGE_H5   = .../hdf5/A654_all_t_sources/A654_gauge_scalar.h5
GLARE_TEST_MATRIX_H5  = .../hdf5/A654_all_t_sources/A654_gauge_matrix.h5
GLARE_TEST_CORR_H5    = .../hdf5/A654_all_t_sources/A654_corr.h5
```

## Notes

- `test_dataset.jl` uses `config_range=1:1` to build temp HDF5 from a single config.
- `test_preprocessing.jl` matrix test (`field=:matrix`) is skipped unless `GLARE_TEST_MATRIX_H5` is set.
- Pearson r is unreliable with N < ~20 train configs (only N-2 degrees of freedom).
