# Dataset

Functions for building and merging the three HDF5 databases from raw gauge
configurations and LMA correlator files.

## HDF5 schemas

**Scalar gauge database** (`*_gauge_scalar.h5`):
```
├── metadata/  vol, svol, ensemble, config_fmt
└── configs/<id>/
    └── plaq_scalar  Float64[iL1, iL2, iL3, iL4, npls]
```

**Matrix gauge database** (`*_gauge_matrix.h5`):
```
├── metadata/  vol, svol, ensemble, config_fmt
└── configs/<id>/
    └── plaq_matrix  ComplexF64[6, iL1, iL2, iL3, iL4, npls]
```

**Correlator database** (`*_corr.h5`):
```
├── metadata/  lma_path, em, polarizations
└── configs/<id>/<polarization>/
    ├── correlator  Float64[T, nsrcs]
    └── sources     String[nsrcs]
```

Config ids are the trailing integers after `n` in gauge filenames
(e.g. `A654r000n42` → `"42"`). All three databases share the same id keys.

## API

```@docs
build_gauge_dataset
build_gauge_matrix_dataset
build_corr_dataset
merge_dataset
```
