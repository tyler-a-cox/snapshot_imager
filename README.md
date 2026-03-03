# snapshot_imager

`snapshot_imager` is a Python package for radio interferometric snapshot imaging using Non-Uniform Fast Fourier Transforms (NUFFT). It is designed to efficiently produce dirty image cubes from visibility data, with support for multiple NUFFT strategies (Type 1, Type 3, and multi-frequency synthesis) and optional GPU acceleration via CuPy and cuFINUFFT.

## Installation

Install from source:

```bash
git clone https://github.com/tyler-a-cox/snapshot_imager.git
cd snapshot_imager
pip install .
```

For GPU support, install the optional dependencies:

```bash
pip install cupy cufinufft
```

## Basic Usage

The typical workflow is to unpack HERA `DataContainer` objects into an `ImagingData` container, then pass that to one of the imaging functions.

```python
import numpy as np
from snapshot_imager import unpack_data_containers, snapshot_imager_type1

# data, flags, and nsamples are hera_cal DataContainer objects
imaging_data = unpack_data_containers(
    data=data,
    flags=flags,
    nsamples=nsamples,
    pol="ee",
    antpos=antpos,
    freqs=freqs,
)

# Produce a (ntimes, nfreqs, npix, npix) image cube
result = snapshot_imager_type1(
    imaging_data,
    npix=256,
    fov=10.0,       # Field of view in degrees
    use_cupy=False, # Set to True to use GPU acceleration
)

print(result.images.shape)  # (ntimes, nfreqs, npix, npix)
```

The returned `ImageResult` contains the image cube along with the corresponding `l_coords` and `m_coords` (direction cosines) for plotting or downstream analysis.

## GPU Acceleration

`snapshot_imager` supports GPU-accelerated imaging via [CuPy](https://cupy.dev/) and [cuFINUFFT](https://github.com/flatironinstitute/cufinufft). Simply pass `use_cupy=True` to any imaging function:

```python
result = snapshot_imager_type1(imaging_data, npix=256, fov=10.0, use_cupy=True)
```

If CuPy or cuFINUFFT are not available, the package will automatically fall back to the CPU implementation with a warning.

## License

MIT
