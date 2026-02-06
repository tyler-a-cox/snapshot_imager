"""
Core imaging utilities shared across different imaging algorithms.

This module contains common functionality used by multiple imaging methods,
including library selection, plan management, and coordinate scaling.
"""
import numpy as np
from typing import Tuple, Optional


def get_nufft_library(use_cupy: bool = False):
    """
    Get the appropriate NUFFT library and array library.
    
    Parameters
    ----------
    use_cupy : bool, optional
        Whether to attempt GPU acceleration with CuPy/cuFINUFFT.
        Default is False.
    
    Returns
    -------
    xp : module
        Array library (numpy or cupy)
    nufft_lib : module
        NUFFT library (finufft or cufinufft)
    use_gpu : bool
        Whether GPU acceleration is actually being used
    
    Notes
    -----
    If CuPy/cuFINUFFT are requested but not available, falls back to
    CPU version with a warning.
    """
    if use_cupy:
        try:
            import cupy as cp
            import cufinufft
            xp = cp
            nufft_lib = cufinufft
            use_gpu = True
            print("Using GPU acceleration with CuPy/cuFINUFFT")
        except ImportError:
            print("Warning: CuPy/cuFINUFFT not available, falling back to CPU")
            import numpy as np
            import finufft
            xp = np
            nufft_lib = finufft
            use_gpu = False
    else:
        import numpy as np
        import finufft
        xp = np
        nufft_lib = finufft
        use_gpu = False
    
    return xp, nufft_lib, use_gpu


def scale_uv_coordinates(
    u: np.ndarray,
    v: np.ndarray,
    npix: int,
    normalization: str = 'type1'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale UV coordinates for NUFFT operations.
    
    Parameters
    ----------
    u : np.ndarray
        U-coordinates in wavelengths
    v : np.ndarray
        V-coordinates in wavelengths
    npix : int
        Number of pixels in the image
    normalization : str, optional
        Normalization scheme: 'type1' or 'type3'. Default is 'type1'.
    
    Returns
    -------
    u_scaled : np.ndarray
        Scaled U-coordinates
    v_scaled : np.ndarray
        Scaled V-coordinates
    
    Notes
    -----
    Type 1 NUFFT uses: uv_scale = 4π / npix
    Type 3 NUFFT uses: uv_scale = 2π / umax
    """
    if normalization == 'type1':
        uv_scale = 4 * np.pi / npix
        return u * uv_scale, v * uv_scale
    elif normalization == 'type3':
        umax = max(np.max(np.abs(u)), np.max(np.abs(v)))
        uv_scale = 2 * np.pi / umax
        return u * uv_scale, v * uv_scale
    else:
        raise ValueError(f"Unknown normalization: {normalization}")


def prepare_weighted_visibilities(
    vis: np.ndarray,
    weights: np.ndarray,
    freq_idx: int
) -> np.ndarray:
    """
    Prepare weighted visibilities for a specific frequency.
    
    Parameters
    ----------
    vis : np.ndarray
        Visibility data, shape (nbls, ntimes, nfreqs)
    weights : np.ndarray
        Visibility weights, shape (nbls, ntimes, nfreqs)
    freq_idx : int
        Frequency channel index
    
    Returns
    -------
    np.ndarray
        Weighted visibilities, shape (ntimes, nbls)
    
    Notes
    -----
    The output is transposed to (ntimes, nbls) for compatibility with
    FINUFFT's n_trans parameter.
    """
    weighted_data = vis[:, :, freq_idx] * weights[:, :, freq_idx]
    return weighted_data.T  # Transpose to (ntimes, nbls)


def validate_imaging_inputs(
    vis: np.ndarray,
    weights: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    npix: int,
    fov: float,
):
    """
    Validate imaging input parameters.
    
    Parameters
    ----------
    vis : np.ndarray
        Visibility data
    weights : np.ndarray
        Visibility weights
    u : np.ndarray
        U-coordinates
    v : np.ndarray
        V-coordinates
    npix : int
        Number of pixels
    fov : float
        Field of view in degrees
    
    Raises
    ------
    ValueError
        If any inputs are invalid
    """
    if vis.shape != weights.shape:
        raise ValueError(
            f"vis shape {vis.shape} doesn't match weights shape {weights.shape}"
        )
    
    nbls, ntimes, nfreqs = vis.shape
    
    if u.shape != (nbls, nfreqs):
        raise ValueError(
            f"u shape {u.shape} doesn't match expected shape ({nbls}, {nfreqs})"
        )
    
    if v.shape != (nbls, nfreqs):
        raise ValueError(
            f"v shape {v.shape} doesn't match expected shape ({nbls}, {nfreqs})"
        )
    
    if npix <= 0:
        raise ValueError(f"npix must be positive, got {npix}")
    
    if fov <= 0 or fov > 180:
        raise ValueError(f"fov must be between 0 and 180 degrees, got {fov}")


def estimate_memory_requirements(
    nbls: int,
    ntimes: int,
    nfreqs: int,
    npix: int,
    dtype: np.dtype = np.complex128
) -> dict:
    """
    Estimate memory requirements for imaging.
    
    Parameters
    ----------
    nbls : int
        Number of baselines
    ntimes : int
        Number of time samples
    nfreqs : int
        Number of frequency channels
    npix : int
        Number of pixels per dimension
    dtype : np.dtype, optional
        Data type for complex arrays
    
    Returns
    -------
    dict
        Dictionary with memory estimates in GB:
        - 'input_data': Input visibility data
        - 'output_images': Output image cube
        - 'working_memory': Estimated working memory
        - 'total': Total estimated memory
    """
    bytes_per_element = np.dtype(dtype).itemsize
    
    input_data = nbls * ntimes * nfreqs * bytes_per_element / 1e9
    output_images = ntimes * nfreqs * npix * npix * bytes_per_element / 1e9
    
    # Rough estimate for FINUFFT working memory
    # This is approximate and depends on the NUFFT algorithm
    working_memory = max(input_data, output_images) * 2
    
    total = input_data + output_images + working_memory
    
    return {
        'input_data': input_data,
        'output_images': output_images,
        'working_memory': working_memory,
        'total': total
    }
