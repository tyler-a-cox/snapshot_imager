"""
Snapshot imaging algorithms using Non-Uniform FFT (NUFFT).

This module implements different snapshot imaging algorithms for radio
interferometry data. All algorithms use FINUFFT for efficient computation
and support optional GPU acceleration via CuPy.
"""
import numpy as np
import tqdm

from .data_models import ImagingData, ImageResult
from .coordinates import compute_image_grid
from .core import (
    get_nufft_library,
    prepare_weighted_visibilities,
    validate_imaging_inputs,
)


def snapshot_imager_type1(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 180,
    eps: float = 1e-13,
    use_cupy: bool = False,
    modeord: int = 0,
) -> ImageResult:
    """
    Snapshot imager using Type 1 NUFFT with plan reuse.
    
    Type 1 NUFFT transforms from non-uniform points (UV coordinates) to
    a uniform grid (image pixels). This is the most efficient approach
    for snapshot imaging when the output grid is regular.
    
    Parameters
    ----------
    imaging_data : ImagingData
        Prepared visibility data
    npix : int, optional
        Number of pixels per dimension. Default is 200.
    fov : float, optional
        Field of view in degrees. Default is 180.
    eps : float, optional
        FINUFFT tolerance (precision). Default is 1e-13.
    use_cupy : bool, optional
        Whether to use GPU acceleration. Default is False.
    modeord : int, optional
        Mode ordering: 0 for FINUFFT default (fftshift), 1 for FFT-style.
        Default is 0.
    
    Returns
    -------
    ImageResult
        Container with image cube and coordinate information
    
    Notes
    -----
    This algorithm creates a FINUFFT Plan once per frequency channel and
    processes all time samples together using the n_trans parameter for
    optimal performance.
    
    The UV coordinates are scaled by 4π/npix for Type 1 NUFFT.
    """
    # Validate inputs
    validate_imaging_inputs(
        imaging_data.vis,
        imaging_data.weights,
        imaging_data.u,
        imaging_data.v,
        npix,
        fov
    )
    
    # Get appropriate libraries
    xp, nufft_lib, use_gpu = get_nufft_library(use_cupy)
    
    # Extract dimensions
    nbls, ntimes, nfreqs = imaging_data.shape
    
    # Set up the image grid
    lcoords, mcoords, _, _ = compute_image_grid(npix, fov)
    
    # Pre-allocate output array
    image_stack = np.zeros((ntimes, nfreqs, npix, npix), dtype=complex)
    
    # Compute the normalization factor for Type 1 NUFFT
    umax = max(np.max(np.abs(imaging_data.u)), np.max(np.abs(imaging_data.v)))
    l_max = np.max([np.abs(lcoords).max(), np.abs(mcoords).max()])
    norm_factor = 4 * np.pi / umax * l_max

    # Process each frequency
    for fi in tqdm.tqdm(range(nfreqs), desc="Imaging frequencies"):
        # Get and scale UV coordinates for this frequency
        u_scaled = imaging_data.u[:, fi] * norm_factor
        v_scaled = imaging_data.v[:, fi] * norm_factor
        
        if use_gpu:
            # Transfer UV coords to GPU once per frequency
            u_gpu = xp.asarray(u_scaled)
            v_gpu = xp.asarray(v_scaled)
            
            # Create Type 1 plan
            plan = nufft_lib.Plan(
                nufft_type=1,
                n_modes=(npix, npix),
                n_trans=ntimes,
                eps=eps,
                dtype=xp.complex64 if imaging_data.vis.dtype == np.complex64 else xp.complex128,
                modeord=modeord
            )
            
            # Set the non-uniform points
            plan.setpts(u_gpu, v_gpu)
            
            # Prepare weighted data for all times
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                freq_idx=fi
            )
            data_gpu = xp.asarray(weighted_data)
            
            # Execute transform for all time steps at once
            output_gpu = plan.execute(data_gpu)
            
            # Transfer back to CPU, reshape, and store results
            image_stack[:, fi, :, :] = np.transpose(xp.asnumpy(output_gpu), axes=(0, 2, 1))
            
        else:
            # CPU version
            plan = nufft_lib.Plan(
                nufft_type=1,
                n_modes=(npix, npix),
                n_trans=ntimes,
                eps=eps,
                dtype=np.complex64 if imaging_data.vis.dtype == np.complex64 else np.complex128,
                modeord=modeord
            )
            
            # Set the non-uniform points
            plan.setpts(u_scaled, v_scaled)
            
            # Prepare weighted data for all times
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                freq_idx=fi
            )
            
            # Execute transform
            output = plan.execute(weighted_data)
            
            # Store results, Type 1 is transposed compared to Type 3
            image_stack[:, fi, :, :] = np.transpose(output, axes=(0, 2, 1))
    
    return ImageResult(
        images=image_stack,
        l_coords=lcoords,
        m_coords=mcoords,
        fov=fov,
        npix=npix
    )


def snapshot_imager_type3(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 180,
    eps: float = 1e-13,
    use_cupy: bool = False,
) -> ImageResult:
    """
    Snapshot imager using Type 3 NUFFT with plan reuse.
    
    Type 3 NUFFT transforms from non-uniform points to non-uniform points.
    This can be useful for irregular output grids or when evaluating at
    specific sky positions.
    
    Parameters
    ----------
    imaging_data : ImagingData
        Prepared visibility data
    npix : int, optional
        Number of pixels per dimension. Default is 200.
    fov : float, optional
        Field of view in degrees. Default is 180.
    eps : float, optional
        FINUFFT tolerance (precision). Default is 1e-13.
    use_cupy : bool, optional
        Whether to use GPU acceleration. Default is False.
    
    Returns
    -------
    ImageResult
        Container with image cube and coordinate information
    
    Notes
    -----
    Type 3 NUFFT is generally slower than Type 1 for regular output grids,
    but provides more flexibility in choosing output positions.
    
    The output coordinates are flattened (1D arrays) for Type 3.
    """
    # Validate inputs
    validate_imaging_inputs(
        imaging_data.vis,
        imaging_data.weights,
        imaging_data.u,
        imaging_data.v,
        npix,
        fov
    )
    
    # Get appropriate libraries
    xp, nufft_lib, use_gpu = get_nufft_library(use_cupy)
    
    # Extract dimensions
    nbls, ntimes, nfreqs = imaging_data.shape
    
    # Set up the image grid
    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)
    lgrid_flat = np.ravel(lgrid)
    mgrid_flat = np.ravel(mgrid)

    # Get maximum baseline extent for scaling
    umax = max(np.max(np.abs(imaging_data.u)), np.max(np.abs(imaging_data.v)))
    
    # Pre-compute grid coordinates scaled by umax
    lgrid_scaled = lgrid_flat * umax
    mgrid_scaled = mgrid_flat * umax
    
    # Normalization factor
    norm_factor = 2 * np.pi / umax
    
    # Number of output points
    n_out = npix * npix
    
    # Pre-allocate output array
    image_stack = np.zeros((ntimes, nfreqs, npix, npix), dtype=complex)
    
    # Process each frequency
    for fi in tqdm.tqdm(range(nfreqs), desc="Imaging frequencies"):
        # Scale UV coordinates for this frequency
        u_scaled = imaging_data.u[:, fi] * norm_factor
        v_scaled = imaging_data.v[:, fi] * norm_factor
        
        if use_gpu:
            # Transfer coordinates to GPU once per frequency
            u_gpu = xp.asarray(u_scaled)
            v_gpu = xp.asarray(v_scaled)
            lgrid_gpu = xp.asarray(lgrid_scaled)
            mgrid_gpu = xp.asarray(mgrid_scaled)
            
            # Create Type 3 plan
            plan = nufft_lib.Plan(
                nufft_type=3,
                n_modes=2,  # Dimension (2D transform)
                n_trans=ntimes,
                eps=eps,
                dtype=xp.complex64 if imaging_data.vis.dtype == np.complex64 else xp.complex128
            )
            
            # Set source and target points
            plan.setpts(u_gpu, v_gpu, s=lgrid_gpu, t=mgrid_gpu)
            
            # Prepare weighted data
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                freq_idx=fi
            )
            data_gpu = xp.asarray(weighted_data)
            
            # Execute transform
            output_gpu = plan.execute(data_gpu)
            
            # Transfer back and reshape
            output_cpu = xp.asnumpy(output_gpu)
            image_stack[:, fi, :, :] = output_cpu.reshape(ntimes, npix, npix)
            
        else:
            # CPU version
            plan = nufft_lib.Plan(
                nufft_type=3,
                n_modes_or_dim=2,
                n_trans=ntimes,
                eps=eps,
                dtype=np.complex64 if imaging_data.vis.dtype == np.complex64 else np.complex128
            )
            
            # Set source and target points
            plan.setpts(u_scaled, v_scaled, s=lgrid_scaled, t=mgrid_scaled)
            
            # Prepare weighted data
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                freq_idx=fi
            )
            
            # Execute transform
            output = plan.execute(weighted_data)
            
            # Reshape and store
            image_stack[:, fi, :, :] = output.reshape(ntimes, npix, npix)
    
    return ImageResult(
        images=image_stack,
        l_coords=lgrid_flat,
        m_coords=mgrid_flat,
        fov=fov,
        npix=npix
    )

def snapshot_imager_mfs_type_1(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 10,
    eps: float = 1e-13,
    use_cupy: bool = False,
) -> ImageResult:
    """
    Multi-frequency synthesis (MFS) snapshot imager using Type 3 NUFFT with plan reuse.
    
    This implementation uses FINUFFT plans for efficient Type 3 NUFFT computation,
    processing all time samples together using the n_trans parameter. Supports
    optional GPU acceleration via CuPy/cuFINUFFT.
    
    Parameters
    ----------
    imaging_data : ImagingData
        Prepared visibility data
    npix : int, optional
        Number of pixels per dimension. Default is 200.
    fov : float, optional
        Field of view in degrees. Default is 10.
    eps : float, optional
        FINUFFT tolerance (precision). Default is 1e-13.
    use_cupy : bool, optional
        Whether to use GPU acceleration. Default is False.
    
    Returns
    -------
    ImageResult
        Container with image cube and coordinate information
    
    Notes
    -----
    Type 3 NUFFT transforms from non-uniform points to non-uniform points.
    This implementation creates a plan for each frequency channel and processes
    all time samples together for better efficiency.
    
    For regular output grids, consider snapshot_imager_type1 which is typically
    faster than Type 3.
    """
    # Validate inputs
    validate_imaging_inputs(
        imaging_data.vis,
        imaging_data.weights,
        imaging_data.u,
        imaging_data.v,
        npix,
        fov
    )
    
    # Get appropriate libraries
    xp, nufft_lib, use_gpu = get_nufft_library(use_cupy)
    
    # Extract dimensions
    nbls, ntimes, nfreqs = imaging_data.shape
    
    # Set up the image grid
    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)
    lgrid_flat = np.ravel(lgrid)
    mgrid_flat = np.ravel(mgrid)

    # Get maximum grid extent for scaling
    l_max = np.max([np.abs(lgrid_flat).max(), np.abs(mgrid_flat).max()])
    
    # Get maximum baseline extent for scaling
    umax = max(np.max(np.abs(imaging_data.u)), np.max(np.abs(imaging_data.v)))
    
    # Pre-compute grid coordinates scaled by umax
    lgrid_scaled = lgrid_flat * umax
    mgrid_scaled = mgrid_flat * umax
    
    # Normalization factor
    norm_factor = 4 * np.pi / umax * l_max
        
    # Pre-allocate output array
    image_stack = np.zeros((ntimes, nfreqs, npix, npix), dtype=complex)
    
    # Process each frequency
    for ti in tqdm.tqdm(range(ntimes), desc="Imaging Times"):
        # Scale UV coordinates for this frequency
        u_scaled = np.ravel(imaging_data.u * norm_factor)
        v_scaled = np.ravel(imaging_data.v * norm_factor)
        
        if use_gpu:
            # Transfer coordinates to GPU once per frequency
            u_gpu = xp.asarray(u_scaled)
            v_gpu = xp.asarray(v_scaled)
            lgrid_gpu = xp.asarray(lgrid_scaled)
            mgrid_gpu = xp.asarray(mgrid_scaled)
            
            # Create Type 3 plan
            plan = nufft_lib.Plan(
                nufft_type=1,
                n_modes=(npix, npix),  # Dimension (2D transform)
                n_trans=1,
                eps=eps,
                dtype=xp.complex64 if imaging_data.vis.dtype == np.complex64 else xp.complex128,
                modeord=0,
            )
            
            # Set source and target points
            plan.setpts(u_gpu, v_gpu, s=lgrid_gpu, t=mgrid_gpu)
            
            # Prepare weighted data
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                time_idx=ti
            )
            data_gpu = xp.asarray(xp.ravel(weighted_data))

            # Execute transform
            output_gpu = plan.execute(data_gpu)
            
            # Transfer back and reshape
            # All frequencies are processed together, so we need to reshape correctly
            output_cpu = xp.asnumpy(output_gpu)
            image_stack[ti, :, :, :] = output_cpu.reshape(nfreqs, npix, npix)

        else:
            # CPU version
            plan = nufft_lib.Plan(
                nufft_type=1,
                n_modes=(npix, npix),
                n_trans=1,
                eps=eps,
                dtype=np.complex64 if imaging_data.vis.dtype == np.complex64 else np.complex128,
                modeord=0,
            )
            
            # Set source and target points
            plan.setpts(u_scaled, v_scaled, s=lgrid_scaled, t=mgrid_scaled)
            
            # Prepare weighted data
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                time_idx=ti
            )
            
            # Execute transform
            output = plan.execute(np.ravel(weighted_data))
            
            # Reshape and store
            #All frequencies are processed together, so we need to reshape correctly
            image_stack[ti, :, :, :] = output.reshape(nfreqs, npix, npix)

    return ImageResult(
        images=image_stack,
        l_coords=lgrid_flat,
        m_coords=mgrid_flat,
        fov=fov,
        npix=npix
    )

def snapshot_imager_mfs_type_3(
    imaging_data: ImagingData,
    npix: int = 200,
    fov: float = 10,
    eps: float = 1e-13,
    use_cupy: bool = False,
) -> ImageResult:
    """
    Multi-frequency synthesis (MFS) snapshot imager using Type 3 NUFFT with plan reuse.
    
    This implementation uses FINUFFT plans for efficient Type 3 NUFFT computation,
    processing all time samples together using the n_trans parameter. Supports
    optional GPU acceleration via CuPy/cuFINUFFT.
    
    Parameters
    ----------
    imaging_data : ImagingData
        Prepared visibility data
    npix : int, optional
        Number of pixels per dimension. Default is 200.
    fov : float, optional
        Field of view in degrees. Default is 10.
    eps : float, optional
        FINUFFT tolerance (precision). Default is 1e-13.
    use_cupy : bool, optional
        Whether to use GPU acceleration. Default is False.
    
    Returns
    -------
    ImageResult
        Container with image cube and coordinate information
    
    Notes
    -----
    Type 3 NUFFT transforms from non-uniform points to non-uniform points.
    This implementation creates a plan for each frequency channel and processes
    all time samples together for better efficiency.
    
    For regular output grids, consider snapshot_imager_type1 which is typically
    faster than Type 3.
    """
    # Validate inputs
    validate_imaging_inputs(
        imaging_data.vis,
        imaging_data.weights,
        imaging_data.u,
        imaging_data.v,
        npix,
        fov
    )
    
    # Get appropriate libraries
    xp, nufft_lib, use_gpu = get_nufft_library(use_cupy)
    
    # Extract dimensions
    nbls, ntimes, nfreqs = imaging_data.shape
    
    # Set up the image grid
    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix, fov)
    lgrid_flat = np.ravel(lgrid)
    mgrid_flat = np.ravel(mgrid)

    # Get maximum baseline extent for scaling
    umax = max(np.max(np.abs(imaging_data.u)), np.max(np.abs(imaging_data.v)))
    
    # Pre-compute grid coordinates scaled by umax
    lgrid_scaled = lgrid_flat * umax
    mgrid_scaled = mgrid_flat * umax
        
    # Compute the normalization factor for Type 3 NUFFT
    norm_factor = 2 * np.pi / umax
    
    # Pre-allocate output array
    image_stack = np.zeros((ntimes, nfreqs, npix, npix), dtype=complex)
    
    # Process each frequency
    for ti in tqdm.tqdm(range(ntimes), desc="Imaging Times"):
        # Scale UV coordinates for this frequency
        u_scaled = np.ravel(imaging_data.u * norm_factor)
        v_scaled = np.ravel(imaging_data.v * norm_factor)
        
        if use_gpu:
            # Transfer coordinates to GPU once per frequency
            u_gpu = xp.asarray(u_scaled)
            v_gpu = xp.asarray(v_scaled)
            lgrid_gpu = xp.asarray(lgrid_scaled)
            mgrid_gpu = xp.asarray(mgrid_scaled)
            
            # Create Type 3 plan
            plan = nufft_lib.Plan(
                nufft_type=3,
                n_modes=2,  # Dimension (2D transform)
                n_trans=1,
                eps=eps,
                dtype=xp.complex64 if imaging_data.vis.dtype == np.complex64 else xp.complex128
            )
            
            # Set source and target points
            plan.setpts(u_gpu, v_gpu, s=lgrid_gpu, t=mgrid_gpu)
            
            # Prepare weighted data
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                time_idx=ti
            )
            data_gpu = xp.asarray(xp.ravel(weighted_data))

            # Execute transform
            output_gpu = plan.execute(data_gpu)
            
            # Transfer back and reshape
            # All frequencies are processed together, so we need to reshape correctly
            output_cpu = xp.asnumpy(output_gpu)
            image_stack[ti, :, :, :] = output_cpu.reshape(nfreqs, npix, npix)

        else:
            # CPU version
            plan = nufft_lib.Plan(
                nufft_type=3,
                n_modes_or_dim=2,
                n_trans=1,
                eps=eps,
                dtype=np.complex64 if imaging_data.vis.dtype == np.complex64 else np.complex128
            )
            
            # Set source and target points
            plan.setpts(u_scaled, v_scaled, s=lgrid_scaled, t=mgrid_scaled)
            
            # Prepare weighted data
            weighted_data = prepare_weighted_visibilities(
                imaging_data.vis,
                imaging_data.weights,
                time_idx=ti
            )
            
            # Execute transform
            output = plan.execute(np.ravel(weighted_data))
            
            # Reshape and store
            #All frequencies are processed together, so we need to reshape correctly
            image_stack[ti, :, :, :] = output.reshape(nfreqs, npix, npix)

    return ImageResult(
        images=image_stack,
        l_coords=lgrid_flat,
        m_coords=mgrid_flat,
        fov=fov,
        npix=npix
    )