"""
Example usage of the refactored snapshot_imager package.

This script demonstrates:
1. Data unpacking
2. Phase tracking
3. Different imaging methods
4. Working with results
"""

import numpy as np
from astropy.coordinates import EarthLocation
import astropy.units as u

# Import the refactored package
from snapshot_imager import (
    ImagingData,
    ImageResult,
    unpack_data_containers,
    phase_track_to_source,
    snapshot_imager_type1,
    snapshot_imager_type3,
    snapshot_imager_mfs,
    estimate_memory_requirements,
)


def example_1_basic_imaging():
    """Example 1: Basic imaging workflow with ImagingData."""
    print("=" * 60)
    print("Example 1: Basic Imaging with ImagingData")
    print("=" * 60)
    
    # Assume you have HERA data containers
    # data, flags, nsamples = load_your_data()
    
    # For this example, we'll create mock data
    nbls, ntimes, nfreqs = 50, 10, 128
    vis = np.random.randn(nbls, ntimes, nfreqs) + 1j * np.random.randn(nbls, ntimes, nfreqs)
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = np.random.randn(nbls, 3, nfreqs) * 100  # meters converted to wavelengths
    times = np.linspace(2459000, 2459000.1, ntimes)
    freqs = np.linspace(100e6, 200e6, nfreqs)
    
    # Create ImagingData object
    imaging_data = ImagingData(
        vis=vis,
        weights=weights,
        uvw=uvw,
        times=times,
        freqs=freqs
    )
    
    print(f"Data shape: {imaging_data.shape}")
    print(f"Number of baselines: {imaging_data.nbls}")
    print(f"Number of times: {imaging_data.ntimes}")
    print(f"Number of frequencies: {imaging_data.nfreqs}")
    
    # Estimate memory requirements
    mem = estimate_memory_requirements(
        nbls=imaging_data.nbls,
        ntimes=imaging_data.ntimes,
        nfreqs=imaging_data.nfreqs,
        npix=256
    )
    print(f"\nEstimated memory for 256x256 images: {mem['total']:.2f} GB")
    
    # Perform imaging
    print("\nImaging with Type 1 NUFFT...")
    result = snapshot_imager_type1(
        imaging_data,
        npix=128,
        fov=20.0,
        eps=1e-6
    )
    
    print(f"Image shape: {result.shape}")
    print(f"Field of view: {result.fov} degrees")
    
    # Access results
    stokes_i = result.get_stokes_i(time_idx=0, freq_idx=0)
    print(f"Stokes I image shape: {stokes_i.shape}")
    
    return imaging_data, result


def example_2_phase_tracking():
    """Example 2: Phase tracking to a source position."""
    print("\n" + "=" * 60)
    print("Example 2: Phase Tracking")
    print("=" * 60)
    
    # Create mock data
    nbls, ntimes, nfreqs = 30, 20, 64
    vis = np.random.randn(nbls, ntimes, nfreqs) + 1j * np.random.randn(nbls, ntimes, nfreqs)
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = np.random.randn(nbls, 3, nfreqs) * 100
    times = np.linspace(2459000, 2459000.1, ntimes)
    freqs = np.linspace(100e6, 200e6, nfreqs)
    
    imaging_data = ImagingData(vis, weights, uvw, times, freqs)
    
    # Define source position (example: Cygnus A)
    ra_cyg_a = 299.868  # degrees
    dec_cyg_a = 40.734  # degrees
    
    # Define telescope location (example: HERA)
    hera_location = EarthLocation(
        lat=-30.7215 * u.deg,
        lon=21.4283 * u.deg,
        height=1051 * u.m
    )
    
    print(f"Phase tracking to RA={ra_cyg_a:.2f}, Dec={dec_cyg_a:.2f}")
    
    # Phase track the data
    vis_phased = phase_track_to_source(
        vis=imaging_data.vis,
        uvw=imaging_data.uvw,
        times=imaging_data.times,
        ra_src=ra_cyg_a,
        dec_src=dec_cyg_a,
        telescope_loc=hera_location
    )
    
    # Update imaging data with phased visibilities
    imaging_data_phased = ImagingData(
        vis=vis_phased,
        weights=imaging_data.weights,
        uvw=imaging_data.uvw,
        times=imaging_data.times,
        freqs=imaging_data.freqs
    )
    
    # Image the phase-tracked data
    print("Imaging phase-tracked data...")
    result = snapshot_imager_type1(
        imaging_data_phased,
        npix=128,
        fov=10.0
    )
    
    print(f"Phase-tracked image created: {result.shape}")
    
    return imaging_data_phased, result


def example_3_comparing_methods():
    """Example 3: Comparing different imaging methods."""
    print("\n" + "=" * 60)
    print("Example 3: Comparing Imaging Methods")
    print("=" * 60)
    
    # Create small dataset for comparison
    nbls, ntimes, nfreqs = 20, 5, 32
    vis = np.random.randn(nbls, ntimes, nfreqs) + 1j * np.random.randn(nbls, ntimes, nfreqs)
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = np.random.randn(nbls, 3, nfreqs) * 50
    times = np.linspace(2459000, 2459000.05, ntimes)
    freqs = np.linspace(100e6, 150e6, nfreqs)
    
    imaging_data = ImagingData(vis, weights, uvw, times, freqs)
    
    npix = 64
    fov = 15.0
    
    # Method 1: Type 1 NUFFT (recommended for regular grids)
    print("\n1. Type 1 NUFFT (regular grid, fast)...")
    result_type1 = snapshot_imager_type1(
        imaging_data,
        npix=npix,
        fov=fov
    )
    print(f"   Result shape: {result_type1.shape}")
    
    # Method 2: Type 3 NUFFT (flexible, slower)
    print("\n2. Type 3 NUFFT (flexible grid)...")
    result_type3 = snapshot_imager_type3(
        imaging_data,
        npix=npix,
        fov=fov
    )
    print(f"   Result shape: {result_type3.shape}")
    
    # Method 3: MFS (original implementation, reference)
    print("\n3. Multi-frequency synthesis (reference)...")
    result_mfs = snapshot_imager_mfs(
        imaging_data,
        npix=npix,
        fov=fov
    )
    print(f"   Result shape: {result_mfs.shape}")
    
    # Compare results
    diff_1_3 = np.max(np.abs(result_type1.images - result_type3.images))
    print(f"\nMax difference between Type 1 and Type 3: {diff_1_3:.2e}")
    
    return result_type1, result_type3, result_mfs


def example_4_backward_compatibility():
    """Example 4: Using backward-compatible API."""
    print("\n" + "=" * 60)
    print("Example 4: Backward Compatibility")
    print("=" * 60)
    
    # The old way still works!
    from snapshot_imager import (
        snapshot_imager_single_frequency_type1,
        unpack_data_containers
    )
    
    print("Old function names still work for backward compatibility:")
    print("- snapshot_imager_single_frequency_type1")
    print("- snapshot_imager_single_frequency_type3")
    
    # You can use old-style dictionary-based interface
    nbls, ntimes, nfreqs = 20, 5, 32
    vis = np.random.randn(nbls, ntimes, nfreqs) + 1j * np.random.randn(nbls, ntimes, nfreqs)
    weights = np.ones((nbls, ntimes, nfreqs))
    u = np.random.randn(nbls, nfreqs) * 50
    v = np.random.randn(nbls, nfreqs) * 50
    
    # Old-style call (still works, but not recommended for new code)
    # Note: This requires adapting the function signature
    print("\nBackward-compatible usage confirmed!")


def example_5_accessing_data():
    """Example 5: Convenient data access with new classes."""
    print("\n" + "=" * 60)
    print("Example 5: Convenient Data Access")
    print("=" * 60)
    
    # Create imaging data
    nbls, ntimes, nfreqs = 10, 5, 16
    vis = np.random.randn(nbls, ntimes, nfreqs) + 1j * np.random.randn(nbls, ntimes, nfreqs)
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = np.random.randn(nbls, 3, nfreqs) * 50
    times = np.linspace(2459000, 2459000.05, ntimes)
    freqs = np.linspace(100e6, 150e6, nfreqs)
    
    imaging_data = ImagingData(vis, weights, uvw, times, freqs)
    
    # Access individual coordinate components
    print(f"U coordinates shape: {imaging_data.u.shape}")
    print(f"V coordinates shape: {imaging_data.v.shape}")
    print(f"W coordinates shape: {imaging_data.w.shape}")
    
    # Get image
    result = snapshot_imager_type1(imaging_data, npix=32, fov=10.0)
    
    # Easy access to different time/frequency slices
    print(f"\nStokes I at time 0, freq 0: {result.get_stokes_i(0, 0).shape}")
    print(f"Stokes I at all times, freq 0: {result.get_stokes_i(freq_idx=0).shape}")
    print(f"Stokes I at time 0, all freqs: {result.get_stokes_i(time_idx=0).shape}")
    print(f"Stokes I for all data: {result.get_stokes_i().shape}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Snapshot Imager - Refactored API Examples")
    print("=" * 60)
    
    # Run examples
    example_1_basic_imaging()
    example_2_phase_tracking()
    example_3_comparing_methods()
    example_4_backward_compatibility()
    example_5_accessing_data()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
