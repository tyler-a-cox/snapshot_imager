"""
Simple tests to verify the refactored snapshot_imager package works correctly.

Run with: python -m pytest test_refactored.py
Or simply: python test_refactored.py
"""

import numpy as np
from astropy.coordinates import EarthLocation
import astropy.units as u


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    from snapshot_imager import (
        ImagingData,
        ImageResult,
        unpack_data_containers,
        phase_track_to_source,
        snapshot_imager_type1,
        snapshot_imager_type3,
        snapshot_imager_mfs,
        estimate_memory_requirements,
        compute_image_grid,
        compute_baseline_extent,
    )
    
    # Backward compatibility
    from snapshot_imager import (
        snapshot_imager_single_frequency_type1,
        snapshot_imager_single_frequency_type3,
    )
    
    print("All imports successful")


def test_imaging_data():
    """Test ImagingData class."""
    print("\nTesting ImagingData class...")
    from snapshot_imager import ImagingData
    
    # Create valid data
    nbls, ntimes, nfreqs = 10, 5, 16
    vis = np.random.randn(nbls, ntimes, nfreqs) + 1j * np.random.randn(nbls, ntimes, nfreqs)
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = np.random.randn(nbls, 3, nfreqs) * 50
    times = np.linspace(2459000, 2459000.05, ntimes)
    freqs = np.linspace(100e6, 150e6, nfreqs)
    
    # Create ImagingData
    imaging_data = ImagingData(vis, weights, uvw, times, freqs)
    
    # Test properties
    assert imaging_data.shape == (nbls, ntimes, nfreqs)
    assert imaging_data.nbls == nbls
    assert imaging_data.ntimes == ntimes
    assert imaging_data.nfreqs == nfreqs
    assert imaging_data.u.shape == (nbls, nfreqs)
    assert imaging_data.v.shape == (nbls, nfreqs)
    assert imaging_data.w.shape == (nbls, nfreqs)
    
    print("✓ ImagingData class works correctly")
    
    # Test validation
    print("  Testing validation...")
    try:
        bad_data = ImagingData(
            vis=vis,
            weights=np.ones((nbls, ntimes, nfreqs + 1)),  # Wrong shape
            uvw=uvw,
            times=times,
            freqs=freqs
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Validation caught error: {str(e)[:50]}...")


def test_type1_imager():
    """Test Type 1 NUFFT imager."""
    print("\nTesting Type 1 imager...")
    from snapshot_imager import ImagingData, snapshot_imager_type1
    
    # Create small dataset
    nbls, ntimes, nfreqs = 20, 5, 8
    vis = np.random.randn(nbls, ntimes, nfreqs) + 1j * np.random.randn(nbls, ntimes, nfreqs)
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = np.random.randn(nbls, 3, nfreqs) * 50
    times = np.linspace(2459000, 2459000.05, ntimes)
    freqs = np.linspace(100e6, 150e6, nfreqs)
    
    imaging_data = ImagingData(vis, weights, uvw, times, freqs)
    
    # Image
    result = snapshot_imager_type1(imaging_data, npix=32, fov=10.0)
    
    # Check result
    assert result.shape == (ntimes, nfreqs, 32, 32)
    assert result.npix == 32
    assert result.fov == 10.0
    assert len(result.l_coords) == 32
    assert len(result.m_coords) == 32
    
    # Test get_stokes_i
    stokes_i = result.get_stokes_i(0, 0)
    assert stokes_i.shape == (32, 32)
    
    print("✓ Type 1 imager works correctly")


def test_type3_imager():
    """Test Type 3 NUFFT imager."""
    print("\nTesting Type 3 imager...")
    from snapshot_imager import ImagingData, snapshot_imager_type3
    
    # Create small dataset
    nbls, ntimes, nfreqs = 20, 5, 8
    vis = np.random.randn(nbls, ntimes, nfreqs) + 1j * np.random.randn(nbls, ntimes, nfreqs)
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = np.random.randn(nbls, 3, nfreqs) * 50
    times = np.linspace(2459000, 2459000.05, ntimes)
    freqs = np.linspace(100e6, 150e6, nfreqs)
    
    imaging_data = ImagingData(vis, weights, uvw, times, freqs)
    
    # Image
    result = snapshot_imager_type3(imaging_data, npix=32, fov=10.0)
    
    # Check result
    assert result.shape == (ntimes, nfreqs, 32, 32)
    
    print("✓ Type 3 imager works correctly")


def test_mfs_imager():
    """Test MFS imager."""
    print("\nTesting MFS imager...")
    from snapshot_imager import ImagingData, snapshot_imager_mfs
    
    # Create very small dataset (MFS is slow)
    nbls, ntimes, nfreqs = 10, 2, 4
    vis = np.random.randn(nbls, ntimes, nfreqs) + 1j * np.random.randn(nbls, ntimes, nfreqs)
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = np.random.randn(nbls, 3, nfreqs) * 50
    times = np.linspace(2459000, 2459000.05, ntimes)
    freqs = np.linspace(100e6, 150e6, nfreqs)
    
    imaging_data = ImagingData(vis, weights, uvw, times, freqs)
    
    # Image
    result = snapshot_imager_mfs(imaging_data, npix=16, fov=10.0)
    
    # Check result
    assert result.shape == (ntimes, nfreqs, 16, 16)
    
    print("✓ MFS imager works correctly")


def test_phase_tracking():
    """Test phase tracking."""
    print("\nTesting phase tracking...")
    from snapshot_imager import ImagingData, phase_track_to_source
    
    # Create data
    nbls, ntimes, nfreqs = 10, 5, 8
    vis = np.random.randn(nbls, ntimes, nfreqs) + 1j * np.random.randn(nbls, ntimes, nfreqs)
    weights = np.ones((nbls, ntimes, nfreqs))
    uvw = np.random.randn(nbls, 3, nfreqs) * 50
    times = np.linspace(2459000, 2459000.05, ntimes)
    freqs = np.linspace(100e6, 150e6, nfreqs)
    
    imaging_data = ImagingData(vis, weights, uvw, times, freqs)
    
    # Define telescope location
    telescope = EarthLocation(
        lat=-30.7215 * u.deg,
        lon=21.4283 * u.deg,
        height=1051 * u.m
    )
    
    # Phase track
    vis_phased = phase_track_to_source(
        vis=imaging_data.vis,
        uvw=imaging_data.uvw,
        times=imaging_data.times,
        ra_src=299.868,
        dec_src=40.734,
        telescope_loc=telescope
    )
    
    # Check shape is preserved
    assert vis_phased.shape == vis.shape
    
    # Check that phase tracking actually modified the data
    assert not np.allclose(vis_phased, vis)
    
    print("✓ Phase tracking works correctly")


def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")
    from snapshot_imager import (
        estimate_memory_requirements,
        compute_image_grid,
        compute_baseline_extent,
    )
    
    # Memory estimation
    mem = estimate_memory_requirements(nbls=100, ntimes=60, nfreqs=1024, npix=512)
    assert 'total' in mem
    assert 'input_data' in mem
    assert 'output_images' in mem
    assert mem['total'] > 0
    print(f"  Memory estimate: {mem['total']:.2f} GB")
    
    # Image grid
    lcoords, mcoords, lgrid, mgrid = compute_image_grid(npix=64, fov=20.0)
    assert len(lcoords) == 64
    assert len(mcoords) == 64
    assert lgrid.shape == (64, 64)
    assert mgrid.shape == (64, 64)
    
    # Baseline extent
    u = np.random.randn(100, 10) * 50
    v = np.random.randn(100, 10) * 30
    umax = compute_baseline_extent(u, v)
    assert umax > 0
    assert umax >= np.max(np.abs(u))
    assert umax >= np.max(np.abs(v))
    
    print("✓ Utilities work correctly")


def test_backward_compatibility():
    """Test backward compatibility with old API."""
    print("\nTesting backward compatibility...")
    from snapshot_imager import (
        snapshot_imager_single_frequency_type1,
        snapshot_imager_single_frequency_type3,
    )
    
    # Just verify they exist and are callable
    assert callable(snapshot_imager_single_frequency_type1)
    assert callable(snapshot_imager_single_frequency_type3)
    
    print("✓ Backward-compatible aliases exist")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running refactored snapshot_imager tests")
    print("=" * 60)
    
    test_imports()
    test_imaging_data()
    test_type1_imager()
    test_type3_imager()
    test_mfs_imager()
    test_phase_tracking()
    test_utilities()
    test_backward_compatibility()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
