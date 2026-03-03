"""
Snapshot Imager - Radio interferometry imaging using NUFFT.

This package provides efficient snapshot imaging algorithms for radio
interferometry data using Non-Uniform Fast Fourier Transforms (NUFFT).
"""

# Data models
from .data_models import ImagingData, ImageResult

# Preprocessing
from .preprocessing import unpack_data_containers, unpack_uvdata

# Coordinate transformations
from .coordinates import (
    phase_track_to_source,
    compute_image_grid,
    compute_baseline_extent,
)

# Core utilities
from .core import (
    get_nufft_library,
    estimate_memory_requirements,
)

# Imaging algorithms
from .imager import (
    snapshot_imager_type1,
    snapshot_imager_type3,
    snapshot_imager_mfs_type_1,
    snapshot_imager_mfs_type_3,
)

__version__ = "0.2.0"

__all__ = [
    # Data models
    "ImagingData",
    "ImageResult",
    # Preprocessing
    "unpack_data_containers",
    "unpack_uvdata",
    # Coordinates
    "phase_track_to_source",
    "compute_image_grid",
    "compute_baseline_extent",
    # Core utilities
    "get_nufft_library",
    "estimate_memory_requirements",
    # Imaging algorithms
    "snapshot_imager_type1",
    "snapshot_imager_type3",
    "snapshot_imager_mfs_type_1",
    "snapshot_imager_mfs_type_3",
]
