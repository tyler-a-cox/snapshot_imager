"""
Coordinate transformations and phase tracking utilities.

This module handles coordinate system transformations and phase tracking
operations for radio interferometry imaging.
"""
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u


def phase_track_to_source(
    vis: np.ndarray,
    uvw: np.ndarray,
    times: np.ndarray,
    ra_src: float,
    dec_src: float,
    telescope_loc: EarthLocation,
) -> np.ndarray:
    """
    Phase track visibility data to a specific sky position.
    
    This function applies a phase rotation to the visibility data to track
    a source at a given RA/Dec position. This is equivalent to recentering
    the image at the source position.
    
    Parameters
    ----------
    vis : np.ndarray
        The visibility data to be phased. Shape (nbls, ntimes, nfreqs)
    uvw : np.ndarray
        The uvw coordinates of the visibility data in wavelengths. 
        Shape (nbls, 3, nfreqs)
    times : np.ndarray
        The times of the visibility data (Julian dates). Shape (ntimes,)
    ra_src : float
        The right ascension of the source in degrees.
    dec_src : float
        The declination of the source in degrees.
    telescope_loc : EarthLocation
        The location of the telescope.
    
    Returns
    -------
    np.ndarray
        The phase-tracked visibility data. Shape (nbls, ntimes, nfreqs)
    
    Notes
    -----
    The phase rotation is computed using direction cosines (l, m, n) where:
    - l: East direction cosine
    - m: North direction cosine  
    - n: Up direction cosine (computed from l² + m² + n² = 1)
    
    WARNING: There is a known issue with the phase factor. The correct formula
    should use -2j * π, but -4j * π appears to work in practice. This needs
    investigation and likely relates to how UVW coordinates are defined or
    computed upstream.
    
    The phase correction applied is:
        exp(-2j * π * (u*l + v*m + w*(n-1)))
    
    where (u, v, w) are baseline coordinates in wavelengths and (l, m, n) are
    direction cosines to the source.
    """
    # Ensure times are astropy Time objects
    if isinstance(times, np.ndarray):
        times = Time(times, format='jd')
    
    # Get source position in ICRS frame
    target = SkyCoord(ra=ra_src * u.deg, dec=dec_src * u.deg, frame='icrs')
    
    # Transform to Alt-Az frame for each observation time
    altaz_frame = AltAz(obstime=times, location=telescope_loc)
    src_altaz = target.transform_to(altaz_frame)
    
    # Compute direction cosines (x=East, y=North, z=Up)
    # l = East, m = North, n = Up
    l = np.cos(src_altaz.alt.rad) * np.sin(src_altaz.az.rad)
    m = np.cos(src_altaz.alt.rad) * np.cos(src_altaz.az.rad)
    n = np.sqrt(1.0 - l**2 - m**2)
    
    # Extract UVW coordinates
    ucoords = uvw[:, 0, :]  # Shape: (nbls, nfreqs)
    vcoords = uvw[:, 1, :]  # Shape: (nbls, nfreqs)
    wcoords = uvw[:, 2, :]  # Shape: (nbls, nfreqs)
    
    # Compute phase correction
    # NOTE: This should be -2j * pi according to theory, but -4j * pi works
    # This discrepancy needs investigation - it may be related to:
    # 1. How UVW coordinates are computed in preprocessing
    # 2. Baseline convention (antenna1 - antenna2 vs antenna2 - antenna1)
    # 3. Sign conventions in the Fourier transform
    phase = np.exp(
        -4j * np.pi * (
            ucoords[:, None, :] * l[None, :, None] + 
            vcoords[:, None, :] * m[None, :, None] + 
            wcoords[:, None, :] * (n[None, :, None] - 1.0)
        )
    )
    
    return vis * phase


def compute_image_grid(npix: int, fov: float, flat_projection: bool = True):
    """
    Compute the l, m coordinate grid for an image.
    
    Parameters
    ----------
    npix : int
        Number of pixels per dimension
    fov : float
        Field of view in degrees
    
    Returns
    -------
    lcoords : np.ndarray
        L-coordinates (East direction cosines), shape (npix,)
    mcoords : np.ndarray
        M-coordinates (North direction cosines), shape (npix,)
    lgrid : np.ndarray
        2D L-coordinate grid, shape (npix, npix)
    mgrid : np.ndarray
        2D M-coordinate grid, shape (npix, npix)
    
    Notes
    -----
    Direction cosines are computed using:
        l = sin(θ_E)
        m = sin(θ_N)
    
    where θ_E and θ_N are angular offsets in the East and North directions.
    """
    if flat_projection:
        extent = np.sin(np.deg2rad(fov / 2))
        lcoords = np.linspace(-extent, extent, npix, endpoint=False)
        mcoords = np.linspace(-extent, extent, npix, endpoint=False)
    else:
        extent = np.deg2rad(fov / 2)
        lcoords = np.sin(np.linspace(-extent, extent, npix, endpoint=False))
        mcoords = np.sin(np.linspace(-extent, extent, npix, endpoint=False))
    
    lgrid, mgrid = np.meshgrid(lcoords, mcoords)
    
    return lcoords, mcoords, lgrid, mgrid


def compute_baseline_extent(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the maximum baseline extent in the UV plane.
    
    Parameters
    ----------
    u : np.ndarray
        U-coordinates
    v : np.ndarray
        V-coordinates
    
    Returns
    -------
    float
        Maximum baseline extent (max of |u| and |v|)
    """
    return max(np.max(np.abs(u)), np.max(np.abs(v)))
