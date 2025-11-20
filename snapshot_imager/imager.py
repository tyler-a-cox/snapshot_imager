import tqdm
import numpy as np
import finufft
from hera_cal import redcal, datacontainer, utils
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u

def unpack_data_containers(
    data: datacontainer.DataContainer, 
    flags: datacontainer.DataContainer, 
    nsamples: datacontainer.DataContainer,
    pol: str='ee',
    antpos: dict=None,
    freqs: np.ndarray=None,
    time_slice: tuple=slice(0, None),
    freq_slice: tuple=slice(0, None),
    antpairs: list[tuple]=None,
    weight_by_nsamples: bool=True,
):
    """
    Unpack the data containers into a format suitable for imaging.

    Parameters
    ----------
    data : datacontainer.DataContainer
        The visibility data to be unpacked.
    flags : datacontainer.DataContainer
        The flags for the visibility data.
    nsamples : datacontainer.DataContainer
        The nsamples for the visibility data.
    pol : str
        The polarization to be used. Default is 'ee'.
    antpos : dict
        The antenna positions. If None, will use the positions from the data container.
    freqs : np.ndarray
        The frequencies to be used. If None, will use the frequencies from the data container.
    time_slice : tuple
        The time slice to be used. Default is (0, None).
    freq_slice : tuple
        The frequency slice to be used. Default is (0, None).
    antpairs : list[tuple]
        The antenna pairs to be used. If None, will use all antenna pairs.
    weight_by_nsamples : bool
        Whether to weight the data by nsamples. Default is True.

    Returns
    -------
    imaging_data : dict
        A dictionary containing the unpacked data, weights, uvw coordinates, times, and frequencies.
    """
    if antpairs is None:
        antpairs = data.antpairs()

    if freqs is None:
        freqs = hd.freqs

    if antpos is None:
        antpos = data.antpos
    
    vis = []
    weights = []
    uvw = []
    for ap in antpairs:
        blpol = ap + (pol,)
        blvec = antpos[ap[1]] - antpos[ap[0]]
        
        if weight_by_nsamples:
            weights += [
                nsamples[blpol][time_slice, freq_slice]
                * (~flags[blpol][time_slice, freq_slice]).astype(float)
            ] * 2
        else:
            weights += [
                np.ones_like(flags[blpol][time_slice, freq_slice])
                * (~flags[blpol][time_slice, freq_slice]).astype(float)
            ] * 2
        
        vis += [
            data[blpol][time_slice, freq_slice],
            data[utils.reverse_bl(blpol)][time_slice, freq_slice]
        ]
        uvw += [
            blvec[:, None] * freqs[freq_slice][None] / 2.998e8,
            -blvec[:, None] * freqs[freq_slice][None] / 2.998e8,
        ]

    imaging_data = {
        "vis": np.array(vis),
        "weights": np.array(weights),
        "uvw": np.array(uvw),
        "times": hd.times[time_slice],
        "freqs": freqs[freq_slice]
    }

    return imaging_data

def phase_track_to_source(
    vis: np.ndarray,
    uvw: np.ndarray,
    times: np.ndarray,
    ra_src: float,
    dec_src: float,
    telescope_loc: EarthLocation,
):
    """
    Phase track the data to a source position.

    Parameters
    ----------
    vis : np.ndarray
        The visibility data to be phased. Size (nbls, ntimes, nfreqs)
    uvw : np.ndarray
        The uvw coordinates of the visibility data. Size (nbls, 3, nfreqs)
    times : np.ndarray
        The times of the visibility data. Size (ntimes,)
    ra_src : float
        The right ascension of the source in degrees.
    dec_src : float
        The declination of the source in degrees.
    telescope_loc : EarthLocation
        The location of the telescope. Should be an astropy EarthLocation object.

    Returns
    -------
    vis : np.ndarray
        The phased visibility data. Size (nbls, ntimes, nfreqs)
    """
    if isinstance(times, np.ndarray):
        times = Time(times, format='jd')
        
    # Get source positions throughout obstime
    target = SkyCoord(ra=ra_src * u.deg, dec=dec_src * u.deg, frame='icrs')
    altaz_frame = AltAz(obstime=times, location=telescope_loc)
    src_altaz = target.transform_to(altaz_frame)
    
    # direction cosines (x=East, y=North, z=Up):
    l = np.cos(src_altaz.alt.rad) * np.sin(src_altaz.az.rad)
    m = np.cos(src_altaz.alt.rad) * np.cos(src_altaz.az.rad)
    n = np.sqrt(1.0 - l**2 - m**2) 

    ucoords, vcoords, wcoords = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    phase = np.exp(
        -2j * np.pi * (
            ucoords * l[None, :, None] + 
            vcoords * m[None, :, None] + 
            wcoords * (n[None, :, None] - 1.0)
        )
    )

    return vis * phase


def snapshot_imager_single_frequency(
    data: np.ndarray, 
    weights: np.ndarray,
    ucoords: np.ndarray,
    vcoords: np.ndarray,
    npix: int=200,
    fov: float=10,
    eps=1e-13,
):
    """
    TODO: change the changes of names of the input variables to match unpack_data_containers output

    Perform a snapshot image of the data using the finufft library.


    Parameters
    ----------
    data : np.ndarray
        The visibility data to be imaged. Size (nbls, ntimes, nfreqs)
    weights : np.ndarray
        The weights for the visibility data. Size (nbls, ntimes, nfreqs)
    ucoords : np.ndarray
        The u-coordinates of the visibility data. Size (nbls, nfreqs)
    vcoords : np.ndarray
        The v-coordinates of the visibility data. Size (nbls, nfreqs)
    npix : int
        The number of pixels in the image along one axis
    fov : float
        The field of view in degrees.
    eps : float
        The tolerance for the NUFFT algorithm.

    Returns
    -------
    image_stack : np.ndarray
        The resulting image stack.
    lgrid : np.ndarray
        The l-coordinates of the image grid.
    mgrid : np.ndarray
        The m-coordinates of the image grid.
    """
    # Get maximum extent of the coordinates
    umax = np.max([
        np.max(np.abs(ucoords)), np.max(np.abs(vcoords))
    ])

    # Get the shape of the data for looping
    nbls, ntimes, nfreqs = data.shape

    # Set the grid coordinates
    extent = np.deg2rad(fov/2)
    lcoords = np.sin(np.linspace(-extent, extent, npix))
    mcoords = np.sin(np.linspace(-extent, extent, npix))
    lgrid, mgrid = np.meshgrid(lcoords, mcoords)
    lgrid, mgrid = np.ravel(lgrid), np.ravel(mgrid)
    
    image_stack = []
    for ti in tqdm.tqdm(range(ntimes)):
        freq_stack = []
        for fi in range(nfreqs):
            image = finufft.nufft2d3(
                2 * np.pi * ucoords[:, fi] / umax,
                2 * np.pi * vcoords[:, fi] / umax,
                data[:, ti, fi] * weights[:, ti, fi],
                lgrid * umax,
                mgrid * umax,
                eps=eps,
            )
            freq_stack.append(image)
            
        image_stack.append(freq_stack)

    image_stack = np.reshape(image_stack, (ntimes, nfreqs, npix, npix))

    return image_stack, lgrid, mgrid

def snapshot_imager_mfs(
    data: np.ndarray, 
    weights: np.ndarray,
    ucoords: np.ndarray,
    vcoords: np.ndarray,
    npix: int=200,
    fov: float=10,
    eps=1e-13,
):
    """
    TODO: change the changes of names of the input variables to match unpack_data_containers output

    Perform a snapshot image of the data using the finufft library. This function performs multi-frequency synthesis (MFS) imaging.

    Parameters
    ----------
    data : np.ndarray
        The visibility data to be imaged. Size (nbls, ntimes, nfreqs)
    weights : np.ndarray
        The weights for the visibility data. Size (nbls, ntimes, nfreqs)
    ucoords : np.ndarray
        The u-coordinates of the visibility data. Size (nbls, nfreqs)
    vcoords : np.ndarray
        The v-coordinates of the visibility data. Size (nbls, nfreqs)
    npix : int
        The number of pixels in the image along one axis
    fov : float
        The field of view in degrees.
    eps : float
        The tolerance for the NUFFT algorithm.

    Returns
    -------
    image_stack : np.ndarray
        The resulting image stack.
    lgrid : np.ndarray
        The l-coordinates of the image grid.
    mgrid : np.ndarray
        The m-coordinates of the image grid.
    """
    # Get maximum extent of the coordinates
    umax = np.max([
        np.max(np.abs(ucoords)), np.max(np.abs(vcoords))
    ])

    # Get the shape of the data for looping
    nbls, ntimes, nfreqs = data.shape

    # Set the grid coordinates
    extent = np.deg2rad(fov/2)
    lcoords = np.sin(np.linspace(-extent, extent, npix))
    mcoords = np.sin(np.linspace(-extent, extent, npix))
    lgrid, mgrid = np.meshgrid(lcoords, mcoords)
    lgrid, mgrid = np.ravel(lgrid), np.ravel(mgrid)
    
    image_stack = []
    for ti in tqdm.tqdm(range(ntimes)):
        freq_stack = []
        for fi in range(nfreqs):
            image = finufft.nufft2d3(
                2 * np.pi * ucoords[:, fi] / umax,
                2 * np.pi * vcoords[:, fi] / umax,
                data[:, ti, fi] * weights[:, ti, fi],
                lgrid * umax,
                mgrid * umax,
                eps=eps,
            )
            freq_stack.append(image)
            
        image_stack.append(freq_stack)

    image_stack = np.reshape(image_stack, (ntimes, nfreqs, npix, npix))

    return image_stack, lgrid, mgrid