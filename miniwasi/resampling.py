"""
Loading the SIOP / water data files in miniwasi/data and resampling them to sensor bands.

Only the functions used by miniwasi.MiniWASI.MiniWasi are kept.
"""

import numpy as np
import pandas as pd
from spectral import BandResampler
from pathlib import Path

### ADAPTED FROM THE BIO-OPTICS PACKAGE SOURCECODE
# König, M., Noel, P., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023): 
# bio_optics python package (Version x) [Software]. Available from https://github.com/CMLandOcean/bio_optics. 
# [https://doi.org/10.5281/zenodo.10246860]

# Data folder inside the package (miniwasi/data)
data_dir = Path(__file__).resolve().parent / "data"

### Water

def resample_a_w(wavelengths = np.arange(400,800), FWHMs = None):
    """
    Absorption coefficient of pure water [m-1] at a reference temperature of 20 degC 
    as a compilation from different sources as distributed with the Water Color Simulator 6 (WASI6) [1]

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to resample the absorption coefficient of pure water to
    :return: absorption coefficient of pure water absorption resampled to input wavelengths
    """
    a_w_db = pd.read_csv(data_dir / 'a_w.txt', skiprows=14, sep='\t', usecols=[0,1])
    # resample to sensor bands
    band_resampler = BandResampler(a_w_db.wavelength_nm.values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    a_w = band_resampler(a_w_db["a"])
    return a_w


def resample_da_W_div_dT(wavelengths = np.arange(400,800), FWHMs = None):
    """
    Temperature gradient of pure water absorption [m-1  degC-1]
    after Roettgers et al. (2013) [1] as distributed with the Water Color Simulator 6 (WASI6) [2]

    [1] Roettgers et al. (2013): Pure water spectral absorption, scattering, and real part of refractive index model.
                                 Algorithm Theoretical Basis Document "The Water Optical Properties Processor (WOPP).
                                 Distribution: Marc Bouvet, ESA/ESRIN
                                 Revision 7, May 2013
    [2] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

    :param wavelengths: wavelengths to resample the temperature gradient of pure water absorption to
    :return: temperature gradient of pure water absorption resampled to input wavelengths
    """
    # read file
    da_W_div_dT_db = pd.read_csv(data_dir / 'daWdT.txt', skiprows=9, sep='\t')
    # resample to sensor bands
    band_resampler = BandResampler(da_W_div_dT_db.wavelength_nm.values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    da_W_div_dT = band_resampler(da_W_div_dT_db["daW/dT"])
    return da_W_div_dT


### Phytoplankton

def _pad_spectrum_with_zeros(wl, values, out_min, out_max, step):
    """
    Extend a single spectrum with zero-valued samples outside its measured
    wavelength range, so that a BandResampler built on it covers the full
    requested output range [out_min, out_max] without extrapolating
    non-zero values beyond where the spectrum was actually measured.

    :param wl: measured wavelengths [nm] of the spectrum, 1D
    :param values: measured spectral values at wl, 1D, same length as wl
    :param out_min: lowest wavelength [nm] the output needs to cover
    :param out_max: highest wavelength [nm] the output needs to cover
    :param step: wavelength spacing [nm] to use for the added zero-fill samples
    :return: (wl_padded, values_padded), sorted by wavelength
    """
    wl = np.asarray(wl, dtype=float)
    values = np.asarray(values, dtype=float)

    pad_lo = np.arange(out_min, wl.min(), step)
    pad_lo = pad_lo[pad_lo < wl.min()]

    pad_hi = np.arange(wl.max() + step, out_max + step, step)
    pad_hi = pad_hi[pad_hi > wl.max()]

    wl_padded = np.concatenate([pad_lo, wl, pad_hi])
    values_padded = np.concatenate([np.zeros_like(pad_lo), values, np.zeros_like(pad_hi)])

    order = np.argsort(wl_padded)
    return wl_padded[order], values_padded[order]


def resample_a_i_spec_loe(wavelengths = np.arange(400,900), FWHMs = None):
    """
    Specific absorption coefficients [m2 mg-1] of eight phytoplankton types:
    the six standard classes distributed with the Water Color Simulator 6
    (WASI6) [1], plus two additional species-specific spectra (Loé).

    0. phytoplankton (WASI6 typical Lake Constance mixture)
    1. cryptophyta (WASI6)
    2. cyanobacteria (WASI6)
    3. diatoms (WASI6)
    4. dinoflagellates (WASI6)
    5. green algae (WASI6)
    6. CYANO_PC: Microcystis aeruginosa, PC-pigmented cyanobacteria
       (Wojtasiewicz 2016, monoculture; smoothed with a Savitzky-Golay
       filter, window 21 / order 4, by Loé)
    7. DINOFLAGELLATES_G: Gymnodinium sanquineum (Röttgers), normalized to
       a*(440) = 0.0315 m2/mg and rescaled by a factor of 1.6 according to
       Gege (2012), by Loé

    Spectra 6 and 7 only cover part of the requested wavelength range
    (CYANO_PC: 400-800 nm; DINOFLAGELLATES_G: 350-726 nm). Outside their
    measured range they are set to 0 before resampling, so that all eight
    spectra are aligned on the same output wavelength grid.

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

    :param wavelengths: wavelengths to resample the specific absorption coefficients to
    :param FWHMs: full width at half maximum of the output bands (optional)
    :return: specific absorption coefficients of eight phytoplankton types resampled to
             input wavelengths, shape (len(wavelengths), 8), columns in the order listed above
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    out_min, out_max = wavelengths.min(), wavelengths.max()

    resampled_columns = []

    # --- 0-5: six standard WASI6 phytoplankton classes ---
    a_phyto_db = pd.read_csv(data_dir / 'a_phy_spec.txt', skiprows=25, sep=",")
    wl_std = a_phyto_db["wavelength_nm"].values
    std_cols = [c for c in a_phyto_db.columns if c != "wavelength_nm"]

    for col in std_cols:
        wl_pad, val_pad = _pad_spectrum_with_zeros(wl_std, a_phyto_db[col].values, out_min, out_max, step=1)
        band_resampler = BandResampler(wl_pad, wavelengths, fwhm1=None, fwhm2=FWHMs)
        resampled_columns.append(band_resampler(val_pad))

    # --- 6: CYANO_PC (Microcystis aeruginosa, PC-pigmented cyanobacteria) ---
    cyano_pc = pd.read_csv(data_dir / 'CYANO_PC.txt', skiprows=9, sep='\t',
                            names=['wavelength_nm', 'a_spec'], engine='python')
    cyano_pc = cyano_pc.dropna()
    wl_pad, val_pad = _pad_spectrum_with_zeros(
        cyano_pc.wavelength_nm.values.astype(float), cyano_pc.a_spec.values.astype(float),
        out_min, out_max, step=2
    )
    band_resampler = BandResampler(wl_pad, wavelengths, fwhm1=None, fwhm2=FWHMs)
    resampled_columns.append(band_resampler(val_pad))

    # --- 7: DINOFLAGELLATES_G (Gymnodinium sanquineum, Röttgers) ---
    dino_g = pd.read_csv(data_dir / 'DINOFLAGELLATES-G.sanquineum(Rottgers).txt', skiprows=11, sep='\t',
                          names=['wavelength_nm', 'a_spec'], engine='python')
    dino_g = dino_g.dropna()
    wl_pad, val_pad = _pad_spectrum_with_zeros(
        dino_g.wavelength_nm.values.astype(float), dino_g.a_spec.values.astype(float),
        out_min, out_max, step=2
    )
    band_resampler = BandResampler(wl_pad, wavelengths, fwhm1=None, fwhm2=FWHMs)
    resampled_columns.append(band_resampler(val_pad))

    a_i_spec = np.column_stack(resampled_columns)

    return a_i_spec


def resample_b_phy_norm(wavelengths = np.arange(400,800), FWHMs = None):
    """
    Normalized backscattering coefficient of phytoplankton as distributed with the Water Color Simulator 6 (WASI6) [1]
    obtained by fitting a measurement of b_b_phy(lambda) for green algae from Lake Garda in the range from 400 to 900 nm (Giardino, personal communication) 
    and extrapolating the fit curve to the range from 350 to 1000 nm.

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute normalized backscattering coefficient of phytoplankton for
    :return: normalized backscattering coefficient of phytoplankton for input wavelengths
    """
    # READ DATA FROM DATABASE
    b_phy_norm = pd.read_csv(data_dir / 'b_phy_norm.txt', skiprows=6, sep="\t")
    # RESAMPLE TO WAVELENGTHS
    band_resampler = BandResampler(b_phy_norm.wavelength_nm.values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    b_phy_norm = band_resampler(b_phy_norm["bb_phy_norm"])
    
    return b_phy_norm


### CDOM

def resample_a_Y_norm(wavelengths = np.arange(400,800), FWHMs = None):
    """
    Gaussian approximation of normalized cdom absorption (Lake Constance)
    """
    a_Y = pd.read_csv(data_dir / 'Y_Gauss.txt', skiprows=11, sep=' ', usecols=[0,1])
    # resample to sensor bands
    band_resampler = BandResampler(a_Y.nm.values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    a_Y_res = band_resampler(a_Y["dimensionless"])
    return a_Y_res
