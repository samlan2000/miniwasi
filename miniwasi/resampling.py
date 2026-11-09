"""
Reading the WASI spectrum files in miniwasi/data and resampling them to sensor bands.

The files are the original WASI7 database files [1] (plus one extra phytoplankton spectrum), so
the model uses exactly the spectra of WASI7:

    WATER.A        absorption of pure water at 20 degC [1/m]
    daWdT.txt      temperature gradient of pure water absorption [1/m/degC] (Roettgers et al.
                   2013, as distributed with WASI6)
    Phyto.bN       normalized backscattering of phytoplankton (reference 526 nm)
    Y_Gauss.aN     normalized CDOM absorption, Gaussian model of Gege (2000), Lake Constance
                   (used with MiniWasi(a_norm_y_from_file=True); otherwise exp(-S (l - 440)))

    specific absorption of the phytoplankton classes C_0 ... C_6 [m2/mg], in this order:
    0  PHYTO.A                         mean phytoplankton of Lake Constance (Heege)
    1  CRYPTO.A                        cryptophyta
    2  CYANO_PE.A                      cyanobacteria (phycoerythrin)
    3  DIA.A                           diatoms
    4  DINOFLAGELLATES.A               dinoflagellates
    5  GREEN_ALGAE.A                   green algae
    6  CYANO_PC.A                      extra: Microcystis aeruginosa, PC-pigmented cyanobacteria
                                       (Wojtasiewicz 2016; Savitzky-Golay smoothed by Loé)

WASI files: header lines of any length, then one line per wavelength, "wavelength value"
separated by tabs or spaces. The source details are in the header of each file.

Resampling to the sensor bands (method=...):

    "wasi" (default)  as WASI (resample_Gauss in misc.pas of WASI 7): Gaussian band response
                      with sigma = FWHM / 2.35, evaluated at the wavelengths of the file within
                      |lambda - centre| < 3 sigma; value = sum of y * 1.0025 / (sigma sqrt(2 pi))
                      * exp(-(lambda - centre)^2 / (2 sigma^2)) * dlambda, dlambda = spacing to the
                      previous wavelength of the file (not renormalised; the amplitude 1.0025
                      compensates for the cut-off at 3 sigma, as in WASI).
    "bandresampler"   spectral.BandResampler as in the bio_optics package [2] (miniwasi <= 0.3):
                      Gaussian weights of the file samples within centre +- FWHM/2 only,
                      renormalised to 1.

Bands outside the range of a file are NaN; MiniWasi leaves them out of the fit ("wasi": the
whole +- 3 sigma window must lie within the file; WASI itself would sum the covered part). The
phytoplankton spectra are an exception: they only cover part of the range (e.g. DIA.A
380-817 nm), absorption is taken as 0 outside it.

[1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6 (and WASI7 database).
[2] König et al. (2023): bio_optics python package. https://github.com/CMLandOcean/bio_optics,
    https://doi.org/10.5281/zenodo.10246860
"""

from pathlib import Path

import numpy as np
from spectral import BandResampler
from spectral.algorithms.resampling import build_fwhm

# Data folder inside the package (miniwasi/data)
data_dir = Path(__file__).resolve().parent / "data"

WATER = "WATER.A"
DAW_DT = "daWdT.txt"
BB_PHY_NORM = "Phyto.bN"
CDOM_NORM = "Y_Gauss.aN"
PHYTO = ("PHYTO.A", "CRYPTO.A", "CYANO_PE.A", "DIA.A", "DINOFLAGELLATES.A", "GREEN_ALGAE.A",
         "CYANO_PC.A")
METHODS = ("wasi", "bandresampler")


def read_spectrum(name):
    """
    Wavelengths [nm] and values of a WASI spectrum file in miniwasi/data.
    Data lines are all lines that start with two numbers; everything else is header.
    """
    wl, values = [], []
    with open(data_dir / name, encoding="latin-1") as f:      # WASI files are Windows-1252/Latin-1
        for line in f:
            parts = line.split()
            try:
                x, y = float(parts[0]), float(parts[1])
            except (IndexError, ValueError):
                continue
            wl.append(x)
            values.append(y)
    wl, values = np.array(wl), np.array(values)
    if wl.size < 2 or np.any(np.diff(wl) <= 0):
        raise ValueError(f"{name}: could not read an increasing wavelength column.")
    return wl, values


def _pad_with_zeros(wl, values, out_min, out_max):
    """
    Extend a spectrum with zeros (at its own spacing) down to out_min and up to out_max, so that
    absorption outside the measured range is 0 instead of undefined.

    method "bandresampler": the padding reaches the outermost band centres, not the edge of their
    band response (centre +- FWHM/2), so BandResampler averages the edge bands over the covered
    part (kept so that miniwasi <= 0.3 results are reproduced). Method "wasi" pads to +- 3 sigma.
    """
    step = np.median(np.diff(wl))
    lo = np.arange(wl[0] - step, out_min - step, -step)[::-1]
    hi = np.arange(wl[-1] + step, out_max + step, step)
    return (np.concatenate([lo, wl, hi]),
            np.concatenate([np.zeros(lo.size), values, np.zeros(hi.size)]))


def _resample_wasi(wl, values, centres, fwhms, zero_outside):
    """Gaussian resampling exactly as WASI (see the module docstring)."""
    sigma = fwhms / 2.35
    half = 3 * sigma
    if zero_outside:
        wl, values = _pad_with_zeros(wl, values, np.min(centres - half), np.max(centres + half))
    dl = np.abs(np.diff(wl, prepend=2 * wl[0] - wl[1]))    # first sample: spacing to the next one
    out = np.full(centres.size, np.nan)
    for k, (c, s, h) in enumerate(zip(centres, sigma, half)):
        if c - h < wl[0] or c + h > wl[-1]:
            continue                                        # window not covered by the file: NaN
        sel = np.abs(wl - c) < h
        g = 1.0025 / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wl[sel] - c) / s) ** 2)
        out[k] = np.sum(values[sel] * g * dl[sel])
    return out


def resample(name, wavelengths, FWHMs=None, zero_outside=False, method="wasi"):
    """
    Spectrum of a data file resampled to the sensor bands.

    :param name: file name in miniwasi/data, e.g. resampling.WATER or "DIA.A"
    :param wavelengths: band centres [nm]
    :param FWHMs: band widths [nm] (scalar or one per band); None: from the band spacing
    :param zero_outside: 0 instead of NaN for bands outside the range of the file
    :param method: "wasi" (Gaussian as in WASI, default) or "bandresampler" (miniwasi <= 0.3)
    :return: array (len(wavelengths),)
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}.")
    wavelengths = np.asarray(wavelengths, dtype=float)
    wl, values = read_spectrum(name)
    if FWHMs is not None and np.ndim(FWHMs) == 0:
        FWHMs = np.full(wavelengths.size, float(FWHMs))
    if method == "wasi":
        fwhms = np.asarray(build_fwhm(wavelengths) if FWHMs is None else FWHMs, dtype=float)
        return _resample_wasi(wl, values, wavelengths, fwhms, zero_outside)
    if zero_outside:
        wl, values = _pad_with_zeros(wl, values, wavelengths.min(), wavelengths.max())
    return BandResampler(wl, wavelengths, fwhm1=None, fwhm2=FWHMs)(values)


def resample_phyto(wavelengths, FWHMs=None, method="wasi"):
    """Specific absorption of the 7 phytoplankton classes, shape (len(wavelengths), 7), order of PHYTO."""
    return np.column_stack([resample(name, wavelengths, FWHMs, zero_outside=True, method=method)
                            for name in PHYTO])
