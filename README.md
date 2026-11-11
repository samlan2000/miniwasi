# miniwasi

Vectorised WASI-type bio-optical model for deep water (SIOPs from the WASI7 database files
plus CYANO_PC) with a batched, bounded Levenberg-Marquardt inversion using the analytic
Jacobian, an `ImageProcessor` that inverts ENVI BSQ images with it, and a
`BatchImageProcessor` that processes whole folders of images in parallel.

## Install

Clone the repository and, from its folder, in your (conda) environment:

```
pip install -e .
```

`-e` (editable) means changes to the code in `miniwasi/` take effect without reinstalling.
Dependencies: numpy, pandas, spectral, matplotlib, joblib, tqdm.

## Use

### Model: forward and inversion of spectra

```python
import numpy as np
from miniwasi import MiniWasi

# Sentinel-3 OLCI bands 1-10 [nm]
wl = np.array([400, 412.5, 442.5, 490, 510, 560, 620, 665, 681.25, 708.75])
fwhm = np.array([15, 10, 10, 10, 10, 10, 10, 10, 7.5, 10])
model = MiniWasi(wavelengths=wl, FWHMs=fwhm, sza=41.3, vza=5, T=16.2)

# forward model: scalars -> R_rs of shape (L,), arrays of shape (N,) -> (N, L)
rrs = model.forward(C_x=1.0, C_y=0.2, C_3=4.0)

# inversion of one spectrum (floats in the result) or many spectra (N, L) (arrays of shape (N,));
# weights: 1 = fitted band, 0 = ignored (here 442.5-665 nm)
w = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
r = model.invert(rrs, weights=w, vary={"C_x": True, "C_y": True, "C_3": True},
                 init={"C_x": 0.5, "C_y": 0.1, "C_3": 1.0})
r["params"], r["stderr"], r["residual"], r["success"]

# many in-situ spectra, each with its own geometry and temperature (arrays of shape (N,))
model.set_conditions(sza=sza_arr, vza=vza_arr, T=T_arr)
rs = model.invert(rrs_insitu, weights=w, vary=..., init=...)
```

Parameters: `C_0` ... `C_6` phytoplankton classes [µg/L] (order in `miniwasi/resampling.py`),
`C_x` non-algal particles [mg/L], `C_mie` second particle type [mg/L], `C_y` CDOM absorption at
440 nm [1/m]. SIOP settings (e.g. `bb_nap_spec`, `s_cdom`, `sigma_Lu`, `resampling`) are arguments
of `MiniWasi`.

### One ENVI image

```python
from miniwasi import ImageProcessor

if __name__ == "__main__":
    p = ImageProcessor("S3A_20210906T095043.bsq", "fitted/S3A_20210906T095043.bsq",
                       weights=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                       vary={"C_x": True, "C_y": True, "C_3": True, "C_5": True},
                       init={"C_x": 1, "C_y": 0.1, "C_3": 5, "C_5": 5},
                       output_wcs=["C_x", "C_y", "C_3", "C_5"])
    p.run()
    p.plot_results(out_path="fitted/S3A_20210906T095043.png")
```

The input is an ENVI BSQ image of R_rs [1/sr]. The header needs `wavelength` and `fwhm`; the sun
zenith and viewing angle are read from `sza` and `vza` if present, can be given as
`ImageProcessor(..., sza=..., vza=...)` (overrides the header), and are otherwise set to 30 and 0
deg with a warning. Pixels with NaN, -9999 or all zeros are skipped. The output image has one
band per fitted concentration, `C_phy` (sum of all phytoplankton classes) and the fit residual,
and the angles and processing information in its header.

### Many images

```python
from miniwasi import BatchImageProcessor

if __name__ == "__main__":
    BatchImageProcessor(
        "images", "fitted",
        siops={"T": 12},                    # MiniWasi settings; T is applied per image
        processor_kwargs=dict(weights=w, vary=..., init=..., output_wcs=[...]),
        n_workers=8, chunk=25,
    ).run()
```

Images are processed in parallel, each on one core, with one model per chunk of images (SIOPs
resampled once per chunk and band setup). The output folder gets one result per image and
`batch_summary.csv` (valid / failed pixels, angles used, errors); images whose output exists are
skipped, so an interrupted run can be resumed. Images with different band setups can be mixed if
`weights=None`.

On Windows, scripts that run in parallel (`n_jobs != 1` in ImageProcessor, `n_workers > 1` in
BatchImageProcessor) need an `if __name__ == "__main__":` guard.

### Examples

The [`examples`](https://github.com/samlan2000/miniwasi/tree/main/examples) folder contains small test images in
`examples/data` (five Sentinel-3 OLCI images of Lake Geneva, September 2021, and a PACE OCI
subset) and scripts that write to `examples/output`:

- [`image_processor_s3.py`](https://github.com/samlan2000/miniwasi/blob/main/examples/image_processor_s3.py): one OLCI image, two
  phytoplankton classes (diatoms and green algae) during a Uroglena sp. bloom
- [`image_processor_pace.py`](https://github.com/samlan2000/miniwasi/blob/main/examples/image_processor_pace.py): hyperspectral PACE image,
  four phytoplankton classes
- [`batch_image_processor.py`](https://github.com/samlan2000/miniwasi/blob/main/examples/batch_image_processor.py): all example images with
  `BatchImageProcessor`, once with `weights=None` and once with the OLCI weights (the PACE image
  then fails, which is recorded in `batch_summary.csv`)

## Tests

```
pip install -e .[test]
pytest tests
```

## Credits

SIOPs and water data: original WASI7 database files in `miniwasi/data` (Gege (2021), The Water
Colour Simulator WASI, user manual for version 6; sources in the file headers), plus CYANO_PC
(Wojtasiewicz 2016). The file list and the order of the phytoplankton classes C_0 ... C_6 are in
`miniwasi/resampling.py`.

The data files are resampled to the sensor bands as in WASI (Gaussian band response from the
FWHMs, +- 3 sigma; `resample_Gauss` in WASI 7). `MiniWasi(..., resampling="bandresampler")`
(or `siops={"resampling": "bandresampler"}` in the image processors) reproduces miniwasi <= 0.3,
which used `spectral.BandResampler` as in the bio_optics package: König, M., Noel, P.,
Hondula, K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023),
https://doi.org/10.5281/zenodo.10246860. The method is written to the output header
(`siop resampling`).

## License

The code is released under the MIT License (see [`LICENSE`](https://github.com/samlan2000/miniwasi/blob/main/LICENSE)).

The spectra in `miniwasi/data` are not covered by the MIT License: the WASI7 database files are
(c) Peter Gege (DLR) and are redistributed under the WASI7 terms (free of charge, no commercial
distribution or use without an agreement with the author; publications must quote WASI, cite a
WASI publication and inform the author). `CYANO_PC.A` is from Wojtasiewicz (2016). Details in
[`miniwasi/data/LICENSE_DATA.txt`](https://github.com/samlan2000/miniwasi/blob/main/miniwasi/data/LICENSE_DATA.txt).

