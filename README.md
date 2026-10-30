# miniwasi

Vectorised WASI-type bio-optical model for deep water (SIOPs from the WASI7 database files
plus CYANO_PC and DINOFLAGELLATES_GSANQUINEUM) with a batched, bounded Levenberg-Marquardt inversion using the analytic
Jacobian, an `ImageProcessor` that inverts ENVI BSQ images with it, and a
`BatchImageProcessor` that processes whole folders of images in parallel.

## Install

From this folder, in your conda environment:

```
pip install -e .
```

`-e` (editable) means changes to the code in `miniwasi/` take effect without reinstalling.
Dependencies: numpy, pandas, spectral, matplotlib, joblib, tqdm.

## Use

```python
import numpy as np
from miniwasi import MiniWasi, ImageProcessor

model = MiniWasi(wavelengths=wl, FWHMs=fwhm, sza=41.3, va=5, T=16.2)

# forward model: scalars -> (L,), arrays of shape (N,) -> (N, L)
Rrs = model.forward(C_x=1.0, C_y=0.2, C_3=4.0)

# one spectrum (floats in the result) or many spectra (arrays of shape (N,))
r = model.invert(rrs, weights=w, vary={'C_x': True, 'C_y': True, 'C_3': True}, init={'C_0': 0})
r['params'], r['stderr'], r['residual'], r['success']

# many in-situ spectra with their own geometry / temperature
model.set_conditions(sza=sza_arr, va=va_arr, T=T_arr)
rs = model.invert(R_insitu, weights=w, vary=..., init=...)

# ENVI image (header needs wavelength, fwhm, sza, vza)
ImageProcessor(image_path, out_path, weights=w, vary=..., init=..., output_wcs=[...]).run()
```

See `examples/image_processor_pace.py`.

Many images (e.g. a time series of one lake): images are processed in parallel, each on one
core, with one model per chunk of images (SIOPs resampled once per chunk). Writes one output per
image and `batch_summary.csv`; images whose output exists are skipped, so a run can be resumed.

```python
from miniwasi import BatchImageProcessor

if __name__ == "__main__":
    BatchImageProcessor(
        r"D:\bsq_lakes\geneva", r"D:\fitted_lakes\geneva",
        siops={"bb_phy_spec": 0.001, "T": 12},          # model settings; T is applied per image
        processor_kwargs=dict(weights=w, vary=..., init=..., output_wcs=[...]),
        images=None,                                    # or a list of image names
        n_workers=8, chunk=25,
    ).run()
```

On Windows, scripts that run in parallel (`n_jobs != 1` in ImageProcessor, `n_workers > 1` in
BatchImageProcessor) need an `if __name__ == "__main__":` guard.

## Tests

```
pip install -e .[test]
pytest tests
```

## Credits

Resampling code adapted from the bio_optics package: König, M., Noel, P., Hondula, K.L.,
Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023), https://doi.org/10.5281/zenodo.10246860.
SIOPs and water data: original WASI7 database files in `miniwasi/data` (Gege (2021), The Water
Colour Simulator WASI, user manual for version 6; sources in the file headers), plus CYANO_PC
(Wojtasiewicz 2016) and DINOFLAGELLATES_GSANQUINEUM (Röttgers). The file list and the order of
the phytoplankton classes C_0 ... C_7 are in `miniwasi/resampling.py`.
