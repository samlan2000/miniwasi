"""
ImageProcessor: inverts every valid pixel of an ENVI BSQ image with miniwasi.MiniWasi
and writes the results as ENVI BSQ (no lmfit needed).

This class only handles image I/O: read the ENVI cube, mask invalid pixels, call
model.invert() on all valid pixels at once, get the IOPs from model.forward(), write the
ENVI output, plot. All physics and fitting live in the model. After run(), self.fit holds
the result dict of model.invert() for the valid pixels (params, stderr, residual,
success, n_iter; arrays in the order of the valid pixels) and self.valid the pixel mask.

Same constructor arguments, run() and plot_results() as the original lmfit-based
ImageProcessor. Differences:
  * `minimizer` and `use_previous_init` are accepted but ignored (always LM, fixed init).
  * Bands with zero weight are dropped before fitting (same minimum, less work).
  * The output ENVI file is written with a .bsq extension (ImageProcessor writes .img).
  * Extra arguments: block_size (spectra per block), n_jobs (cores, -1 = all),
    progress (tqdm progress bar over blocks, shown when there is more than one block).
The "residual" band has the same definition as in the original ImageProcessor.
"""

import warnings

import numpy as np
from spectral import envi
import matplotlib.pyplot as plt

from .MiniWASI import MiniWasi

IOP_NAMES = ("a", "a_cdom", "a_nap", "a_phy", "bb", "bb_nap", "bb_phy")
UNITS = {**{f"C_{i}": "µg/L" for i in range(8)}, "C_x": "mg/L", "C_mie": "mg/L", "C_y": "1/m"}


# ---------------------------------------------------------------------------
# Image processor
# ---------------------------------------------------------------------------

class ImageProcessor():

    def __init__(self, img_path, out_path, weights=None, vary=None, bounds=None,
                 init=None, output_wcs=None, output_iops=None,
                 use_previous_init=False, minimizer="leastsq", siops=None,
                 block_size=2000, n_jobs=-1, progress=True):

        self.img_path = img_path
        self.out_path = out_path
        self.weights = weights
        self.vary = dict(vary or {"C_0": True, "C_x": True, "C_y": True})
        self.bounds = dict(bounds or {"C_y": (0, 1), "C_0": (0, 15), "C_x": (0, 50)})
        self.init = dict(init or {"C_y": 0.1, "C_0": 3, "C_x": 1})
        self.output_wcs = sorted(output_wcs or ["C_0", "C_x", "C_y"])
        self.output_iops = sorted(output_iops if output_iops is not None else [])
        unknown = set(self.output_iops) - set(IOP_NAMES)
        if unknown:
            raise ValueError(f"Unknown output_iops: {sorted(unknown)}")
        if use_previous_init:
            warnings.warn("use_previous_init is ignored by ImageProcessor.")
        if minimizer != "leastsq":
            warnings.warn("minimizer is ignored by ImageProcessor (always Levenberg-Marquardt).")
        self.siops = dict(siops or {})
        self.block_size = block_size
        self.n_jobs = n_jobs
        self.progress = progress
        self.results = None

    def run(self):
        """
        Process ENVI BSQ image with the batched WASI inversion and write ENVI BSQ output
        using georeference from input header.
        """
        hdr_path = self.img_path.replace(".bsq", ".hdr")

        # -------------------------------------------------
        # Read ENVI header (metadata + georeference) and cube
        # -------------------------------------------------
        header = envi.read_envi_header(hdr_path)
        wavelengths = np.array([float(w) for w in header["wavelength"]])
        FWHMs = np.array([float(w) for w in header["fwhm"]])
        sza = float(header["sza"][0])
        vza = float(header["vza"][0])

        cube = np.asarray(envi.open(hdr_path, self.img_path).load())  # (lines, samples, bands)
        lines, samples, bands = cube.shape

        # -------------------------------------------------
        # Create WASI object
        # -------------------------------------------------
        self.model = MiniWasi(wavelengths=wavelengths, FWHMs=FWHMs, va=vza, sza=sza, **self.siops)

        # fitted concentrations + one spectrum per IOP + residual band
        n_scalar = len(self.output_wcs)
        self.n_bands_out = n_scalar + len(self.output_iops) * len(wavelengths) + 1

        # -------------------------------------------------
        # Invert all valid pixels at once
        # -------------------------------------------------
        pixels = cube.reshape(-1, bands)
        valid = np.all(np.isfinite(pixels), axis=1) & ~np.all(pixels == 0, axis=1) & ~np.any(pixels == -9999, axis=1)
        self.valid = valid.reshape(lines, samples)
        print(f"Starting batched inversion of {valid.sum()} valid pixels...")

        flat = np.zeros((lines * samples, self.n_bands_out), dtype=np.float32)
        self.fit = None
        if valid.any():
            self.fit = self.model.invert(pixels[valid], weights=self.weights, vary=self.vary,
                                         init=self.init, bounds=self.bounds,
                                         block_size=self.block_size, n_jobs=self.n_jobs,
                                         progress=self.progress)
            if not np.all(self.fit["success"]):
                warnings.warn(f"{np.sum(~self.fit['success'])} pixels did not converge within max_iter.")

            out_valid = np.zeros((valid.sum(), self.n_bands_out), dtype=np.float32)
            for k, name in enumerate(self.output_wcs):
                out_valid[:, k] = self.fit["params"][name]
            if self.output_iops:
                # forward() with the fitted concentrations sets the IOPs, shape (n_valid, L)
                self.model.forward(**self.fit["params"])
                out_valid[:, n_scalar:-1] = np.concatenate([getattr(self.model, name) for name in self.output_iops], axis=1)
            out_valid[:, -1] = self.fit["residual"]
            flat[valid] = out_valid
        self.results = flat.reshape(lines, samples, self.n_bands_out)

        # -------------------------------------------------
        # Build output ENVI header
        # -------------------------------------------------
        out_hdr = header.copy()
        expanded_iop_names = [f"{name}_{int(wl)}" for name in self.output_iops for wl in wavelengths]

        out_hdr["samples"] = samples
        out_hdr["lines"] = lines
        out_hdr["bands"] = self.n_bands_out
        out_hdr["interleave"] = "bsq"
        out_hdr["data type"] = 4  # float32
        out_hdr["band names"] = self.output_wcs + expanded_iop_names + ["residual"]
        # scalars and residual get dummy wavelengths
        out_hdr["wavelength"] = [0] * n_scalar + wavelengths.tolist() * len(self.output_iops) + [0]
        out_hdr.pop("fwhm", None)

        # -------------------------------------------------
        # Write ENVI BSQ
        # -------------------------------------------------
        envi.save_image(
            self.out_path.replace(".bsq", ".hdr"),
            self.results,
            dtype=np.float32,
            interleave="bsq",
            ext=".bsq",
            metadata=out_hdr,
            force=True,
        )
        print("Saved ENVI BSQ:", self.out_path)

        return self.results

    def plot_results(self, out_path=None):

        def pclip(a, low=2, high=98):
            return np.percentile(a, [low, high])

        n = len(self.output_wcs)
        fig, axs = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axs = [axs]

        for k, (ax, param) in enumerate(zip(axs, self.output_wcs)):
            data = self.results[:, :, k]
            vmin, vmax = pclip(data)
            im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f"{param} [{UNITS.get(param, '')}]")
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=300)
            print(f".png of inversion results saved at: {out_path}")
        plt.show()


# Old name, kept so existing scripts keep working
ImageProcessorExperimental = ImageProcessor
