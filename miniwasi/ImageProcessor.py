import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError

import numpy as np
from spectral import envi

from .MiniWASI import MiniWasi

IOP_NAMES = ("a", "a_cdom", "a_nap", "a_phy", "bb", "bb_nap", "bb_phy")
UNITS = {**{f"C_{i}": "µg/L" for i in range(8)}, "C_phy": "µg/L", "C_x": "mg/L", "C_mie": "mg/L",
         "C_y": "1/m"}


# ---------------------------------------------------------------------------
# Image processor
# ---------------------------------------------------------------------------

class ImageProcessor():

    def __init__(self, img_path, out_path, weights=None, vary=None, bounds=None,
                 init=None, output_wcs=None, output_iops=None, siops=None,
                 block_size=2000, n_jobs=-1, progress=True, model=None):

        self.img_path = img_path
        self.out_path = out_path
        self.weights = weights
        self.vary = dict(vary or {"C_0": True, "C_x": True, "C_y": True})
        self.bounds = dict(bounds or {"C_y": (0, 1), "C_0": (0, 15), "C_x": (0, 50)})
        self.init = dict(init or {"C_y": 0.1, "C_0": 3, "C_x": 1})
        # C_phy is always written (after the concentrations), so it is not listed here
        self.output_wcs = sorted(set(output_wcs or ["C_0", "C_x", "C_y"]) - {"C_phy"})
        self.output_iops = sorted(output_iops if output_iops is not None else [])
        unknown = set(self.output_iops) - set(IOP_NAMES)
        if unknown:
            raise ValueError(f"Unknown output_iops: {sorted(unknown)}")
        self.siops = dict(siops or {})
        if model is not None and set(self.siops) - {"T"}:
            raise ValueError("With model=..., siops may only contain 'T'; set "
                             f"{sorted(set(self.siops) - {'T'})} when creating the model.")
        self.model_in = model
        self.block_size = block_size
        self.n_jobs = n_jobs
        self.progress = progress
        self.results = None

    def run(self):
        """
        Process ENVI BSQ image with the batched WASI inversion and write ENVI BSQ output
        using georeference from input header.
        """
        t_start = time.perf_counter()
        hdr_path = self.img_path.replace(".bsq", ".hdr")

        # -------------------------------------------------
        # Read ENVI header (metadata + georeference) and cube
        # -------------------------------------------------
        header = envi.read_envi_header(hdr_path)
        wavelengths = np.array([float(w) for w in header["wavelength"]])
        FWHMs = np.array([float(w) for w in header["fwhm"]])
        sza = float(header["sza"][0])
        vza = float(header["vza"][0])

        with warnings.catch_warnings():     # NaN outside the lake is expected
            warnings.filterwarnings("ignore", "Image data contains NaN values.")
            cube = np.asarray(envi.open(hdr_path, self.img_path).load())  # (lines, samples, bands)
        lines, samples, bands = cube.shape

        # -------------------------------------------------
        # Create WASI object
        # -------------------------------------------------
        if self.model_in is None:
            self.model = MiniWasi(wavelengths=wavelengths, FWHMs=FWHMs, va=vza, sza=sza, **self.siops)
        else:
            # reuse the model (SIOPs already resampled): only update geometry / temperature
            self.model = self.model_in
            if (len(self.model.wavelengths) != len(wavelengths)
                    or not np.allclose(self.model.wavelengths, wavelengths)):
                raise ValueError(f"model wavelengths {self.model.wavelengths} do not match the image "
                                 f"wavelengths {wavelengths} ({self.img_path}).")
            self.model.set_conditions(sza=sza, va=vza, T=self.siops.get("T"))

        # fitted concentrations + C_phy + one spectrum per IOP + residual band
        n_wcs = len(self.output_wcs)
        n_scalar = n_wcs + 1
        self.n_bands_out = n_scalar + len(self.output_iops) * len(wavelengths) + 1

        # -------------------------------------------------
        # Invert all valid pixels at once
        # -------------------------------------------------
        pixels = cube.reshape(-1, bands)
        valid = np.all(np.isfinite(pixels), axis=1) & ~np.all(pixels == 0, axis=1) & ~np.any(pixels == -9999, axis=1)
        self.valid = valid.reshape(lines, samples)
        print(f"Starting batched inversion of {valid.sum()} valid pixels...")

        flat = np.full((lines * samples, self.n_bands_out), np.nan, dtype=np.float32)
        self.fit = None
        t_inv = 0.0
        n_failed = n_not_converged = 0
        if valid.any():
            t_inv = time.perf_counter()
            self.fit = self.model.invert(pixels[valid], weights=self.weights, vary=self.vary,
                                         init=self.init, bounds=self.bounds,
                                         block_size=self.block_size, n_jobs=self.n_jobs,
                                         progress=self.progress)
            t_inv = time.perf_counter() - t_inv
            # failed = numerical failure (NaN results); not converged = stopped at max_iter
            n_failed = int(np.sum(np.isnan(self.fit["residual"])))
            n_not_converged = int(np.sum(~self.fit["success"])) - n_failed
            if n_not_converged:
                warnings.warn(f"{n_not_converged} pixels did not converge within max_iter.")
            if n_failed:
                warnings.warn(f"{n_failed} pixels failed numerically (written as NaN).")

            out_valid = np.zeros((valid.sum(), self.n_bands_out), dtype=np.float32)
            for k, name in enumerate(self.output_wcs):
                out_valid[:, k] = self.fit["params"][name]
            # total phytoplankton: sum of all classes (fixed ones at their init value)
            out_valid[:, n_wcs] = sum(self.fit["params"][f"C_{i}"] for i in range(8))
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
        out_hdr["band names"] = self.output_wcs + ["C_phy"] + expanded_iop_names + ["residual"]
        # scalars and residual get dummy wavelengths
        out_hdr["wavelength"] = [0] * n_scalar + wavelengths.tolist() * len(self.output_iops) + [0]
        out_hdr.pop("fwhm", None)

        # processing information (wall-clock times; "total" excludes writing the file itself)
        try:
            pkg_version = version("miniwasi")
        except PackageNotFoundError:
            pkg_version = "unknown"
        n_valid = int(valid.sum())
        t_total = time.perf_counter() - t_start
        self.timing = {
            "total_s": t_total,
            "inversion_s": t_inv,
            "valid_pixels": n_valid,
            "ms_per_pixel": 1e3 * t_inv / n_valid if n_valid else 0.0,
        }
        out_hdr["processing date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out_hdr["miniwasi version"] = pkg_version
        out_hdr["processing time total (s)"] = round(t_total, 3)
        out_hdr["processing time inversion (s)"] = round(t_inv, 3)
        out_hdr["valid pixels"] = n_valid
        out_hdr["time per pixel (ms)"] = round(self.timing["ms_per_pixel"], 4)
        out_hdr["pixels not converged"] = n_not_converged
        out_hdr["pixels failed"] = n_failed
        out_hdr["n_jobs"] = self.n_jobs

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
        print(f"Saved ENVI BSQ: {self.out_path}  "
              f"(total {t_total:.1f} s, inversion {t_inv:.1f} s, {n_valid} valid pixels)")

        return self.results

    def plot_results(self, out_path=None):
        import matplotlib.pyplot as plt     # only needed here: processing does not load matplotlib

        def pclip(a, low=2, high=98):
            return np.nanpercentile(a, [low, high])   # failed pixels are NaN

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
