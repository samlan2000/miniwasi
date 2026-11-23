import inspect
import os
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError

import numpy as np
from spectral import envi
from spectral.algorithms.resampling import build_fwhm

from .MiniWASI import MiniWasi

IOP_NAMES = ("a", "a_cdom", "a_nap", "a_phy", "bb", "bb_nap", "bb_phy")
UNITS = {**{f"C_{i}": "µg/L" for i in range(7)}, "C_phy": "µg/L", "C_x": "mg/L", "C_mie": "mg/L",
         "C_y": "1/m"}


# ---------------------------------------------------------------------------
# Image processor
# ---------------------------------------------------------------------------

# MiniWasi constructor arguments accepted as **model_kwargs (wavelengths and FWHMs are named arguments)
MODEL_ARGS = set(inspect.signature(MiniWasi.__init__).parameters) - {"self", "wavelengths", "FWHMs"}
# conditions that change per image: also allowed together with model=...
CONDITIONS = ("T", "sza", "vza")


def check_model_kwargs(model_kwargs, caller, model=None):
    """Raise a clear error for arguments that are not MiniWasi arguments (e.g. typos)."""
    unknown = set(model_kwargs) - MODEL_ARGS
    if unknown:
        raise TypeError(f"{caller}: unknown argument(s) {sorted(unknown)}. Accepted: the {caller} arguments "
                        f"and the MiniWasi arguments {sorted(MODEL_ARGS)}.")
    if model is not None and set(model_kwargs) - set(CONDITIONS):
        raise ValueError(f"{caller}: with model=..., only {list(CONDITIONS)} can be given; set "
                         f"{sorted(set(model_kwargs) - set(CONDITIONS))} when creating the model.")


def resolve_bands(header, wavelengths=None, FWHMs=None, source=""):
    """
    Band centres and widths [nm] of an image: from the header ("wavelength", "fwhm"), else the
    given values, else (FWHMs only) half the distance to the neighbouring bands, as
    MiniWasi(FWHMs=None) would use. Given values never override the header (a warning is
    issued if they differ from it).
    Returns wavelengths, FWHMs (arrays, one value per band) and {"wavelength": src, "fwhm": src}
    with src "header", "header (given values differ)", "argument" or "band spacing".
    """
    n = int(header["bands"])
    values, sources = {}, {}
    for key, given in (("wavelength", wavelengths), ("fwhm", FWHMs)):
        if key in header:
            value = np.array([float(w) for w in header[key]])
            if value.size != n:
                raise ValueError(f"{source}: header has {value.size} {key} values for {n} bands.")
            sources[key] = "header"
            if given is not None:
                g = np.asarray(given, dtype=float)
                if g.shape not in ((), (n,)) or not np.allclose(g, value):
                    sources[key] = "header (given values differ)"
                    warnings.warn(f"{source}: the given {key} values differ from the header; the header is used.")
        elif given is not None:
            g = np.asarray(given, dtype=float)
            if g.shape != (n,) and not (key == "fwhm" and g.ndim == 0):
                raise ValueError(f"{source}: {g.size} {key} values given for {n} bands.")
            value = np.broadcast_to(g, (n,)).copy()
            sources[key] = "argument"
        elif key == "wavelength":
            raise ValueError(f"{source}: no 'wavelength' in the header and no wavelengths given.")
        else:
            value = np.asarray(build_fwhm(values["wavelength"]), dtype=float)
            sources[key] = "band spacing"
            warnings.warn(f"{source}: no 'fwhm' in the header and no FWHMs given; using half the distance "
                          f"to the neighbouring bands.")
        values[key] = value
    return values["wavelength"], values["fwhm"], sources


class ImageProcessor():
    """
    Invert an ENVI image of R_rs [1/sr] pixel by pixel with MiniWasi and write the results as ENVI BSQ.

    img_path:     ENVI image (.bsq, .bil or .bip; the interleave is read from the header), with
                  the header next to it (same name, .hdr). The header needs "wavelength" and "fwhm".
    out_path:     output image; always written as ENVI BSQ, so the extension is set to .bsq
    wavelengths, FWHMs: band centres / widths [nm], only used if the header has no "wavelength" /
                  "fwhm" (the header always wins; a warning is issued if they differ). FWHMs may be
                  one number for all bands. Without fwhm in header and argument, half the distance
                  to the neighbouring bands is used (with a warning).

    Inversion (as in MiniWasi.invert):
    weights, vary, init, bounds, max_iter, tol
    block_size, n_jobs: pixels are fitted in blocks of block_size, on n_jobs cores (-1 = all, default)
    progress:     tqdm progress bar over the blocks

    Output:
    output_wcs:   fitted concentrations written as bands (C_phy, the sum of all phytoplankton
                  classes, the fit residual and the relative residual are always written; relative
                  residual = residual / mean R_rs over the fitted bands, e.g. keep pixels <= 0.2)
    output_iops:  IOP spectra written as bands ("a", "a_cdom", "a_nap", "a_phy", "bb", "bb_nap", "bb_phy")

    Model:
    **model_kwargs: any MiniWasi argument (e.g. T, resampling, bb_phy_spec, s_cdom, sza, vza);
                  the bands are given by the header (or wavelengths / FWHMs above).
                  sza, vza [deg]: read from the header ("sza", "vza"); the given values are only used if
                  the header has none (the header always wins; a warning is issued if they differ).
                  Neither: 30 and 0 deg are used with a warning.
    model:        optional MiniWasi object to use instead of creating one (skips resampling the
                  SIOPs, e.g. for many images with the same bands). Its wavelengths must match the
                  image; only T, sza and vza may then be given, and are set on the model.
    """

    def __init__(self, img_path, out_path, *, wavelengths=None, FWHMs=None, weights=None, vary=None, init=None, bounds=None,
                 max_iter=200, tol=1e-12, block_size=2000, n_jobs=1, progress=True,
                 output_wcs=None, output_iops=None, model=None, **model_kwargs):
        check_model_kwargs(model_kwargs, "ImageProcessor", model)
        self.img_path = img_path
        self.wavelengths_in, self.FWHMs_in = wavelengths, FWHMs
        self.out_path = os.path.splitext(out_path)[0] + ".bsq"     # output is always BSQ
        self.weights = weights
        self.vary = dict(vary or {"C_0": True, "C_x": True, "C_y": True})
        self.bounds = dict(bounds or {"C_y": (0, 10), "C_0": (0, 100), "C_x": (0, 50)})
        self.init = dict(init or {"C_y": 0.1, "C_0": 3, "C_x": 1})
        self.max_iter, self.tol = max_iter, tol
        # C_phy is always written (after the concentrations), so it is not listed here
        self.output_wcs = sorted(set(output_wcs or ["C_0", "C_x", "C_y"]) - {"C_phy"})
        self.output_iops = sorted(output_iops if output_iops is not None else [])
        unknown = set(self.output_iops) - set(IOP_NAMES)
        if unknown:
            raise ValueError(f"Unknown output_iops: {sorted(unknown)}")
        # None means "not given" (header / model value is used)
        self.model_kwargs = {k: v for k, v in model_kwargs.items() if not (k in CONDITIONS and v is None)}
        self.sza = self.model_kwargs.pop("sza", None)
        self.vza = self.model_kwargs.pop("vza", None)
        self.model_in = model
        self.block_size = block_size
        self.n_jobs = n_jobs
        self.progress = progress
        self.results = None

    def run(self):
        """
        Process an ENVI image (BSQ, BIL or BIP) with the batched WASI inversion and write ENVI BSQ output
        using georeference from input header.
        """
        t_start = time.perf_counter()
        hdr_path = os.path.splitext(self.img_path)[0] + ".hdr"

        # -------------------------------------------------
        # Read ENVI header (metadata + georeference) and cube
        # -------------------------------------------------
        header = envi.read_envi_header(hdr_path)
        wavelengths, FWHMs, self.band_source = resolve_bands(header, self.wavelengths_in, self.FWHMs_in,
                                                             self.img_path)
        # geometry [deg]: ENVI header ("sza = 31" or "sza = {31}"), else given value, else default
        self.defaulted = []                 # angles for which the default was used
        self.angles = angles = {}
        self.angle_source = {}
        for key, given, default in (("sza", self.sza, 30.0), ("vza", self.vza, 0.0)):
            if key in header:
                value = header[key][0] if isinstance(header[key], list) else header[key]
                angles[key] = float(value.strip("{} "))
                self.angle_source[key] = "header"
                if given is not None and not np.isclose(float(given), angles[key]):
                    self.angle_source[key] = "header (given values differ)"
                    warnings.warn(f"{self.img_path}: the given {key} ({float(given):g} deg) differs from the header "
                                  f"({angles[key]:g} deg); the header is used.")
            elif given is not None:
                angles[key] = float(given)
                self.angle_source[key] = "argument"
            else:
                angles[key] = default
                self.angle_source[key] = "default"
                self.defaulted.append(key)
                warnings.warn(f"{self.img_path}: '{key}' neither in the header nor given; using {default:g} deg.")
        sza, vza = angles["sza"], angles["vza"]

        with warnings.catch_warnings():     # NaN outside the lake is expected
            warnings.filterwarnings("ignore", "Image data contains NaN values.")
            cube = np.asarray(envi.open(hdr_path, self.img_path).load())  # (lines, samples, bands)
        lines, samples, bands = cube.shape

        # -------------------------------------------------
        # Create WASI object
        # -------------------------------------------------
        if self.model_in is None:
            self.model = MiniWasi(wavelengths=wavelengths, FWHMs=FWHMs, vza=vza, sza=sza, **self.model_kwargs)
        else:
            # reuse the model (SIOPs already resampled): only update geometry / temperature
            self.model = self.model_in
            if (len(self.model.wavelengths) != len(wavelengths)
                    or not np.allclose(self.model.wavelengths, wavelengths)):
                raise ValueError(f"model wavelengths {self.model.wavelengths} do not match the image "
                                 f"wavelengths {wavelengths} ({self.img_path}).")
            self.model.set_conditions(sza=sza, vza=vza, T=self.model_kwargs.get("T"))

        # fitted concentrations + C_phy + one spectrum per IOP + residual + relative residual
        n_wcs = len(self.output_wcs)
        n_scalar = n_wcs + 1
        self.n_bands_out = n_scalar + len(self.output_iops) * len(wavelengths) + 2

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
                                         max_iter=self.max_iter, tol=self.tol,
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
            out_valid[:, n_wcs] = sum(self.fit["params"][f"C_{i}"] for i in range(7))
            if self.output_iops:
                # forward() with the fitted concentrations sets the IOPs, shape (n_valid, L)
                self.model.forward(**self.fit["params"])
                out_valid[:, n_scalar:-2] = np.concatenate([getattr(self.model, name) for name in self.output_iops], axis=1)
            out_valid[:, -2] = self.fit["residual"]
            fitted = np.ones(len(wavelengths), bool) if self.weights is None else np.asarray(self.weights) > 0
            out_valid[:, -1] = self.fit["residual"] / pixels[valid][:, fitted].mean(axis=1)
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
        out_hdr["band names"] = self.output_wcs + ["C_phy"] + expanded_iop_names + ["residual", "relative_residual"]
        # scalars and residuals get dummy wavelengths
        out_hdr["wavelength"] = [0] * n_scalar + wavelengths.tolist() * len(self.output_iops) + [0, 0]
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
        out_hdr["siop resampling"] = self.model.resampling
        out_hdr["processing time total (s)"] = round(t_total, 3)
        out_hdr["processing time inversion (s)"] = round(t_inv, 3)
        out_hdr["valid pixels"] = n_valid
        out_hdr["time per pixel (ms)"] = round(self.timing["ms_per_pixel"], 4)
        out_hdr["pixels not converged"] = n_not_converged
        out_hdr["pixels failed"] = n_failed
        out_hdr["n_jobs"] = self.n_jobs
        out_hdr["input wavelength"] = wavelengths.tolist()     # bands used for the inversion
        out_hdr["input fwhm"] = FWHMs.tolist()
        out_hdr["band source"] = ", ".join(f"{k}: {v}" for k, v in self.band_source.items())
        out_hdr["angle source"] = ", ".join(f"{k}: {v}" for k, v in self.angle_source.items())
        out_hdr["sza"] = [sza]              # angles used for the inversion
        out_hdr["vza"] = [vza]

        # -------------------------------------------------
        # Write ENVI BSQ
        # -------------------------------------------------
        envi.save_image(
            os.path.splitext(self.out_path)[0] + ".hdr",
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
