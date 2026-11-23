"""
Batch processing of many ENVI images with ImageProcessor.

Images are processed in parallel (n_workers images at once). Each image is inverted on a single
core in one block, which is fastest for lake-sized images (up to some 10^4 valid pixels). The
images are split into chunks of `chunk` images; within a chunk, one MiniWasi model is created per
band setup (wavelengths + FWHMs of the image), so the SIOP files are resampled once per setup
and chunk, not for every image. Images with different band setups (e.g. OLCI and MSI) can be in
the same folder; note that `weights` must then fit all of them (or be None).
Per image only the geometry and the water temperature are updated. Geometry per image: the "sza" / "vza"
entries of the header, else the sza / vza argument (one value for all images, or a dict
{image name: value}), else 30 / 0 deg. T can also be one value or a dict per image. The angles used are written to batch_summary.csv (column default_angles:
angles for which the default was used), and one warning at the end counts these images.
No plots are made.

Input: ENVI images (.bsq, .bil or .bip, each with its .hdr). Output: always ENVI BSQ,
out_dir/<image>.bsq/.hdr (as ImageProcessor) and out_dir/batch_summary.csv with one row
per image (valid pixels, failed / not converged pixels, times, error message). The summary is
appended to, so an interrupted run can be resumed with skip_existing=True.

On Windows, scripts that use n_workers > 1 need an  if __name__ == "__main__":  guard.
"""

import contextlib
import csv
import glob
import inspect
import io
import os
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from spectral import envi
from tqdm import tqdm

from .ImageProcessor import ImageProcessor, CONDITIONS, check_model_kwargs, resolve_bands
from .MiniWASI import MiniWasi

SUMMARY_COLUMNS = ["image", "valid_pixels", "failed", "not_converged", "inversion_s", "total_s",
                   "sza", "vza", "default_angles", "input_notes", "error"]


ENVI_EXTENSIONS = (".bsq", ".bil", ".bip")      # input images; output is always .bsq
# batch_summary.csv, column input_notes: bands / angles not from the header, or header differing from
# the given values (angles set to the default are in the column default_angles)
INPUT_NOTES = {"default": None, "argument": "argument", "band spacing": "band_spacing",
               "header (given values differ)": "header_differs_from_given"}
T_DEFAULT = inspect.signature(MiniWasi.__init__).parameters["T"].default


class BatchImageProcessor():
    """
    Invert all ENVI images of a folder (or a given list of images) with ImageProcessor.

    in_dir:           folder with the input ENVI images (.bsq, .bil or .bip, each with a .hdr)
    out_dir:          output folder (created if needed)
    wavelengths, FWHMs: band centres / widths [nm] for images whose header has no "wavelength" /
                      "fwhm" (as in ImageProcessor; the header always wins). Bands and angles that do
                      not come from the header, or headers that differ from the given values, are
                      noted in the column input_notes of batch_summary.csv.

    Inversion and output, as in ImageProcessor:
    weights, vary, init, bounds, max_iter, tol, block_size (default 10 000: lake images in one block),
    output_wcs, output_iops. Each image is inverted on one core (no n_jobs / progress).

    Batch:
    images:           optional list of image names or paths (with or without extension); default:
                      all .bsq, .bil and .bip files in in_dir
    skip_existing:    skip images whose output .hdr already exists (resume)
    n_workers:        images processed at the same time (joblib processes)
    chunk:            images per task; one model per task and band setup

    Model:
    **model_kwargs:   any MiniWasi argument (e.g. T, resampling, bb_phy_spec, sza, vza).
                      sza, vza [deg]: read from each header; the given values are only used for images
                      without them in the header (the header always wins): a number = for all these
                      images, a dict {image name (without extension): value} = per image. Neither:
                      30 / 0 deg.
                      T [deg C]: a number for all images or a dict {image name: value}; images not in
                      the dict get the MiniWasi default.
                      All other arguments are applied when the model is created.
    """

    def __init__(self, in_dir, out_dir, *, wavelengths=None, FWHMs=None, weights=None, vary=None, init=None, bounds=None,
                 max_iter=200, tol=1e-12, block_size=10_000, output_wcs=None, output_iops=None,
                 images=None, skip_existing=True, n_workers=8, chunk=25, **model_kwargs):
        check_model_kwargs(model_kwargs, "BatchImageProcessor")
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.processor_kwargs = dict(wavelengths=wavelengths, FWHMs=FWHMs, weights=weights, vary=vary, init=init, bounds=bounds,
                                     max_iter=max_iter, tol=tol, block_size=block_size,
                                     output_wcs=output_wcs, output_iops=output_iops)
        # per-image conditions (number or {image: value}); everything else is fixed in the model
        self.conditions = {k: model_kwargs.pop(k) for k in CONDITIONS if k in model_kwargs}
        self.model_kwargs = model_kwargs
        self.images = images
        self.skip_existing = skip_existing
        self.n_workers = n_workers
        self.chunk = chunk
        self.summary = None

    # ---------------------------------------------------------------------
    def _process_chunk(self, images):
        """Invert a list of images with a shared model. Runs in a worker process."""
        # per-image warnings are summarised in batch_summary.csv / at the end of run(); catch_warnings
        # restores the filters afterwards (with n_workers=1 this runs in the calling process)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._process_chunk_quiet(images)

    def _process_chunk_quiet(self, images):
        rows, models = [], {}                   # one model per band setup
        for img in images:
            name = os.path.splitext(os.path.basename(img))[0]
            out = os.path.join(self.out_dir, name + ".bsq")
            row = dict(image=name, valid_pixels=0, failed=0, not_converged=0,
                       inversion_s=np.nan, total_s=np.nan, sza=np.nan, vza=np.nan, default_angles="",
                       input_notes="", error="")
            # per-image or common value; the model is shared, so a missing T falls back to the default
            conditions = {k: v.get(name) if isinstance(v, dict) else v for k, v in self.conditions.items()}
            if conditions.get("T") is None:
                conditions["T"] = T_DEFAULT
            t0 = time.perf_counter()
            try:
                hdr = envi.read_envi_header(os.path.splitext(img)[0] + ".hdr")
                wl, fwhm, _ = resolve_bands(hdr, self.processor_kwargs["wavelengths"],
                                            self.processor_kwargs["FWHMs"], img)
                setup = (tuple(wl), tuple(fwhm))
                if setup not in models:
                    models[setup] = MiniWasi(wavelengths=wl, FWHMs=fwhm, **self.model_kwargs)
                model = models[setup]
                p = ImageProcessor(img, out, n_jobs=1, progress=False, model=model,
                                   **conditions, **self.processor_kwargs)
                with contextlib.redirect_stdout(io.StringIO()):     # keep the console readable
                    p.run()
                row.update(sza=p.angles["sza"], vza=p.angles["vza"], default_angles=" ".join(p.defaulted),
                           input_notes=" ".join(f"{k}={INPUT_NOTES[v]}"
                                                for k, v in {**p.band_source, **p.angle_source}.items()
                                                if v != "header" and INPUT_NOTES[v]))
                row.update(valid_pixels=int(p.valid.sum()), inversion_s=p.timing["inversion_s"])
                if p.fit is not None:
                    failed = np.isnan(p.fit["residual"])
                    row.update(failed=int(failed.sum()), not_converged=int(np.sum(~p.fit["success"] & ~failed)))
            except Exception as e:                                  # one bad file must not stop the batch
                row["error"] = repr(e)
            row["total_s"] = time.perf_counter() - t0
            rows.append(row)
        return rows

    def _pending_images(self):
        """Full paths of the input images, minus those with an existing output if skip_existing."""
        if self.images is None:
            if not os.path.isdir(self.in_dir):
                raise FileNotFoundError(f"Input folder not found: {self.in_dir}")
            paths = sorted(p for ext in ENVI_EXTENSIONS for p in glob.glob(os.path.join(self.in_dir, "*" + ext)))
            if not paths:
                warnings.warn(f"No ENVI images (.bsq, .bil, .bip) found in {self.in_dir}")
        else:
            paths = []
            for p in self.images:
                p = p if os.path.isabs(p) else os.path.join(self.in_dir, p)
                if os.path.splitext(p)[1].lower() not in ENVI_EXTENSIONS:     # name without extension
                    found = [p + ext for ext in ENVI_EXTENSIONS if os.path.exists(p + ext)]
                    p = found[0] if found else p + ".bsq"                   # missing: error in the summary
                paths.append(p)
        # one entry per image name (the output is <name>.bsq, so the same name twice would collide)
        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        unique = {}
        for p in paths:
            if stem(p) in unique:
                warnings.warn(f"{p} skipped: {unique[stem(p)]} has the same name.")
            else:
                unique[stem(p)] = p
        paths = list(unique.values())
        if self.skip_existing:
            paths = [p for p in paths if not os.path.exists(os.path.join(self.out_dir, stem(p) + ".hdr"))]
        return paths

    # ---------------------------------------------------------------------
    def run(self):
        """Process all pending images. Returns a dict with totals (also in self.summary)."""
        os.makedirs(self.out_dir, exist_ok=True)
        images = self._pending_images()
        chunks = [images[i:i + self.chunk] for i in range(0, len(images), self.chunk)]
        print(f"{len(images)} images in {len(chunks)} chunks, {self.n_workers} workers")
        stats = dict(images=0, pixels=0, inversion_s=0.0, failed_pixels=0, errors=0, default_geometry=0,
                     input_notes=0, wall_s=0.0)
        if not chunks:
            self.summary = stats
            return stats

        summary_path = os.path.join(self.out_dir, "batch_summary.csv")
        if os.path.exists(summary_path):
            # a summary written by another miniwasi version (other columns) is kept under a new name
            with open(summary_path, newline="") as f:
                old_cols = next(csv.reader(f), [])
            if old_cols != SUMMARY_COLUMNS:
                stamp = time.strftime("%Y%m%dT%H%M%S")
                os.replace(summary_path, summary_path.replace(".csv", f"_old_{stamp}.csv"))
        write_header = not os.path.exists(summary_path)
        t0 = time.perf_counter()
        results = Parallel(n_jobs=self.n_workers, return_as="generator_unordered")(
            delayed(self._process_chunk)(chunk) for chunk in chunks)
        with open(summary_path, "a", newline="") as f, tqdm(total=len(images), unit="img") as bar:
            writer = None
            for rows in results:
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
                    if write_header:
                        writer.writeheader()
                writer.writerows(rows)
                f.flush()                      # keep the CSV usable if the run dies
                for r in rows:
                    stats["images"] += 1
                    stats["pixels"] += r["valid_pixels"]
                    stats["inversion_s"] += np.nan_to_num(r["inversion_s"])
                    stats["failed_pixels"] += r["failed"]
                    stats["errors"] += bool(r["error"])
                    stats["default_geometry"] += bool(r["default_angles"])
                    stats["input_notes"] += bool(r["input_notes"])
                bar.update(len(rows))

        stats["wall_s"] = time.perf_counter() - t0
        self.summary = stats
        print(f"Done in {stats['wall_s'] / 60:.1f} min: {stats['images']} images, {stats['pixels']} valid pixels, "
              f"{stats['failed_pixels']} failed pixels, {stats['errors']} images with errors. Summary: {summary_path}")
        if stats["default_geometry"]:
            warnings.warn(f"{stats['default_geometry']} images had no sza and/or vza in the header and none was given: "
                          f"default sza = 30 deg / vza = 0 deg used; see the 'default_angles' column in {summary_path}.")
        if stats["input_notes"]:
            warnings.warn(f"{stats['input_notes']} images had bands or angles that are not (all) in the header, or a "
                          f"header that differs from the given values; see the 'input_notes' "
                          f"column in {summary_path}.")
        if stats["errors"]:
            warnings.warn(f"{stats['errors']} images could not be processed; see the 'error' column in {summary_path}.")
        return stats
