"""
Batch processing of many ENVI images with ImageProcessor.

Images are processed in parallel (n_workers images at once). Each image is inverted on a single
core in one block, which is fastest for lake-sized images (up to some 10^4 valid pixels). The
images are split into chunks of `chunk` images; within a chunk, one MiniWasi model is created per
band setup (wavelengths + FWHMs from the header), so the SIOP files are resampled once per setup
and chunk, not for every image. Images with different band setups (e.g. OLCI and MSI) can be in
the same folder; note that `weights` in processor_kwargs must then fit all of them (or be None).
Per image only the geometry (sza, vza from the header) and the water temperature are updated.
No plots are made.

Output: out_dir/<image>.bsq/.hdr (as ImageProcessor) and out_dir/batch_summary.csv with one row
per image (valid pixels, failed / not converged pixels, times, error message). The summary is
appended to, so an interrupted run can be resumed with skip_existing=True.

On Windows, scripts that use n_workers > 1 need an  if __name__ == "__main__":  guard.
"""

import contextlib
import csv
import glob
import io
import os
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from spectral import envi
from tqdm import tqdm

from .ImageProcessor import ImageProcessor
from .MiniWASI import MiniWasi

class BatchImageProcessor():
    """
    Invert all ENVI BSQ images of a folder (or a given list of images) with ImageProcessor.

    in_dir:           folder with the input .bsq/.hdr files
    out_dir:          output folder (created if needed)
    siops:            MiniWasi settings (bb_nap_spec, a_spec_nap_440nm, bb_phy_spec, s_cdom,
                      s_nap, n, a_norm_y_from_file, T). "T" is applied per image, all others
                      when the model is created.
    processor_kwargs: ImageProcessor arguments (weights, vary, bounds, init, output_wcs,
                      output_iops, block_size); not siops, model, n_jobs or progress
    images:           optional list of image names or paths (with or without .bsq); default:
                      all .bsq files in in_dir
    skip_existing:    skip images whose output .hdr already exists (resume)
    n_workers:        images processed at the same time (joblib processes)
    chunk:            images per task; one model per task
    """

    def __init__(self, in_dir, out_dir, siops=None, processor_kwargs=None,
                 images=None, skip_existing=True, n_workers=8, chunk=25):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.model_kwargs = dict(siops or {})
        self.T = self.model_kwargs.pop("T", None)      # per image (ImageProcessor), not in the model
        self.processor_kwargs = {"block_size": 10_000, **(processor_kwargs or {})}
        self.images = images
        self.skip_existing = skip_existing
        self.n_workers = n_workers
        self.chunk = chunk
        self.summary = None

    # ---------------------------------------------------------------------
    def _process_chunk(self, images):
        """Invert a list of images with a shared model. Runs in a worker process."""
        warnings.filterwarnings("ignore")
        rows, models = [], {}                   # one model per band setup
        siops = {} if self.T is None else {"T": self.T}
        for img in images:
            name = os.path.splitext(os.path.basename(img))[0]
            out = os.path.join(self.out_dir, name + ".bsq")
            row = dict(image=name, valid_pixels=0, failed=0, not_converged=0,
                       inversion_s=np.nan, total_s=np.nan, error="")
            t0 = time.perf_counter()
            try:
                hdr = envi.read_envi_header(img[:-4] + ".hdr")
                wl = np.array([float(w) for w in hdr["wavelength"]])
                fwhm = np.array([float(w) for w in hdr["fwhm"]])
                setup = (tuple(wl), tuple(fwhm))
                if setup not in models:
                    models[setup] = MiniWasi(wavelengths=wl, FWHMs=fwhm, **self.model_kwargs)
                model = models[setup]
                p = ImageProcessor(img, out, n_jobs=1, progress=False, model=model, siops=siops,
                                   **self.processor_kwargs)
                with contextlib.redirect_stdout(io.StringIO()):     # keep the console readable
                    p.run()
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
            paths = sorted(glob.glob(os.path.join(self.in_dir, "*.bsq")))
        else:
            paths = [p if os.path.isabs(p) else os.path.join(self.in_dir, p) for p in self.images]
            paths = [p if p.endswith(".bsq") else p + ".bsq" for p in paths]
        if self.skip_existing:
            paths = [p for p in paths if not os.path.exists(
                os.path.join(self.out_dir, os.path.basename(p)[:-4] + ".hdr"))]
        return paths

    # ---------------------------------------------------------------------
    def run(self):
        """Process all pending images. Returns a dict with totals (also in self.summary)."""
        os.makedirs(self.out_dir, exist_ok=True)
        images = self._pending_images()
        chunks = [images[i:i + self.chunk] for i in range(0, len(images), self.chunk)]
        print(f"{len(images)} images in {len(chunks)} chunks, {self.n_workers} workers")
        stats = dict(images=0, pixels=0, inversion_s=0.0, failed_pixels=0, errors=0, wall_s=0.0)
        if not chunks:
            self.summary = stats
            return stats

        summary_path = os.path.join(self.out_dir, "batch_summary.csv")
        write_header = not os.path.exists(summary_path)
        t0 = time.perf_counter()
        results = Parallel(n_jobs=self.n_workers, return_as="generator_unordered")(
            delayed(self._process_chunk)(chunk) for chunk in chunks)
        with open(summary_path, "a", newline="") as f, tqdm(total=len(images), unit="img") as bar:
            writer = None
            for rows in results:
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
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
                bar.update(len(rows))

        stats["wall_s"] = time.perf_counter() - t0
        self.summary = stats
        print(f"Done in {stats['wall_s'] / 60:.1f} min: {stats['images']} images, {stats['pixels']} valid pixels, "
              f"{stats['failed_pixels']} failed pixels, {stats['errors']} images with errors. Summary: {summary_path}")
        if stats["errors"]:
            warnings.warn(f"{stats['errors']} images could not be processed; see the 'error' column in {summary_path}.")
        return stats
