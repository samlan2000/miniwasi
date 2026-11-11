from pathlib import Path
import pandas as pd
from miniwasi import BatchImageProcessor

# Batch processing of all images in examples/data (5 Sentinel-3 OLCI images of Lake Geneva and
# one PACE OCI image), in two configurations:
#   1) weights=None: all bands of each image are fitted, so images with different band setups
#      (OLCI: 10 bands, PACE: 262 bands) can be processed in the same batch
#   2) the OLCI weights of image_processor_s3.py (442.5 - 665 nm): these only fit images with
#      10 bands, so the PACE image fails; the error is recorded in batch_summary.csv and the
#      batch continues with the other images
# Results: examples/output/batch_weights_none and examples/output/batch_weights_s3

HERE = Path(__file__).resolve().parent
in_dir = HERE / "data"

COMMON = dict(
    vary={"C_x": True, "C_y": True, "C_0": False, "C_1": False, "C_2": False,
          "C_3": True, "C_4": False, "C_5": True, "C_6": False},
    init={"C_x": 1, "C_y": 0.1, "C_3": 5, "C_5": 5},
    output_wcs=["C_x", "C_y", "C_3", "C_5"],
    output_iops=[],
)
CONFIGS = {
    "batch_weights_none": dict(weights=None, **COMMON),
    "batch_weights_s3": dict(weights=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0], **COMMON),   # as image_processor_s3.py
}

# The guard is needed on Windows: the parallel processing starts worker processes
# that import this script again.
if __name__ == "__main__":
    for name, kwargs in CONFIGS.items():
        out_dir = HERE / "output" / name
        print(f"\n=== {name} ===")
        batch = BatchImageProcessor(str(in_dir), str(out_dir), processor_kwargs=kwargs,
                                    skip_existing=False,     # always reprocess (example)
                                    # n_workers: number of images processed at the same time (one
                                    #   process per worker, each image inverted on one core)
                                    # chunk: number of images per task. The images are split into
                                    #   chunks; each worker takes one chunk at a time and creates the
                                    #   MiniWasi model (SIOPs resampled to the image bands) only once
                                    #   per chunk and band setup, then reuses it for all images of the
                                    #   chunk. Larger chunks: less model set-up and process overhead;
                                    #   smaller chunks: work is spread more evenly over the workers.
                                    #   For hundreds of images, ~25 (the default) works well; here 3,
                                    #   so that the 6 example images are shared between the 2 workers.
                                    n_workers=2, chunk=3)
        batch.run()
        summary = pd.read_csv(out_dir / "batch_summary.csv").drop_duplicates("image", keep="last")
        print(summary[["image", "valid_pixels", "failed", "not_converged", "sza", "vza", "default_angles", "error"]]
              .fillna("").to_string(index=False))
