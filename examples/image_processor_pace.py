from pathlib import Path
from miniwasi import ImageProcessor

# Fit 5 Parameters: Diatoms, CyanoPC, green algae, CX, CY
# 5mg.m-3 for the Phytoplankton, 0.5 for the CX and 0.3 for the CY
# until 724: weights=[0]*22+[1]*163+[0]*77
# until 800: weights=[0]*22+[1]*202+[0]*38
PROCESSOR_KWARGS = dict(
    weights=[0]*22+[1]*202+[0]*38,
    vary={"C_x": True, "C_y": True, "C_0": False, "C_1": False, "C_2": False, 
          "C_3": True, "C_4": False,  "C_5": True, "C_6": True},
    init={"C_y": 0.3, "C_0": 0, "C_x": 0.5, "C_3": 5, "C_5": 5, "C_6": 5},
    output_wcs=["C_x", "C_y", "C_3", "C_6", "C_5"],
    output_iops=[],
)

# paths relative to this script: input in examples/data, results in examples/output
HERE = Path(__file__).resolve().parent
image_path = str(HERE / "data" / "PACE_ENVI_WASI_interpolated.bsq")
out_dir = HERE / "output"
out_path = str(out_dir / "PACE_5params_test.bsq")
out_path_figure = out_path.replace(".bsq", ".png")

# The guard is needed on Windows if run on multiple cores (n_jobs != 1)
# the parallel inversion starts worker processes
# that import this script again.
if __name__ == "__main__":
    out_dir.mkdir(exist_ok=True)
    processor = ImageProcessor(image_path, out_path, **PROCESSOR_KWARGS, n_jobs=1)
    processor.run()
    processor.plot_results(out_path=out_path_figure)
