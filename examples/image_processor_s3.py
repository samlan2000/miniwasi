from pathlib import Path
from miniwasi import ImageProcessor

# Example varying two phytoplankton classes ("C_3": diatoms, "C_5": green alage)
# for Uroglena sp. bloom
PROCESSOR_KWARGS = dict(
    # recommended weights: 442.5 - 665 nm
    weights=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    vary={"C_x": True, "C_y": True, "C_0": False, "C_1": False, "C_2": False, 
          "C_3": True, "C_4": False,  "C_5": True, "C_6": False},
    init={"C_x": 1, "C_y": 0.1, "C_3": 5, "C_5": 5},
    output_wcs=["C_x", "C_y", "C_3", "C_5"],
    # options: "a", "a_cdom", "a_nap", "a_phy", "bb", "bb_nap", "bb_phy"
    output_iops=[],
)

# paths relative to this script: input in examples/data, results in examples/output
HERE = Path(__file__).resolve().parent
img = str(HERE / "data" / "S3A_20210906T095043.bsq")
out_dir = HERE / "output"
out_path = str(out_dir / Path(img).name)
out_path_figure = out_path.replace(".bsq", ".png")

# The guard is needed on Windows if run on multiple cores (n_jobs != 1)
# the parallel inversion starts worker processes
# that import this script again.
if __name__ == "__main__":
    out_dir.mkdir(exist_ok=True)
    processor = ImageProcessor(img, out_path, **PROCESSOR_KWARGS) # runs on all available cores by default
    processor.run()
    processor.plot_results(out_path=out_path_figure)
