from miniwasi import ImageProcessor

# DONE // Fit 6 Parameters: Diatoms, CyanoPC, green algae, Cryptophyta, CX, CY
# Fit 5 Parameters: Diatoms, CyanoPC, green algae, CX, CY
# 5mg.m-3 for the Phytoplankton, 0.5 for the CX and 0.3 for the CY
# until 724: weights=[0]*22+[1]*163+[0]*77
# until 800: weights=[0]*22+[1]*202+[0]*38
PROCESSOR_KWARGS = dict(
    weights=[0]*22+[1]*202+[0]*38,
    vary={"C_x": True, "C_y": True, "C_0": False, "C_1": True, "C_2": False, 
          "C_3": True, "C_4": False,  "C_5": True, "C_6": True, "C_7": False},
    init={"C_y": 0.3, "C_0": 0, "C_x": 0.5, "C_1": 5, "C_3": 5, "C_5": 5, "C_6": 5, "C_7": 0},
    output_wcs=["C_x", "C_y", "C_3", "C_1", "C_6", "C_5"],
    output_iops=[],
)

image_path = r"C:\WASI7\DATA\demo\tmp\PACE_ENVI_WASI_interpolated.bsq"
out_path = r"C:\WASI7\DATA\demo\tmp\PACE_6params_test.bsq"
out_path_figure = out_path.replace(".bsq", ".png")

# The guard is needed on Windows: the parallel inversion starts worker processes
# that import this script again.
if __name__ == "__main__":
    processor = ImageProcessor(image_path, out_path, **PROCESSOR_KWARGS)
    processor.run()
    processor.plot_results(out_path=out_path_figure)
