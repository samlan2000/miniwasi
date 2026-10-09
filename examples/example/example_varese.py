from miniwasi import ImageProcessor
import os

# Example varying three phytoplankton classes ("C_3": diatoms, "C_2": cyanobacteria with PE pigment feature, "C_6": cyanobacteria with PC pigment feature)
PROCESSOR_KWARGS = dict(
    # example weights
    weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    vary={"C_x": True, "C_y": True, "C_0": False, "C_1": False, "C_2": True, 
          "C_3": True, "C_4": False,  "C_5": False, "C_6": True, "C_7": False},
    init={"C_x": 3, "C_y": 0.2, "C_0": 0, "C_1": 0, "C_2": 5, "C_3": 5, "C_4": 0, "C_5": 0, "C_6": 5, "C_7": 0},
    output_wcs=["C_x", "C_y", "C_2", "C_3", "C_6"],
    # options: "a", "a_cdom", "a_nap", "a_phy", "bb", "bb_nap", "bb_phy"
    output_iops=[],
)

###############
# CHANGE PATHS!
###############
image_folder = r"D:\bsq"
out_folder = r"D:\fitted_v2"

# The guard is needed on Windows: the parallel inversion starts worker processes
# that import this script again.
if __name__ == "__main__":
    # batch process the whole folder
    for img in os.listdir(image_folder):
        if img.endswith(".bsq"):
            image_path = os.path.join(image_folder, img)
            out_path = os.path.join(out_folder, img)
            out_path_figure = out_path.replace(".bsq", ".png")
            processor = ImageProcessor(image_path, out_path, **PROCESSOR_KWARGS)
            processor.run()
            processor.plot_results(out_path=out_path_figure)
