from bsqConverterPolymer import convert_polymer_batch

# -----------------------
# Adjust paths
# -----------------------
basepath_in = r"D:\greifensee"  # external hard drive (contains the run_*/L2COMBINE folders)
basepath_out_polymer = r"D:\bsq"

# Clip extent as WKT (lon lat order); the product is clipped to the bounding box of this geometry.
# Set to None to keep the full alpinespace grid.
clip_wkt = "POLYGON((8.643 47.383, 8.716 47.383, 8.716 47.312, 8.643 47.312, 8.643 47.383))"  # Lake Varese

convert_polymer_batch(basepath_in, basepath_out_polymer,
                      clip_to_extent=clip_wkt,
                      remove_bad_pixels=True,
                      bands=['Oa1', 'Oa2', 'Oa3', 'Oa4', 'Oa5', 'Oa6', 'Oa7', 'Oa8', 'Oa10', 'Oa11', 'Oa12', 'Oa16', 'Oa17', 'Oa21'])
