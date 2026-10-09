from bsqConverterPolymer import convert_polymer_batch

# -----------------------
# Adjust paths
# -----------------------
basepath_in = r"D:\nc"  # external hard drive (contains the run_*/L2COMBINE folders)
basepath_out_polymer = r"D:\bsq_full_scenes_unfiltered"

# Clip extent as WKT (lon lat order); the product is clipped to the bounding box of this geometry.
# Set to None to keep the full alpinespace grid.
clip_wkt = "POLYGON((8.6997 45.7853, 8.8033 45.7853, 8.8033 45.8446, 8.6997 45.8446, 8.6997 45.7853))"  # Lake Varese

convert_polymer_batch(basepath_in, basepath_out_polymer,
                      clip_to_extent=False,
                      remove_bad_pixels=False,
                      bands=['Oa1', 'Oa2', 'Oa3', 'Oa4', 'Oa5', 'Oa6', 'Oa7', 'Oa8', 'Oa10', 'Oa11', 'Oa12', 'Oa16', 'Oa17', 'Oa21'])
