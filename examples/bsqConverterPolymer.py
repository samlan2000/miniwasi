import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.transform import rowcol
import os
import re


class bsqConverterPolymer():

    def __init__(self, ds, out_bsq, clip_to_extent=None, remove_bad_pixels=True,
                 bands=['Oa1', 'Oa2', 'Oa3', 'Oa4', 'Oa5', 'Oa6', 'Oa7', 'Oa8', 'Oa10']):
        """
        ds                : xarray Dataset with Polymer Rw* bands, sza/vza and (for masking) IDEPix pixel_classif_flags
        out_bsq           : output path of the ENVI .bsq file
        clip_to_extent    : None, or a WKT string (e.g. "POLYGON((lon lat, lon lat, ...))");
                            the product is clipped to the bounding box of that geometry
        remove_bad_pixels : mask every pixel with any IDEPix flag set
        """
        self.ds = self._clip(clip_to_extent, ds) if clip_to_extent else ds

        self.lon, self.lat = self._get_coords()
        self.idepix_mask = self._get_idepix_mask() if remove_bad_pixels else None
        if self.idepix_mask is None:
            print("Warning: No idepix masking!")

        try:
            data, wavelengths, fwhm = self._extract_bands(bands)
        except KeyError as e:
            print("At least one band not matched:")
            print(e)
            print(f"Skipping image {out_bsq}")
            return

        if not np.isfinite(data).any():
            print(f"No valid pixels (outside swath / clip extent, or fully masked). Skipping image {out_bsq}")
            return

        # geometry averaged over the (clipped) scene
        sza = str(np.nanmean(self.ds["sza"].values))
        vza = str(np.nanmean(self.ds["vza"].values))

        self._build_bsq(out_bsq, data, wavelengths, fwhm, sza, vza)

    # ------------------------------------------------------------------------------------------
    # Clipping
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _wkt_bounds(wkt):
        """
        Returns (min_lon, min_lat, max_lon, max_lat) of a 2D WKT geometry (POLYGON, MULTIPOLYGON, ...).
        WKT coordinate order is 'lon lat'.
        """
        nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", wkt)
        if len(nums) < 4 or len(nums) % 2:
            raise ValueError(f"Could not parse 2D coordinates from WKT: {wkt}")
        xy = np.array(nums, dtype=float).reshape(-1, 2)
        return xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()

    def _clip(self, clip_to_extent, ds):
        """
        Clips the ds to the bounding box of the WKT geometry (index slicing, so the grid stays regular).
        """
        min_lon, min_lat, max_lon, max_lat = self._wkt_bounds(clip_to_extent)

        if "lat" in ds.variables and "lon" in ds.variables and ds["lat"].ndim == 1 and ds["lon"].ndim == 1:
            # regular 1D grid axes (L2COMBINE files)
            lat, lon = ds["lat"].values, ds["lon"].values
            rows = np.where((lat >= min_lat) & (lat <= max_lat))[0]
            cols = np.where((lon >= min_lon) & (lon <= max_lon))[0]
            lat_dim, lon_dim = ds["lat"].dims[0], ds["lon"].dims[0]
        else:
            # fallback: 2D latitude/longitude
            lat, lon = ds["latitude"].values, ds["longitude"].values
            inside = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
            rows = np.where(inside.any(axis=1))[0]
            cols = np.where(inside.any(axis=0))[0]
            lat_dim, lon_dim = ds["latitude"].dims

        if rows.size == 0 or cols.size == 0:
            raise ValueError("Clip extent does not overlap the product.")

        return ds.isel({lat_dim: slice(rows.min(), rows.max() + 1),
                        lon_dim: slice(cols.min(), cols.max() + 1)})

    # ------------------------------------------------------------------------------------------
    # Masking / bands / coordinates
    # ------------------------------------------------------------------------------------------
    def _get_idepix_mask(self):
        """True where any IDEPix flag is set (cloud, land, coastline, ...)."""
        if "pixel_classif_flags" not in self.ds.variables:
            print("Warning: 'pixel_classif_flags' not found in product.")
            return None
        # _FillValue = 0 (= no flag set) is decoded to NaN by xarray -> fill back with 0
        return self.ds["pixel_classif_flags"].fillna(0).astype("int32").values > 0

    def _extract_bands(self, bands):
        # [centre wvl, fwhm, alias]
        band_LUT = {
            'Oa1':  [400, 15, "Rw400"],
            'Oa2':  [412.5, 10, "Rw412"],
            'Oa3':  [442.5, 10, "Rw443"],
            'Oa4':  [490, 10, "Rw490"],
            'Oa5':  [510, 10, "Rw510"],
            'Oa6':  [560, 10, "Rw560"],
            'Oa7':  [620, 10, "Rw620"],
            'Oa8':  [665, 10, "Rw665"],
            'Oa9':  False,
            'Oa10': [681.25, 7.5, "Rw681"],
            'Oa11': [708.75, 10, "Rw709"],
            'Oa12': [753.75, 7.5, "Rw754"],
            'Oa13': False,
            'Oa14': False,
            'Oa15': False,
            'Oa16': [778.75, 15, "Rw779"],
            'Oa17': [865, 20, "Rw865"],
            'Oa18': False,
            'Oa19': False,
            'Oa20': False,
            'Oa21': [1029, 40, "Rw1020"]
        }
        band_LUT = {k: v for k, v in band_LUT.items() if k in bands and v}

        # Stack into numpy array (bands, y, x)
        data = np.stack([self.ds[v[2]].values for v in band_LUT.values()])
        # Water leaving reflectance to Rrs conversion
        data = data / np.pi

        # Mask invalid pixels using IDEPIX
        if self.idepix_mask is not None:
            if self.idepix_mask.shape != data.shape[1:]:
                raise ValueError(
                    f"IDEPix mask shape {self.idepix_mask.shape} "
                    f"does not match data shape {data.shape[1:]}"
                )
            data[:, self.idepix_mask] = np.nan

        wavelengths = [v[0] for v in band_LUT.values()]
        fwhm = [v[1] for v in band_LUT.values()]

        return data, wavelengths, fwhm

    def _get_coords(self):
        # L2COMBINE files: 2D latitude/longitude are NaN outside the swath, so their min/max give the
        # swath extent instead of the grid extent. Use the regular 1D grid axes (lat/lon) when available.
        if "lon" in self.ds.variables and "lat" in self.ds.variables \
                and self.ds["lon"].ndim == 1 and self.ds["lat"].ndim == 1:
            lon, lat = np.meshgrid(self.ds["lon"].values, self.ds["lat"].values)
        else:
            lon, lat = self.ds["longitude"].values, self.ds["latitude"].values

        return lon, lat

    # ------------------------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------------------------
    def _build_bsq(self, out_bsq, data, wavelengths, fwhm, sza, vza):

        # lat/lon are pixel CENTRES (nan-safe for 2D lat/lon with fill values)
        min_lon, max_lon = np.nanmin(self.lon), np.nanmax(self.lon)
        min_lat, max_lat = np.nanmin(self.lat), np.nanmax(self.lat)

        height = self.lon.shape[0]
        width  = self.lon.shape[1]

        # from_bounds expects the outer pixel EDGES -> extend by half a pixel on each side
        half_dx = (max_lon - min_lon) / (width - 1) / 2 if width > 1 else 0
        half_dy = (max_lat - min_lat) / (height - 1) / 2 if height > 1 else 0
        self.transform = from_bounds(min_lon - half_dx, min_lat - half_dy,
                                     max_lon + half_dx, max_lat + half_dy, width, height)

        crs = "EPSG:4326"

        # ----------------------------------------------------------------------------------
        # WRITE ENVI BSQ FILE
        # ----------------------------------------------------------------------------------
        profile = {
            "driver": "ENVI",
            "dtype": "float32",
            "count": data.shape[0],
            "height": height,
            "width": width,
            "crs": crs,
            "transform": self.transform,
            "interleave": "band"
        }

        with rasterio.open(out_bsq, "w", **profile) as dst:
            dst.write(data.astype(np.float32))

        print("BSQ file written:", out_bsq)

        # ==============================
        # Reference stations (lat, lon)
        # ==============================
        stations = {
        }

        # ==============================
        # Convert lon/lat to row/col
        # ==============================
        station_indices = {}

        for name, (lat, lon) in stations.items():

            # Convert coordinates (note: rowcol expects lon, lat)
            row, col = rowcol(self.transform, lon, lat)

            # Check raster bounds
            if (0 <= row < height) and (0 <= col < width):
                idx_str = f"{row}, {col}"
            else:
                idx_str = "NaN, NaN"

            station_indices[name] = idx_str

        # ----------------------------------------------------------------------------------
        # MODIFY THE HDR TO ADD WAVELENGTH INFORMATION
        # ----------------------------------------------------------------------------------
        hdr_file = out_bsq.replace(".bsq", ".hdr")

        with open(hdr_file, "a") as hdr:
            hdr.write("\nwavelength units = Nanometers\n")

            hdr.write("wavelength = {\n")
            hdr.write(", ".join(str(w) for w in wavelengths))
            hdr.write("}\n\n")

            hdr.write("fwhm = {\n")
            hdr.write(", ".join(str(s) for s in fwhm))
            hdr.write("}\n\n")

            hdr.write("sza = {\n")
            hdr.write(str(sza))
            hdr.write("}\n\n")

            hdr.write("vza = {\n")
            hdr.write(str(vza))
            hdr.write("}\n\n")

            # ==============================
            # Write all station indices
            # ==============================
            for station, idx in station_indices.items():
                hdr.write(f"{station.lower()}_idx = {{\n")
                hdr.write(str(idx))
                hdr.write("}\n\n")

            print("HDR updated with wavelength and observation geometry information:", hdr_file)


# ==============================================================================================
# Batch conversion
# ==============================================================================================
def _product_alias(product):
    """e.g. 'L2COMBINE_L1P_reproj_S3B_OL_1_EFR____20211005T100000_...' -> 'S3B_20211005T100000'"""
    m = re.search(r"(S3[AB])_OL_1_EFR_+(\d{8}T\d{6})", product)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    # fallback: old naming logic
    return product.split("_")[3] + "_" + product.split("____")[-1][:15]


def _iter_products(basepath_in):
    """
    Yields (product_path, idepix_path_or_None) for every product found in basepath_in.

    Supported folder structures:
      1) External drive (Polymer + IDEPix combined in one file):
           <basepath_in>/run_<...>/L2COMBINE/L2COMBINE_L1P_reproj_<...>.nc        -> idepix_path = None
      2) Old local structure (separate files):
           <basepath_in>/<product_folder>/L2POLY/L2POLY_reproj_NASA_<...>.nc
           <basepath_in>/<product_folder>/L1P/L1P_reproj_<...>.nc                  -> idepix_path = L1P file
    """
    for product_folder in sorted(os.listdir(basepath_in)):
        product_folder_path = os.path.join(basepath_in, product_folder)
        # skip files and system folders ($RECYCLE.BIN, System Volume Information, ...)
        if not os.path.isdir(product_folder_path) or product_folder.startswith(("$", "System Volume")):
            continue
        try:
            subfolders = os.listdir(product_folder_path)
        except PermissionError:
            continue

        if "L2COMBINE" in subfolders:
            combine_path = os.path.join(product_folder_path, "L2COMBINE")
            for product in sorted(os.listdir(combine_path)):
                if product.endswith(".nc"):
                    yield os.path.join(combine_path, product), None

        if "L2POLY" in subfolders:
            L2_folder_path = os.path.join(product_folder_path, "L2POLY")
            for product in sorted(os.listdir(L2_folder_path)):
                if product.endswith(".nc"):
                    idepix_equivalent = product.replace("L2POLY_reproj_NASA", "L1P_reproj")
                    idepix_path = os.path.join(product_folder_path, "L1P", idepix_equivalent)
                    yield (os.path.join(L2_folder_path, product),
                           idepix_path if os.path.exists(idepix_path) else None)


def convert_polymer_batch(basepath_in, basepath_out, clip_to_extent=None, remove_bad_pixels=True,
                          bands=['Oa1', 'Oa2', 'Oa3', 'Oa4', 'Oa5', 'Oa6', 'Oa7', 'Oa8', 'Oa10'],
                          overwrite=False):
    os.makedirs(basepath_out, exist_ok=True)
    products = list(_iter_products(basepath_in))
    print(f"Found {len(products)} products in {basepath_in}\n")

    for n, (product_path, idepix_path) in enumerate(products, 1):
        product = os.path.basename(product_path)
        acolite_alias = _product_alias(product)
        path_out = os.path.join(basepath_out, acolite_alias + ".bsq")

        if not overwrite and os.path.exists(path_out):
            print(f"[{n}/{len(products)}] {acolite_alias} already exists, skipping.\n")
            continue

        print(f"[{n}/{len(products)}] Converting product {acolite_alias}...")
        try:
            with xr.open_dataset(product_path) as ds:
                if idepix_path is not None and remove_bad_pixels:
                    # old structure: copy the IDEPix flags from the separate L1P file into ds
                    with xr.open_dataset(idepix_path) as ds_idepix:
                        ds = ds.assign(pixel_classif_flags=(
                            ds["latitude"].dims, ds_idepix["pixel_classif_flags"].values))
                bsqConverterPolymer(ds, path_out, clip_to_extent=clip_to_extent,
                                    remove_bad_pixels=remove_bad_pixels, bands=bands)
        except Exception as e:
            print(f"Failed to convert {product}: {e}")
        print("")
