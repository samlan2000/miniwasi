"""
Regression tests for miniwasi (synthetic spectra, plus a comparison with WASI7 forward spectra
in miniwasi/data/tests). Run with:  pytest tests
"""

from pathlib import Path

import numpy as np
import pytest

from miniwasi import MiniWasi, ImageProcessor, BatchImageProcessor

NAMES = ['C_x', 'C_mie', 'C_y'] + [f'C_{i}' for i in range(7)]
WL = np.arange(400, 800, 5.0)


@pytest.fixture(scope="module")
def model():
    return MiniWasi(wavelengths=WL, FWHMs=np.full(WL.size, 5.0), sza=35, vza=12, T=14)


def random_params(n, seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.uniform(0, 3, n) for k in NAMES}


def test_forward_shapes(model):
    assert model.forward(C_x=1, C_y=0.2).shape == (WL.size,)
    assert model.forward(**random_params(7)).shape == (7, WL.size)
    assert model.a_phy.shape == (7, WL.size)


def test_forward_vectorised_equals_loop(model):
    P = random_params(10)
    R = model.forward(**P)
    for n in range(10):
        np.testing.assert_allclose(R[n], model.forward(**{k: P[k][n] for k in NAMES}), rtol=1e-12)


def test_per_observation_conditions():
    rng = np.random.default_rng(1)
    sza, vza, T = rng.uniform(20, 60, 5), rng.uniform(0, 30, 5), rng.uniform(5, 25, 5)
    P = random_params(5)
    R = MiniWasi(wavelengths=WL, sza=sza, vza=vza, T=T).forward(**P)
    for n in range(5):
        single = MiniWasi(wavelengths=WL, sza=sza[n], vza=vza[n], T=T[n])
        np.testing.assert_allclose(R[n], single.forward(**{k: P[k][n] for k in NAMES}), rtol=1e-12)


def test_jacobian_matches_finite_differences(model):
    P = random_params(4, seed=2)
    model.forward(**P)
    J = model.jacobian(NAMES)
    h = 1e-6
    for k, name in enumerate(NAMES):
        up = model.forward(**{n: P[n] + h * (n == name) for n in NAMES})
        down = model.forward(**{n: P[n] - h * (n == name) for n in NAMES})
        np.testing.assert_allclose(J[..., k], (up - down) / (2 * h), rtol=1e-5, atol=1e-10)


VARY = {"C_x": True, "C_y": True, "C_3": True}
INIT = {"C_0": 0, "C_x": 0.5, "C_y": 0.3, "C_3": 5}


def test_invert_single_recovers_truth(model):
    truth = dict(C_x=1.3, C_y=0.25, C_3=4.0)
    r = model.invert(model.forward(C_0=0, **truth), vary=VARY, init=INIT)
    assert isinstance(r['params']['C_x'], float) and r['success']
    for k, v in truth.items():
        assert r['params'][k] == pytest.approx(v, rel=1e-6)


def test_invert_many_equals_single(model):
    rng = np.random.default_rng(3)
    P = dict(C_0=0, C_x=rng.uniform(.3, 3, 25), C_y=rng.uniform(.05, .4, 25), C_3=rng.uniform(0, 8, 25))
    R = model.forward(**P) * (1 + 0.01 * rng.normal(size=(25, WL.size)))
    w = np.r_[np.zeros(5), np.ones(WL.size - 10), np.zeros(5)]
    many = model.invert(R, weights=w, vary=VARY, init=INIT, block_size=10)
    for n in range(25):
        one = model.invert(R[n], weights=w, vary=VARY, init=INIT)
        for k in VARY:
            assert many['params'][k][n] == pytest.approx(one['params'][k], rel=1e-6, abs=1e-8)


def test_cdom_shape_from_file():
    m = MiniWasi(wavelengths=WL, a_norm_y_from_file=True)
    assert np.all(np.isfinite(m.a_norm_y))


def test_image_processor(tmp_path, model):
    from spectral import envi
    rng = np.random.default_rng(4)
    P = dict(C_0=0, C_x=rng.uniform(.3, 3, 12), C_y=rng.uniform(.05, .4, 12), C_3=rng.uniform(0, 8, 12))
    cube = model.forward(**P).reshape(3, 4, WL.size).astype(np.float32)
    cube[0, 0] = np.nan
    meta = {"wavelength": list(WL), "fwhm": [5.0] * WL.size, "sza": [35], "vza": [12]}
    envi.save_image(str(tmp_path / "in.hdr"), cube, dtype=np.float32, interleave="bsq", ext=".bsq",
                    metadata=meta, force=True)
    # sza/vza/T must match the fixture model for the truth to be recovered
    p = ImageProcessor(str(tmp_path / "in.bsq"), str(tmp_path / "out.bsq"), vary=VARY, init=INIT,
                       output_wcs=["C_x", "C_y", "C_3"], output_iops=["a_phy"], n_jobs=1, siops={"T": 14})
    res = p.run()
    assert res.shape == (3, 4, 3 + 1 + WL.size + 1)       # C_3, C_x, C_y, C_phy, a_phy, residual
    assert np.all(np.isnan(res[0, 0])) and not p.valid[0, 0]        # no valid input: NaN
    np.testing.assert_allclose(res[..., 0].ravel()[1:], P["C_3"][1:], rtol=1e-4)   # bands sorted: C_3, C_x, C_y
    # C_phy = sum of C_0..C_6 (here only C_3 is non-zero)
    np.testing.assert_allclose(res[..., 3], res[..., 0], rtol=1e-6)
    assert (tmp_path / "out.bsq").exists()
    hdr = envi.read_envi_header(str(tmp_path / "out.hdr"))
    assert hdr["band names"][:4] == ["C_3", "C_x", "C_y", "C_phy"]
    assert hdr["siop resampling"] == p.model.resampling

    # reusing a model gives the same result (no SIOP reload); geometry comes from the header
    shared = MiniWasi(wavelengths=WL, FWHMs=np.full(WL.size, 5.0), sza=10, vza=40, T=14)
    p2 = ImageProcessor(str(tmp_path / "in.bsq"), str(tmp_path / "out2.bsq"), vary=VARY, init=INIT,
                        output_wcs=["C_x", "C_y", "C_3"], output_iops=["a_phy"], n_jobs=1, model=shared)
    np.testing.assert_allclose(p2.run(), res, rtol=1e-6, atol=1e-12)
    assert p2.model is shared and np.isclose(np.degrees(shared.sza), 35)

    with pytest.raises(ValueError):          # other SIOP settings belong to the model
        ImageProcessor("x.bsq", "y.bsq", model=shared, siops={"T": 14, "s_cdom": 0.02})
    other = MiniWasi(wavelengths=WL[:-1], T=14)
    with pytest.raises(ValueError):          # wavelengths must match the image
        ImageProcessor(str(tmp_path / "in.bsq"), str(tmp_path / "out3.bsq"), vary=VARY, init=INIT,
                       n_jobs=1, model=other).run()


def test_batch_image_processor(tmp_path, model):
    from spectral import envi
    rng = np.random.default_rng(5)
    (tmp_path / "in").mkdir()
    truths = {}
    for k in range(3):
        P = dict(C_0=0, C_x=rng.uniform(.3, 3, 6), C_y=rng.uniform(.05, .4, 6), C_3=rng.uniform(0, 8, 6))
        cube = model.forward(**P).reshape(2, 3, WL.size).astype(np.float32)
        meta = {"wavelength": list(WL), "fwhm": [5.0] * WL.size, "sza": [35], "vza": [12]}
        envi.save_image(str(tmp_path / "in" / f"img{k}.hdr"), cube, dtype=np.float32, interleave="bsq",
                        ext=".bsq", metadata=meta, force=True)
        truths[k] = P["C_3"]
    kw = dict(vary=VARY, init=INIT, output_wcs=["C_x", "C_y", "C_3"])
    b = BatchImageProcessor(str(tmp_path / "in"), str(tmp_path / "out"), siops={"T": 14},
                            processor_kwargs=kw, n_workers=1, chunk=2)
    stats = b.run()
    assert stats["images"] == 3 and stats["errors"] == 0 and stats["pixels"] == 18
    for k in range(3):
        out = envi.open(str(tmp_path / "out" / f"img{k}.hdr")).load()
        np.testing.assert_allclose(np.asarray(out)[..., 0].ravel(), truths[k], rtol=1e-4)   # C_3
    import csv
    rows = list(csv.DictReader(open(tmp_path / "out" / "batch_summary.csv")))
    assert [r["image"] for r in sorted(rows, key=lambda r: r["image"])] == ["img0", "img1", "img2"]

    # resume: nothing left to do; a selected subset is found by name
    assert b.run()["images"] == 0
    sub = BatchImageProcessor(str(tmp_path / "in"), str(tmp_path / "out_sub"), siops={"T": 14},
                              processor_kwargs=kw, images=["img1"], n_workers=1)
    assert sub.run()["images"] == 1 and (tmp_path / "out_sub" / "img1.bsq").exists()


def test_batch_mixed_band_setups(tmp_path):
    """Images with different wavelengths and with the same wavelengths but different FWHMs,
    alternating within one chunk: each must be fitted with the model of its own band setup."""
    from spectral import envi
    rng = np.random.default_rng(11)
    (tmp_path / "in").mkdir()
    setups = [(WL, 5.0), (WL, 20.0), (WL[::3], 10.0)]
    truths = {}
    for k in range(6):
        wl, fw = setups[k % 3]
        m = MiniWasi(wavelengths=wl, FWHMs=np.full(wl.size, fw), sza=35, vza=12, T=14)
        P = dict(C_0=0, C_x=rng.uniform(.3, 3, 4), C_y=rng.uniform(.05, .4, 4), C_3=rng.uniform(1, 8, 4))
        cube = m.forward(**P).reshape(2, 2, wl.size).astype(np.float32)
        meta = {"wavelength": list(wl), "fwhm": [fw] * wl.size, "sza": [35], "vza": [12]}
        envi.save_image(str(tmp_path / "in" / f"img{k}.hdr"), cube, dtype=np.float32, interleave="bsq",
                        ext=".bsq", metadata=meta, force=True)
        truths[k] = P["C_3"]
    b = BatchImageProcessor(str(tmp_path / "in"), str(tmp_path / "out"), siops={"T": 14},
                            processor_kwargs=dict(vary=VARY, init=INIT, output_wcs=["C_3"]),
                            n_workers=1, chunk=6)
    assert b.run()["errors"] == 0
    for k in range(6):
        out = np.asarray(envi.open(str(tmp_path / "out" / f"img{k}.hdr")).load())
        np.testing.assert_allclose(out[..., 0].ravel(), truths[k], rtol=1e-4)


def test_wasi_files():
    """All data files are readable and the extras are in WASI format (Latin-1, CRLF, tab-separated)."""
    from miniwasi import resampling
    for name in (resampling.WATER, resampling.DAW_DT, resampling.BB_PHY_NORM, resampling.CDOM_NORM) + resampling.PHYTO:
        wl, v = resampling.read_spectrum(name)
        assert wl.size > 100 and np.all(np.isfinite(v)), name
    for name in ("CYANO_PC.A",):
        raw = (resampling.data_dir / name).read_bytes()
        raw.decode("latin-1")
        assert b"\r\n" in raw and b"\n\n" not in raw.replace(b"\r\n", b"\n")[-3:]
    a = MiniWasi(wavelengths=WL).a_i_spec_res
    assert a.shape == (WL.size, 7) and np.all(np.isfinite(a))


def test_resampling_wasi():
    """Gaussian resampling as WASI resample_Gauss: written out here with WASI's loop."""
    from miniwasi import resampling
    wl, y = resampling.read_spectrum(resampling.WATER)
    centres, fwhm = np.array([442.5, 560.0, 681.25]), np.array([10.0, 10.0, 7.5])
    expected = []
    for c, f in zip(centres, fwhm):
        s, yy = f / 2.35, 0.0
        for i in range(wl.size):
            if abs(wl[i] - c) < 3 / 2.35 * f:
                dy = abs(wl[i] - wl[i - 1]) if i > 0 else abs(wl[1] - wl[0])
                yy += y[i] * 1.0025 / s / np.sqrt(2 * np.pi) * np.exp(-0.5 * ((wl[i] - c) / s) ** 2) * dy
        expected.append(yy)
    np.testing.assert_allclose(resampling.resample(resampling.WATER, centres, fwhm), expected, rtol=1e-12)
    # the old method is still available and close to it
    old = resampling.resample(resampling.WATER, centres, fwhm, method="bandresampler")
    np.testing.assert_allclose(old, expected, rtol=0.05)
    assert MiniWasi(wavelengths=centres, FWHMs=fwhm, resampling="bandresampler").resampling == "bandresampler"
    with pytest.raises(ValueError):
        resampling.resample(resampling.WATER, centres, fwhm, method="boxcar")
    # window not covered by the file: NaN (phytoplankton: 0 outside the file instead)
    assert np.isnan(resampling.resample(resampling.BB_PHY_NORM, [352.0], [10.0]))
    assert np.isfinite(resampling.resample_phyto(np.array([352.0, 900.0]), 10.0)).all()


# ---- comparison with WASI7 ------------------------------------------------------------------------
# Forward R_rs spectra exported from WASI7 (*.FWD, 1 nm, deep water, above surface; WASI settings in
# the WASI7.INI saved with each run). WASI did not resample these spectra (flag_resample = 0), so
# miniwasi uses "bandresampler", which with FWHMs=None reproduces the file values on a 1 nm grid.
WASI_FWD_DIR = Path(__file__).resolve().parents[1] / "miniwasi" / "data" / "tests"
WASI_FWD_CASES = {
    # file: (MiniWasi settings, concentrations)
    "rho_L_test.FWD": (dict(sza=80, vza=80), dict(C_0=2, C_x=1, C_y=0.1)),   # WASI rho_L = 0.347 (Fresnel, 80 deg): glint only, not in R_rs
    "Cmie_test.FWD": (dict(sza=0, vza=0), dict(C_0=2, C_mie=1, C_y=0.1)),    # C_mie with bb_Mie = 0.0042, n = -1
}


def read_wasi_fwd(path):
    """Wavelengths and values of a WASI .FWD file (data lines: "wavelength <tab> value")."""
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return np.array(rows).T


@pytest.mark.parametrize("fname", sorted(WASI_FWD_CASES))
def test_forward_matches_wasi7(fname):
    settings, conc = WASI_FWD_CASES[fname]
    wl, rrs_wasi = read_wasi_fwd(WASI_FWD_DIR / fname)
    m = MiniWasi(resampling="bandresampler", **settings)      # default wavelengths 400-899 nm
    rrs = m.forward(**conc)
    ref = np.interp(m.wavelengths, wl, rrs_wasi)
    # WASI writes 6 decimals: atol covers the rounding of the small NIR values
    np.testing.assert_allclose(rrs, ref, rtol=1e-3, atol=1e-6)


# ---- viewing geometry --------------------------------------------------------------------------------
def _write_image(path, meta_extra):
    from spectral import envi
    cube = np.full((2, 3, WL.size), 0.004, np.float32)
    meta = {"wavelength": list(WL), "fwhm": [5.0] * WL.size, **meta_extra}
    envi.save_image(str(path.with_suffix(".hdr")), cube, dtype=np.float32, interleave="bsq", ext=".bsq",
                    metadata=meta, force=True)
    return str(path.with_suffix(".bsq"))


def test_geometry_from_header_argument_default(tmp_path):
    import warnings
    from spectral import envi
    kw = dict(vary={"C_x": True}, init={"C_x": 1}, output_wcs=["C_x"], n_jobs=1, progress=False)
    # header with braces and as a plain value ("sza = 31", as in some ENVI files)
    img = _write_image(tmp_path / "a", {"sza": [35], "vza": [12]})
    p = ImageProcessor(img, str(tmp_path / "a_out.bsq"), **kw); p.run()
    assert p.angles == {"sza": 35.0, "vza": 12.0} and p.defaulted == []
    import re
    hdr_txt = re.sub(r"(?m)^sza = .*$", "sza = 31", (tmp_path / "a.hdr").read_text())
    hdr_txt = re.sub(r"(?m)^vza = .*$", "vza = 47", hdr_txt)
    (tmp_path / "a.hdr").write_text(hdr_txt)
    p = ImageProcessor(img, str(tmp_path / "a_out.bsq"), **kw); p.run()
    assert p.angles == {"sza": 31.0, "vza": 47.0}
    # argument overrides the header, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = ImageProcessor(img, str(tmp_path / "a_out.bsq"), sza=20, vza=5, **kw); p.run()
    assert p.angles == {"sza": 20.0, "vza": 5.0}
    # missing in the header: defaults with a warning; written to the output header
    img = _write_image(tmp_path / "b", {})
    with pytest.warns(UserWarning, match="neither given nor in the header"):
        p = ImageProcessor(img, str(tmp_path / "b_out.bsq"), **kw); p.run()
    assert p.angles == {"sza": 30.0, "vza": 0.0} and p.defaulted == ["sza", "vza"]
    out = envi.read_envi_header(str(tmp_path / "b_out.hdr"))
    assert float(out["sza"][0]) == 30 and float(out["vza"][0]) == 0


def test_batch_geometry(tmp_path):
    import pandas as pd
    (tmp_path / "in").mkdir()
    _write_image(tmp_path / "in" / "with_angles", {"sza": [35], "vza": [12]})
    _write_image(tmp_path / "in" / "without_angles", {})
    kw = dict(vary={"C_x": True}, init={"C_x": 1}, output_wcs=["C_x"])
    with pytest.warns(UserWarning, match="1 images had no sza"):
        BatchImageProcessor(str(tmp_path / "in"), str(tmp_path / "out"), processor_kwargs=kw, n_workers=1).run()
    s = pd.read_csv(tmp_path / "out" / "batch_summary.csv").set_index("image")
    assert s.loc["with_angles", "sza"] == 35 and pd.isna(s.loc["with_angles", "default_angles"])
    assert s.loc["without_angles", "sza"] == 30 and s.loc["without_angles", "default_angles"] == "sza vza"
    # per-image dict: the image without angles gets them, the other keeps its header values
    BatchImageProcessor(str(tmp_path / "in"), str(tmp_path / "out2"), processor_kwargs=kw, n_workers=1,
                        sza={"without_angles": 50}, vza={"without_angles": 20}).run()
    s = pd.read_csv(tmp_path / "out2" / "batch_summary.csv").set_index("image")
    assert (s.loc["without_angles", "sza"], s.loc["without_angles", "vza"]) == (50, 20)
    assert s.loc["with_angles", "sza"] == 35
    # a summary with other columns (older miniwasi) is kept under a new name, not appended to
    (tmp_path / "out2" / "batch_summary.csv").write_text("image,valid_pixels\nx,1\n")
    BatchImageProcessor(str(tmp_path / "in"), str(tmp_path / "out2"), processor_kwargs=kw, n_workers=1,
                        skip_existing=False).run()
    assert len(list((tmp_path / "out2").glob("batch_summary_old_*.csv"))) == 1
    assert pd.read_csv(tmp_path / "out2" / "batch_summary.csv").columns[-1] == "error"

