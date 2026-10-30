"""
Regression tests for miniwasi (synthetic spectra only). Run with:  pytest tests
"""

import numpy as np
import pytest

from miniwasi import MiniWasi, ImageProcessor, BatchImageProcessor

NAMES = ['C_x', 'C_mie', 'C_y'] + [f'C_{i}' for i in range(8)]
WL = np.arange(400, 800, 5.0)


@pytest.fixture(scope="module")
def model():
    return MiniWasi(wavelengths=WL, FWHMs=np.full(WL.size, 5.0), sza=35, va=12, T=14)


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
    sza, va, T = rng.uniform(20, 60, 5), rng.uniform(0, 30, 5), rng.uniform(5, 25, 5)
    P = random_params(5)
    R = MiniWasi(wavelengths=WL, sza=sza, va=va, T=T).forward(**P)
    for n in range(5):
        single = MiniWasi(wavelengths=WL, sza=sza[n], va=va[n], T=T[n])
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
    # C_phy = sum of C_0..C_7 (here only C_3 is non-zero)
    np.testing.assert_allclose(res[..., 3], res[..., 0], rtol=1e-6)
    assert (tmp_path / "out.bsq").exists()
    hdr = envi.read_envi_header(str(tmp_path / "out.hdr"))
    assert hdr["band names"][:4] == ["C_3", "C_x", "C_y", "C_phy"]

    # reusing a model gives the same result (no SIOP reload); geometry comes from the header
    shared = MiniWasi(wavelengths=WL, FWHMs=np.full(WL.size, 5.0), sza=10, va=40, T=14)
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
        m = MiniWasi(wavelengths=wl, FWHMs=np.full(wl.size, fw), sza=35, va=12, T=14)
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
    for name in ("CYANO_PC.A", "DINOFLAGELLATES_GSANQUINEUM.A"):
        raw = (resampling.data_dir / name).read_bytes()
        raw.decode("latin-1")
        assert b"\r\n" in raw and b"\n\n" not in raw.replace(b"\r\n", b"\n")[-3:]
    a = MiniWasi(wavelengths=WL).a_i_spec_res
    assert a.shape == (WL.size, 8) and np.all(np.isfinite(a))
