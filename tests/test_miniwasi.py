"""
Regression tests for miniwasi (synthetic spectra only). Run with:  pytest tests
"""

import numpy as np
import pytest

from miniwasi import MiniWasi, ImageProcessor

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
    assert res.shape == (3, 4, 3 + WL.size + 1)
    assert np.all(res[0, 0] == 0) and not p.valid[0, 0]
    np.testing.assert_allclose(res[..., 0].ravel()[1:], P["C_3"][1:], rtol=1e-4)   # bands sorted: C_3, C_x, C_y
    assert (tmp_path / "out.bsq").exists()
