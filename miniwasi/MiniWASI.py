import copy
import numpy as np
from . import resampling
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
warnings.filterwarnings("ignore", 'Image data contains NaN values.')


class MiniWasi():
    """
    Vectorised WASI-type bio-optical model for deep water. Same physics, SIOPs and
    attribute names as the original lmfit-based MiniWasi.

    Differences to the original lmfit-based MiniWasi:
      * forward() accepts arrays: every C_* may be a scalar or an array of shape (N,);
        all outputs (self.a, self.bb, self.R_rs, ...) then have shape (N, L) instead of (L,).
      * invert() takes one spectrum (L,) or many spectra (N, L) and fits them all at once
        with a bounded Levenberg-Marquardt using the analytic Jacobian (see jacobian()).
        It no longer uses lmfit and returns a plain dict instead of an lmfit result.
      * T, va and sza may also be arrays of shape (N,), one value per observation
        (e.g. in-situ matchups with their own geometry). set_conditions() changes them
        without reloading the SIOPs.
    """

    def __init__(self, wavelengths = np.arange(400,900), FWHMs = None,
                 T=20, va=0.0001, sza=40,
                 bb_nap_spec=0.0086, a_spec_nap_440nm=0.041, bb_phy_spec=0.001, s_cdom=0.014, s_nap=0.011, n=-1,
                 a_norm_y_from_file = False
                 ):

        self.wavelengths = np.asarray(wavelengths, dtype=float)

        self.bb_nap_spec = bb_nap_spec
        self.a_spec_nap_440nm = a_spec_nap_440nm
        self.bb_phy_spec=bb_phy_spec
        self.s_cdom = s_cdom
        self.s_nap = s_nap
        self.n = n

        # Normalized cdom absorption coefficient at 440 nm
        self.a_norm_y = resampling.resample_a_Y_norm(wavelengths, FWHMs) if a_norm_y_from_file else np.exp(-self.s_cdom*(self.wavelengths-440))

        # Normalized nap absorption coefficient at 440 nm
        self.a_norm_nap = np.exp(-self.s_nap*(self.wavelengths-440))

        # Absorption of water
        self.a_w_res = resampling.resample_a_w(wavelengths, FWHMs)
        # Specific absorption coefficients of 8 phytoplankton types
        # (6 standard WASI6 classes + CYANO_PC + DINOFLAGELLATES_G)
        self.a_i_spec_res = resampling.resample_a_i_spec_loe(wavelengths, FWHMs)
        # Normalized backscattering coefficient of phytoplankton
        self.bb_phy_norm_res = resampling.resample_b_phy_norm(wavelengths, FWHMs)
        # Temperature gradient of pure water absorption
        self.da_W_div_dT_res = resampling.resample_da_W_div_dT(wavelengths, FWHMs)

        # Backscattering of water
        self.bb_w = 0.00111 * (self.wavelengths/500)**(-4.32)

        # Derivatives of a and bb with respect to each concentration.
        # The model is linear in all C_* at the IOP level, so these are constant spectra.
        a_nap_unit = self.a_spec_nap_440nm * self.a_norm_nap
        bb_phy_unit = self.bb_phy_spec * self.bb_phy_norm_res
        self.da_dC = {'C_x': a_nap_unit, 'C_mie': a_nap_unit, 'C_y': self.a_norm_y}
        self.dbb_dC = {'C_x': self.bb_nap_spec * np.ones(self.wavelengths.shape),
                       'C_mie': 0.0042 * (self.wavelengths / 500)**self.n,
                       'C_y': np.zeros(self.wavelengths.shape)}
        for i in range(8):
            self.da_dC[f'C_{i}'] = self.a_i_spec_res[:, i]
            self.dbb_dC[f'C_{i}'] = bb_phy_unit

        # Geometry- and temperature-dependent constants
        self.set_conditions(T=T, va=va, sza=sza)


    def set_conditions(self, T=None, va=None, sza=None):
        """
        Set water temperature [degC], viewing angle and sun zenith angle [deg].
        Each may be a scalar or an array of shape (N,) (one value per observation).
        Arguments left as None keep their current value.
        """
        if T is not None:
            self.T = np.asarray(T, dtype=float)
        if va is not None:
            va = np.asarray(va, dtype=float)
            self.va = np.radians(np.where(va != 0, va, 0.0001))  # avoid division by zero
        if sza is not None:
            self.sza = np.radians(np.asarray(sza, dtype=float))

        # Viewing angle in water
        self.inwater_va = np.arcsin(np.sin(self.va)/1.33)
        # Sun zenith angle in water
        self.inwater_sza = np.arcsin(np.sin(self.sza)/1.33)
        # Fresnel reflectance at the air–water interface
        num_sin = np.sin(self.va - self.inwater_va) ** 2
        den_sin = np.sin(self.va + self.inwater_va) ** 2
        num_tan = np.tan(self.va - self.inwater_va) ** 2
        den_tan = np.tan(self.va + self.inwater_va) ** 2
        self.rho_L = 0.5 * (num_sin / den_sin + num_tan / den_tan) # ca. 0.02 (nadir)

        sizes = {np.size(x) for x in (self.T, self.va, self.sza)} - {1}
        if len(sizes) > 1:
            raise ValueError("T, va and sza must be scalars or arrays of the same length.")
        self.n_obs = sizes.pop() if sizes else None   # None = shared by all spectra


    def band_subset(self, bands):
        """
        Copy of the model restricted to some bands (boolean mask or indices), without
        resampling the SIOPs again. invert() uses it to fit only the bands with weight > 0.
        """
        sub = copy.copy(self)
        for name in ('wavelengths', 'a_norm_y', 'a_norm_nap', 'a_w_res', 'a_i_spec_res',
                     'bb_phy_norm_res', 'da_W_div_dT_res', 'bb_w'):
            setattr(sub, name, getattr(self, name)[bands])
        sub.da_dC = {k: v[bands] for k, v in self.da_dC.items()}
        sub.dbb_dC = {k: v[bands] for k, v in self.dbb_dC.items()}
        return sub


    def _per_obs(self, x, obs=None):
        """Prepare a condition for broadcasting against (..., L): scalars stay scalars,
        per-observation arrays become (N, 1), optionally restricted to the observations `obs`."""
        if np.ndim(x) == 0 or np.size(x) == 1:
            return np.asarray(x).reshape(())
        x = np.asarray(x) if obs is None else np.asarray(x)[obs]
        return x[:, None]


    def forward(self, C_x = 1, C_mie = 0, C_y = 0.2, C_0 = 2, C_1 = 0, C_2 = 0, C_3 = 0, C_4 = 0, C_5 = 0,
                C_6 = 0, C_7 = 0, obs=None):
        """
        Every C_* may be a scalar or an array of shape (N,). With scalars all outputs have
        shape (L,), otherwise (N, L).
        obs: only needed with per-observation conditions; indices of the observations
             (entries of T/va/sza) that the N spectra belong to. Default: all of them.
        """

        # Concentrations as columns so that they broadcast against wavelengths
        C_x, C_mie, C_y, C_0, C_1, C_2, C_3, C_4, C_5, C_6, C_7 = (
            np.asarray(C, dtype=float)[..., None] if np.ndim(C) > 0 else C
            for C in (C_x, C_mie, C_y, C_0, C_1, C_2, C_3, C_4, C_5, C_6, C_7))

        T = self._per_obs(self.T, obs)
        inwater_va = self._per_obs(self.inwater_va, obs)
        inwater_sza = self._per_obs(self.inwater_sza, obs)
        rho_L = self._per_obs(self.rho_L, obs)

        ####
        # Relate IOPs to LUTs
        ####

        ## ABSORPTION

        # CDOM component
        # C_y = a_cdom_440nm
        self.a_cdom = C_y * self.a_norm_y

        # Phytoplankton component
        self.a_phy = C_0*self.a_i_spec_res[:,0] + C_1*self.a_i_spec_res[:,1] + C_2*self.a_i_spec_res[:,2] + C_3*self.a_i_spec_res[:,3] + C_4*self.a_i_spec_res[:,4] + C_5*self.a_i_spec_res[:,5] + C_6*self.a_i_spec_res[:,6] + C_7*self.a_i_spec_res[:,7]
        C_phy = C_0 + C_1 + C_2 + C_3 + C_4 + C_5 + C_6 + C_7

        # NAP component
        # Normalized nap absorption coefficient
        C_nap = C_x + C_mie
        # WASI a_spec_nap_440nm: 0.055, manual: 0.041
        self.a_nap = C_nap * self.a_spec_nap_440nm * self.a_norm_nap

        # Bulk absorption
        T0 = 20
        self.a_wc = self.a_cdom + self.a_nap + self.a_phy
        self.a = self.a_w_res + (T - T0) * self.da_W_div_dT_res + self.a_wc

        ## BACKSCATTERING

        # Water: self.bb_w (constant, computed in __init__)

        # Phytoplankton - ONLY ONE ref spectrum
        self.bb_phy = C_phy * self.bb_phy_spec * self.bb_phy_norm_res

        # NAP
        # WASI bb_nap_spec: 0.013, WASI manual: 0.0086
        self.bb_nap = self.bb_nap_spec * C_x * np.ones(self.wavelengths.shape) + 0.0042 * C_mie * (self.wavelengths / 500)**self.n

        self.bb_wc = self.bb_phy + self.bb_nap
        # Bulk backscattering
        self.bb = self.bb_w + self.bb_wc

        ####
        # Relate IOPs to Rrs
        ####
        self.wb = (self.bb/(self.a+self.bb))
        self.wb = np.clip(self.wb, 0, 1)

        # Account for anisotropy (polynomial in wb times a geometry factor)
        self.f_rs_geometry = 0.0512 * (1 + 0.1098/np.cos(inwater_sza)) * (1 + 0.4021/np.cos(inwater_va))
        self.f_rs_poly = 1 + 4.6659 * self.wb - 7.8387 * self.wb**2 + 5.4571 * self.wb**3
        f_rs = self.f_rs_geometry * self.f_rs_poly

        # below water
        self.r_rs_below = f_rs * self.wb

        self.xi = (1-0.03)*(1-rho_L)/1.33**2 # ca. 0.53

        self.R_rs = self.xi * (self.r_rs_below/(1-0.54*5*self.r_rs_below))

        self.R_rs = np.nan_to_num(
                        self.R_rs,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0
                    )

        # Kd for jonas
        self.Kd = 1.0546 * ((self.a + self.bb) / np.cos(inwater_sza))

        return self.R_rs


    def jacobian(self, names):
        """
        Derivatives dR_rs/dC of the LAST forward() call for the parameters in `names`.
        Returns shape (L, p) for a single spectrum or (N, L, p).

        Chain rule through the model:
            dR_rs/dC = dR_rs/dr_rs * dr_rs/dwb * (dwb/da * da/dC + dwb/dbb * dbb/dC)
        """
        wb = self.wb
        r = self.r_rs_below

        # R_rs = xi * r / (1 - 2.7 r)
        dRrs_dr = self.xi / (1 - 0.54*5*r)**2
        # r = f_rs_geometry * f_rs_poly(wb) * wb
        df_rs_poly_dwb = 4.6659 - 2*7.8387 * wb + 3*5.4571 * wb**2
        dr_dwb = self.f_rs_geometry * (self.f_rs_poly + wb * df_rs_poly_dwb)
        # wb = bb / (a + bb)
        dwb_da = -self.bb / (self.a + self.bb)**2
        dwb_dbb = self.a / (self.a + self.bb)**2

        dRrs_da = dRrs_dr * dr_dwb * dwb_da
        dRrs_dbb = dRrs_dr * dr_dwb * dwb_dbb

        return np.stack([dRrs_da * self.da_dC[name] + dRrs_dbb * self.dbb_dC[name] for name in names], axis=-1)


    def invert(self, Rrs_measured, weights=None, vary=None,
               init=None, bounds=None, max_iter=200, tol=1e-12, n_jobs=1, block_size=2000,
               progress=True):
        """
        Use like this:
        result = model.invert(
                    Rrs_measured,
                    vary={'C_x': True, 'C_y': True},
                    init={'C_x': 5.0},
                    bounds={'C_y': (0, 1.0)}
                )

        Rrs_measured: one spectrum (L,) or many spectra (N, L), e.g. image pixels or a set of
                      in-situ spectra. All spectra share weights, vary, init and bounds.
        n_jobs, block_size: for many spectra, split them into blocks of block_size and fit
                      the blocks on n_jobs cores (-1 = all). Default: one process.
        progress:     show a tqdm progress bar (counts finished blocks; only shown when
                      there is more than one block, i.e. N > block_size).

        Returns a dict (floats for one spectrum, arrays of shape (N,) for many):
            result['params']   {name: value} for all 11 parameters (fixed ones at init),
                               so model.forward(**result['params']) gives the fitted IOPs/Rrs
            result['stderr']   {name: standard error} for the fitted parameters
                               (NaN if the parameter ended on a bound)
            result['residual'] sqrt(mean over all bands of (w/mean(w)) * (R_rs_model - R_rs)^2),
                               the same definition as the residual band of ImageProcessor
            result['success']  True if converged within max_iter
            result['n_iter']   number of iterations
        """

        # defaults
        param_names = ['C_x', 'C_mie', 'C_y', 'C_0', 'C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7']

        default_init = {
            'C_x': 1.0,
            'C_mie': 0,
            'C_y': 0.1,
            'C_0': 4.0,
            'C_1': 0.0,
            'C_2': 0.0,
            'C_3': 0.0,
            'C_4': 0.0,
            'C_5': 0.0,
            'C_6': 0.0,
            'C_7': 0.0,
        }

        default_bounds = {
            'C_x': (0, 100),
            'C_mie': (0, 100),
            'C_y': (0, 10),
            'C_0': (0, 50),
            'C_1': (0, 50),
            'C_2': (0, 50),
            'C_3': (0, 50),
            'C_4': (0, 50),
            'C_5': (0, 50),
            'C_6': (0, 50),
            'C_7': (0, 50),
        }

        # spectra
        single = np.ndim(Rrs_measured) == 1
        Rrs_measured = np.atleast_2d(np.asarray(Rrs_measured, dtype=float))
        N, L = Rrs_measured.shape
        if self.n_obs is not None and self.n_obs != N:
            raise ValueError(f"{N} spectra but {self.n_obs} observations in T/va/sza.")

        # spectral weighting (zero-weight bands are left out of the fit)
        if weights is None:
            weights = np.ones(L)
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.mean(weights)
        fit_bands = weights > 0
        if not np.all(np.isfinite(Rrs_measured[:, fit_bands])):
            raise ValueError("Rrs_measured contains non-finite values on bands with weight > 0.")

        # user overrides
        vary = vary or {}
        init = {**default_init, **(init or {})}
        bounds = {**default_bounds, **(bounds or {})}

        free = [name for name in param_names if vary.get(name, False)]
        if not free:
            raise ValueError("No parameter is set to vary.")
        fixed = {name: init[name] for name in param_names if name not in free}
        lower = np.array([bounds[name][0] for name in free], dtype=float)
        upper = np.array([bounds[name][1] for name in free], dtype=float)
        start = np.clip([init[name] for name in free], lower, upper)

        settings = dict(free=free, fixed=fixed, lower=lower, upper=upper, start=start,
                        weights=weights[fit_bands], max_iter=max_iter, tol=tol)
        fit_model = self.band_subset(fit_bands)
        Rrs_fit = Rrs_measured[:, fit_bands]

        # fit in blocks of block_size spectra (keeps the arrays small), on several cores if asked for
        obs = np.arange(N)
        starts = range(0, N, block_size)
        progress_bar = dict(total=len(starts), desc="Inversion", unit="block",
                            disable=not progress or len(starts) <= 1)
        if n_jobs == 1 or N <= block_size:
            blocks = [fit_model._levenberg_marquardt(Rrs_fit[s:s+block_size], obs[s:s+block_size], **settings)
                      for s in tqdm(starts, **progress_bar)]
        else:
            # drop the (possibly large) outputs of earlier forward() calls before the model
            # is copied to the worker processes
            for name in ('a_cdom', 'a_phy', 'a_nap', 'a_wc', 'a', 'bb_phy', 'bb_nap', 'bb_wc', 'bb', 'wb',
                         'f_rs_geometry', 'f_rs_poly', 'r_rs_below', 'xi', 'R_rs', 'Kd'):
                fit_model.__dict__.pop(name, None)
            tasks = (delayed(fit_model._levenberg_marquardt)(Rrs_fit[s:s+block_size], obs[s:s+block_size], **settings)
                     for s in starts)
            try:
                # results come back in order as blocks finish, so the bar shows real progress
                results = Parallel(n_jobs=n_jobs, prefer="processes", return_as="generator")(tasks)
            except TypeError:
                # joblib < 1.3 has no return_as: the bar only fills once all blocks are done
                results = Parallel(n_jobs=n_jobs, prefer="processes")(tasks)
            blocks = list(tqdm(results, **progress_bar))
        theta, cost, n_iter, success, stderr = (np.concatenate(x) for x in zip(*blocks))

        # results
        params = {name: np.full(N, value, dtype=float) for name, value in fixed.items()}
        params.update({name: theta[:, k] for k, name in enumerate(free)})
        params = {name: params[name] for name in param_names}
        result = {
            'params': params,
            'stderr': {name: stderr[:, k] for k, name in enumerate(free)},
            'residual': np.sqrt(cost / L),
            'success': success,
            'n_iter': n_iter,
        }
        if single:
            result = {key: ({k: v[0].item() for k, v in val.items()} if isinstance(val, dict) else val[0].item())
                      for key, val in result.items()}
        return result


    def _levenberg_marquardt(self, Rrs_measured, obs, free, fixed, lower, upper, start,
                             weights, max_iter, tol):
        """
        Bounded Levenberg-Marquardt, all spectra at once.

        For each spectrum, iterate
            (H + mu * diag(H)) * step = -g,     H = J^T W J,   g = J^T W res
        Accept the step if the weighted cost sum(w * res^2) decreases (then mu /= 3),
        otherwise reject it (mu *= 4). Bounds: the new values are clipped to [lower, upper],
        and a parameter sitting on a bound with the gradient pointing outward is held fixed
        for that step. A spectrum is done when the cost no longer decreases noticeably.
        """
        N, p = Rrs_measured.shape[0], len(free)
        identity = np.eye(p)

        def model(values, rows):
            """Residuals and Jacobian for the spectra `rows`."""
            C = {**fixed, **{name: values[:, k] for k, name in enumerate(free)}}
            Rrs = self.forward(**C, obs=obs[rows])
            return Rrs - Rrs_measured[rows], self.jacobian(free)

        values = np.tile(start, (N, 1))
        res, J = model(values, np.arange(N))
        cost = np.sum(weights * res**2, axis=1)
        mu = np.full(N, 1e-3)
        active = np.ones(N, dtype=bool)
        n_iter = np.zeros(N, dtype=int)

        for _ in range(max_iter):
            rows = np.flatnonzero(active)
            if rows.size == 0:
                break
            n_iter[rows] += 1

            # Normal equations
            JW = J[rows] * weights[:, None]
            H = np.einsum('nlp,nlq->npq', JW, J[rows])
            g = np.einsum('nlp,nl->np', JW, res[rows])

            # Hold parameters fixed that sit on a bound and want to go further out
            on_bound = ((values[rows] <= lower) & (g > 0)) | ((values[rows] >= upper) & (g < 0))
            if on_bound.any():
                free_mask = (~on_bound).astype(float)
                H = H * free_mask[:, :, None] * free_mask[:, None, :] + on_bound[:, :, None] * identity
                g = g * free_mask

            # Damped step, clipped to the bounds
            damping = mu[rows, None] * np.maximum(np.einsum('npp->np', H), 1e-30)
            step = np.linalg.solve(H + damping[:, :, None] * identity, -g[:, :, None])[:, :, 0]
            new_values = np.clip(values[rows] + step, lower, upper)

            new_res, new_J = model(new_values, rows)
            new_cost = np.sum(weights * new_res**2, axis=1)

            # Accept improvements, adapt damping, check convergence
            better = new_cost < cost[rows]
            converged = better & (cost[rows] - new_cost <= tol * cost[rows] + 1e-30)
            accepted = rows[better]
            values[accepted] = new_values[better]
            res[accepted], J[accepted], cost[accepted] = new_res[better], new_J[better], new_cost[better]
            mu[accepted] /= 3
            mu[rows[~better]] *= 4

            active[rows[converged]] = False
            active[rows[mu[rows] > 1e10]] = False   # no further decrease possible: at the minimum

        stderr = self._standard_errors(values, J, cost, weights, lower, upper)
        return values, cost, n_iter, ~active, stderr


    @staticmethod
    def _standard_errors(values, J, cost, weights, lower, upper):
        """
        Standard errors from the covariance s^2 (J^T W J)^-1 with
        s^2 = cost / (number of fitted bands - number of parameters not on a bound).
        NaN for parameters that ended on a bound.
        """
        p = values.shape[1]
        on_bound = (values <= lower) | (values >= upper)
        free_mask = (~on_bound).astype(float)
        H = np.einsum('nlp,nlq->npq', J * weights[:, None], J)
        H = H * free_mask[:, :, None] * free_mask[:, None, :] + on_bound[:, :, None] * np.eye(p)
        dof = len(weights) - np.sum(~on_bound, axis=1)
        s2 = np.where(dof > 0, cost / np.maximum(dof, 1), np.nan)
        with np.errstate(invalid="ignore"):
            stderr = np.sqrt(np.einsum('npp->np', np.linalg.pinv(H)) * s2[:, None])
        stderr[on_bound] = np.nan
        return stderr
