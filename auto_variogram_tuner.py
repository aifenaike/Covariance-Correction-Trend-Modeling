"""
================================================================================
VARIOGRAM TUNER — Automated Fitting Pipeline
================================================================================
Fits spherical, exponential, and Gaussian variogram models to experimental
variogram data and selects the best model by Cressie's WLS criterion.
Used by auto_fit_variogram() to produce geostatspy-compatible variogram models
for sequential Gaussian simulation of corrected and uncorrected residuals.
================================================================================
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')
import geostatspy.GSLIB as GSLIB
import geostatspy.geostats as geostats


# =============================================================================
# VARIOGRAM MODELS γ(h)
# =============================================================================
# Convention: γ(h) = nugget + partial_sill * f(h/range)
# All models take params = (nugget, partial_sill, range)

def spherical(h, n, c, a):
    """Spherical model — reaches sill exactly at h = a."""
    h = np.asarray(h, float)
    u = np.clip(h / a, 0, np.inf)
    return n + c * np.where(h < a, 1.5 * u - 0.5 * u**3, 1.0)


def exponential(h, n, c, a):
    """Exponential model — practical range ≈ 3a (reaches 95% of sill)."""
    h = np.asarray(h, float)
    return n + c * (1.0 - np.exp(-h / a))


def gaussian_vgm(h, n, c, a):
    """Gaussian model — very smooth at origin. Practical range ≈ √3·a."""
    h = np.asarray(h, float)
    return n + c * (1.0 - np.exp(-(h / a)**2))


# =============================================================================
# FITTING CRITERIA
# =============================================================================

def compute_rss(y_true, y_pred, weights=None):
    """Residual sum of squares, optionally weighted by number of pairs."""
    resid = y_true - y_pred
    if weights is not None:
        return float(np.sum(np.asarray(weights, float) * resid**2))
    return float(np.sum(resid**2))


def compute_wls_cressie(y_true, y_pred, npairs):
    """
    Cressie's WLS criterion — recommended for variogram fitting.
    Weights lag bins by pair count and inverse predicted variance,
    giving more influence to near-origin bins with many pairs.
    Reference: Cressie (1985), Mathematical Geology 17(5).
    """
    pred_safe = np.maximum(y_pred, 1e-10)
    N = np.asarray(npairs, float)
    resid = y_true / pred_safe - 1.0   # relative residual
    return float(np.sum(N * resid**2))


def compute_aic(n_obs, rss_val, k_params):
    """AIC — penalizes model complexity. Lower is better."""
    if rss_val <= 0 or n_obs <= 0:
        return np.inf
    return n_obs * np.log(rss_val / n_obs) + 2 * k_params


def compute_bic(n_obs, rss_val, k_params):
    """BIC — penalizes complexity more than AIC for n > ~8. Lower is better."""
    if rss_val <= 0 or n_obs <= 0:
        return np.inf
    return n_obs * np.log(rss_val / n_obs) + k_params * np.log(n_obs)


def loocv_score(lags, gamma_exp, model_func, p0, bounds, weights=None):
    """
    Leave-one-bin-out cross-validation for variogram model selection.
    More robust than AIC/BIC when the number of lag bins is small (10-20),
    which is typical for experimental variograms.
    Returns mean squared CV error — lower is better generalization.
    """
    n = len(lags)
    if n <= 3:
        return np.inf

    cv_errors = []
    for j in range(n):
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        h_train, g_train = lags[mask], gamma_exp[mask]

        sigma_train = None
        if weights is not None:
            sigma_train = 1.0 / np.sqrt(np.maximum(weights[mask], 1e-12))

        try:
            popt, _ = curve_fit(model_func, h_train, g_train, p0=p0,
                                bounds=bounds, sigma=sigma_train, maxfev=10000)
            cv_errors.append((gamma_exp[j] - model_func(lags[j], *popt))**2)
        except Exception:
            cv_errors.append(np.inf)

    return float(np.mean(cv_errors))


# =============================================================================
# MAIN TUNER
# =============================================================================

def tune_variogram(
    lags: np.ndarray,
    gamma_exp: np.ndarray,
    npairs: Optional[np.ndarray] = None,
    model_space: Optional[Dict] = None,
    criterion: str = 'wls_cressie',
    compute_cv: bool = True,
) -> Dict:
    """
    Fit multiple variogram models and return the best by the chosen criterion.

    Multiple initial conditions are tried per model to avoid local minima.
    The best fit per model is selected by RSS across starts, then models are
    ranked by the chosen criterion.

    Parameters
    ----------
    lags       : lag centers (h)
    gamma_exp  : experimental semivariances γ(h)
    npairs     : pair counts per lag bin — required for WLS and Cressie criteria
    model_space: optional dict to override default model configurations
    criterion  : 'rss' | 'wls' | 'wls_cressie' | 'aic' | 'bic' | 'cv_score'
    compute_cv : whether to compute LOOCV scores (slower but informative)

    Returns
    -------
    dict with keys:
        'best'          : best model result dict
        'all_results'   : all fitted models sorted by criterion
        'criterion_used': criterion string used for selection

    Each model result dict contains:
        'model', 'params' (nugget, partial_sill, total_sill, range),
        'rss', 'wls_cressie', 'aic', 'bic', 'cv_score', 'popt', 'pcov'
    """
    h = np.asarray(lags, float)
    g = np.asarray(gamma_exp, float)

    assert len(h) == len(g), "lags and gamma_exp must have the same length"
    assert np.all(np.isfinite(h)) and np.all(np.isfinite(g)), "Non-finite values in input"

    if npairs is not None:
        npairs = np.asarray(npairs, float)

    # Data-informed initial guesses
    gmin, gmax = np.nanmin(g), np.nanmax(g)
    sill0 = max(gmax, 1e-12)
    nug0  = max(min(g[:2].mean() if len(g) >= 2 else gmin, sill0 * 0.5), 0.0)
    a0    = max((np.max(h) - np.min(h)) / 3.0, 1e-6)

    if model_space is None:
        model_space = {
            'spherical': {
                'func': spherical,
                'inits': [
                    (nug0, sill0 - nug0, a0),
                    (0.0,  sill0,        a0),
                    (nug0, sill0 - nug0, a0 * 0.6),
                    (nug0, sill0 - nug0, a0 * 1.5),
                    (gmin, gmax - gmin,  a0),
                ],
                'bounds': ([0.0, 0.0, 1e-6], [np.inf, np.inf, np.inf]),
                'n_params': 3,
            },
            'exponential': {
                'func': exponential,
                'inits': [
                    (nug0, sill0 - nug0, a0),
                    (0.0,  sill0,        a0),
                    (nug0, sill0 - nug0, a0 * 0.6),
                    (nug0, sill0 - nug0, a0 * 1.5),
                    (gmin, gmax - gmin,  a0 / 3.0),
                ],
                'bounds': ([0.0, 0.0, 1e-6], [np.inf, np.inf, np.inf]),
                'n_params': 3,
            },
            'gaussian': {
                'func': gaussian_vgm,
                'inits': [
                    (nug0, sill0 - nug0, a0),
                    (0.0,  sill0,        a0),
                    (nug0, sill0 - nug0, a0 * 0.8),
                    (nug0, sill0 - nug0, a0 * 1.8),
                ],
                'bounds': ([0.0, 0.0, 1e-6], [np.inf, np.inf, np.inf]),
                'n_params': 3,
            },
        }

    all_results = []

    for name, cfg in model_space.items():
        func     = cfg['func']
        inits    = cfg.get('inits', [])
        bounds   = cfg.get('bounds', (-np.inf, np.inf))
        n_params = cfg.get('n_params', 3)

        best_for_model = None

        for p0 in inits:
            try:
                # Weight curve_fit by inverse sqrt of pair counts if available
                sigma = None
                if npairs is not None:
                    sigma = 1.0 / np.sqrt(np.maximum(npairs, 1e-12))

                popt, pcov = curve_fit(func, h, g, p0=p0, bounds=bounds,
                                       sigma=sigma, maxfev=20000)
                pred = func(h, *popt)

                rss_val     = compute_rss(g, pred)
                wls_val     = compute_rss(g, pred, weights=npairs) if npairs is not None else rss_val
                cressie_val = compute_wls_cressie(g, pred, npairs) if npairs is not None else np.inf
                aic_val     = compute_aic(len(g), rss_val, n_params)
                bic_val     = compute_bic(len(g), rss_val, n_params)

                params_dict = {
                    'nugget'      : float(popt[0]),
                    'partial_sill': float(popt[1]),
                    'total_sill'  : float(popt[0] + popt[1]),
                    'range'       : float(popt[2]),
                }

                result = {
                    'model'       : name,
                    'params'      : params_dict,
                    'n_params'    : n_params,
                    'rss'         : rss_val,
                    'wls'         : wls_val,
                    'wls_cressie' : cressie_val,
                    'aic'         : aic_val,
                    'bic'         : bic_val,
                    'cv_score'    : None,
                    'popt'        : popt,
                    'pcov'        : pcov,
                    'func'        : func,
                }

                # Keep the best-fitting start for this model
                if best_for_model is None or rss_val < best_for_model['rss']:
                    best_for_model = result

            except Exception:
                continue

        if best_for_model is not None:
            # Compute LOOCV for the best start of each model
            if compute_cv and len(h) > 4:
                best_for_model['cv_score'] = loocv_score(
                    h, g, func, best_for_model['popt'], bounds, weights=npairs
                )
            all_results.append(best_for_model)

    if not all_results:
        raise RuntimeError("All model fits failed. Check your input data.")

    valid_criteria = {'rss', 'wls', 'wls_cressie', 'aic', 'bic', 'cv_score'}
    if criterion not in valid_criteria:
        raise ValueError(f"criterion must be one of {valid_criteria}")

    def sort_key(r):
        val = r.get(criterion)
        return val if (val is not None and np.isfinite(val)) else np.inf

    all_results.sort(key=sort_key)

    return {
        'best'          : all_results[0],
        'all_results'   : all_results,
        'criterion_used': criterion,
    }


# =============================================================================
# PRACTICAL RANGE HELPER
# =============================================================================

def practical_range(model_name, range_param):
    """
    Convert the fitted range parameter to the practical range —
    the distance at which γ(h) reaches ~95% of the sill.
      Spherical   : practical range = a       (exact)
      Exponential : practical range ≈ 3a
      Gaussian    : practical range ≈ √3·a
    """
    if model_name == 'spherical':
        return range_param
    elif model_name == 'exponential':
        return 3.0 * range_param
    elif model_name == 'gaussian':
        return np.sqrt(3.0) * range_param
    else:
        return range_param


# =============================================================================
# DIAGNOSTIC SUMMARY
# =============================================================================

def variogram_diagnostics(result: Dict) -> Dict:
    """
    Extract key diagnostics from a tune_variogram result.

    Returns the structural ratio η = partial_sill / total_sill, which measures
    how much of the total variance is spatially structured (vs pure nugget noise).
    η < 0.10 triggers a fallback in auto_fit_variogram — not enough spatial
    continuity to simulate meaningfully.
    """
    best = result['best']
    p    = best['params']

    nugget       = p['nugget']
    partial_sill = p['partial_sill']
    total_sill   = p['total_sill']

    # Structural ratio: fraction of variance that has spatial continuity
    eta = partial_sill / total_sill if total_sill > 0 else 0.0
    pr  = practical_range(best['model'], p['range'])

    return {
        'model'               : best['model'],
        'nugget'              : nugget,
        'partial_sill'        : partial_sill,
        'total_sill'          : total_sill,
        'structural_ratio_eta': eta,      # η < 0.10 → near-pure nugget → fallback
        'practical_range'     : pr,
        'fitted_range_param'  : p['range'],
    }


# =============================================================================
# AUTOMATED VARIOGRAM FITTING FOR BATCH PIPELINE
# =============================================================================

def _estimate_vgm_params(lags_f, gamma_f):
    """
    Estimate apparent nugget, partial sill, and range directly from the
    experimental variogram. Used to initialize the optimizer near the
    actual data structure, preventing degenerate solutions (e.g. sill=992,
    range=153,000 m) that occur with uninformed starting points.
    """
    n = len(gamma_f)

    # Apparent sill: median of the upper half of lags (stabilized region)
    apparent_sill = float(np.median(gamma_f[max(n // 2, 2):]))

    # Apparent nugget: back-extrapolate the first rise to lag = 0
    if n >= 2 and lags_f[1] > lags_f[0]:
        slope = (gamma_f[1] - gamma_f[0]) / (lags_f[1] - lags_f[0])
        apparent_nugget = float(gamma_f[0] - slope * lags_f[0])
    else:
        apparent_nugget = 0.0
    apparent_nugget = float(np.clip(apparent_nugget, 0.0, apparent_sill * 0.9))

    apparent_partial = max(0.0, apparent_sill - apparent_nugget)

    # Apparent range: first lag where γ reaches 95% of apparent sill
    at_sill = lags_f[gamma_f >= 0.95 * apparent_sill]
    if len(at_sill) > 0:
        apparent_range = float(at_sill[0])
    else:
        # Variogram hasn't flattened — use lag where γ passes 60%
        at_half = lags_f[gamma_f >= 0.60 * apparent_sill]
        apparent_range = float(at_half[0]) * 1.4 if len(at_half) > 0 else float(lags_f[n // 2])

    return apparent_nugget, apparent_partial, apparent_range


def auto_fit_variogram(sampled_wells, col, fallback_range=100.0, xlag=25.0, nlag=40):
    """
    Compute an omnidirectional experimental variogram for `col`, fit a
    theoretical model, and return a geostatspy-compatible variogram dict
    ready for use in geostats.sgsim().

    Design decisions:
      - Optimizer initialized from data-informed estimates (nugget, sill, range)
        to prevent degenerate fits
      - Sill and range bounds constrained to the observable domain
      - Model selected by Cressie's WLS criterion
      - Near-pure nugget results (η < 0.10) trigger a safe fallback model

    Falls back to a unit-sill Spherical model with range=fallback_range if
    fitting fails or yields insufficient spatial structure.

    Parameters
    ----------
    sampled_wells  : DataFrame with columns X, Y, and `col`
    col            : column name of the variable to model (e.g. 'GP_GS_Residual')
    fallback_range : range (m) for the fallback Spherical model
    xlag           : lag spacing (m)
    nlag           : number of lags

    Returns
    -------
    vmodel : geostatspy variogram dict (from GSLIB.make_variogram)
    diag   : diagnostic dict from variogram_diagnostics
    """
    try:
        # Compute omnidirectional experimental variogram
        lags, gamma_exp, npairs = geostats.gamv(
            sampled_wells, "X", "Y", col,
            tmin=-999, tmax=999,
            xlag=xlag, xltol=xlag * 0.9, nlag=nlag,
            azm=0, atol=90, bandwh=9999.9, isill=0,
        )

        # Filter: require minimum pairs, positive lag, and positive semivariance
        mask = (npairs >= 20) & (lags > 0) & (gamma_exp > 0)
        lags_f, gamma_f, npairs_f = lags[mask], gamma_exp[mask], npairs[mask]

        if len(lags_f) < 4:
            raise ValueError(f"Too few valid lags ({len(lags_f)}) for '{col}'")

        max_lag = xlag * nlag   # maximum observable lag distance (m)

        # Estimate apparent parameters to initialize the optimizer
        nug_est, psill_est, range_est = _estimate_vgm_params(lags_f, gamma_f)
        sill_est  = nug_est + psill_est
        range_est = min(range_est, max_lag * 0.8)           # cap to observable domain
        sill_max  = max(sill_est * 1.5, gamma_f.max() * 1.1)  # generous headroom

        from variogram_tuner import spherical, exponential, gaussian_vgm

        # Build data-informed model space with three starting conditions per model:
        # (1) data-informed, (2) no-nugget variant, (3) range-shifted variant
        model_space = {
            "spherical": {
                "func"    : spherical,
                "inits"   : [
                    (nug_est,       psill_est,       range_est),
                    (0.0,           sill_est,        range_est * 0.5),
                    (nug_est * 0.5, psill_est * 1.1, range_est * 1.5),
                ],
                "bounds"  : ([0.0, 0.0, 1e-6], [sill_max, sill_max, max_lag]),
                "n_params": 3,
            },
            "exponential": {
                # range param ≈ practical_range / 3 for exponential model
                "func"    : exponential,
                "inits"   : [
                    (nug_est,       psill_est,       range_est / 3),
                    (0.0,           sill_est,        range_est / 3 * 0.5),
                    (nug_est * 0.5, psill_est * 1.1, range_est / 3 * 1.5),
                ],
                "bounds"  : ([0.0, 0.0, 1e-6], [sill_max, sill_max, max_lag / 3]),
                "n_params": 3,
            },
            "gaussian": {
                # range param ≈ practical_range / √3 for Gaussian model
                "func"    : gaussian_vgm,
                "inits"   : [
                    (nug_est,       psill_est,       range_est / 1.732),
                    (0.0,           sill_est,        range_est / 1.732 * 0.5),
                    (nug_est * 0.5, psill_est * 1.1, range_est / 1.732 * 1.5),
                ],
                "bounds"  : ([0.0, 0.0, 1e-6], [sill_max, sill_max, max_lag / 1.732]),
                "n_params": 3,
            },
        }

        result = tune_variogram(lags_f, gamma_f, npairs_f,
                                model_space=model_space,
                                criterion="wls_cressie")
        diag = variogram_diagnostics(result)

        # Reject near-pure nugget — insufficient spatial structure for simulation
        if diag["structural_ratio_eta"] < 0.10:
            raise ValueError(
                f"Near-pure nugget (η={diag['structural_ratio_eta']:.2f}) for '{col}'")

        # Convert to geostatspy variogram format (standardized sill = 1.0)
        best       = result["best"]
        _GSLIB_TYPE = {"spherical": 1, "exponential": 2, "gaussian": 3}
        total_sill = diag["nugget"] + diag["partial_sill"]
        nug_std    = diag["nugget"] / total_sill if total_sill > 1e-12 else 0.0
        cc1_std    = 1.0 - nug_std
        pr         = diag["practical_range"]
        it1        = _GSLIB_TYPE.get(best["model"], 1)

        vmodel = GSLIB.make_variogram(
            nug=nug_std, nst=1, it1=it1, cc1=cc1_std,
            azi1=0, hmaj1=pr, hmin1=pr,
        )
        return vmodel, diag

    except Exception as e:
        # Fallback: unit-sill Spherical with user-specified range
        print(f"    [auto_fit_variogram] Warning ({col}): {e} "
              f"=> fallback Spherical(range={fallback_range:.0f}m)")
        diag_fb = {
            "model"               : "Spherical (fallback)",
            "nugget"              : 0.0,
            "partial_sill"        : 1.0,
            "total_sill"          : 1.0,
            "structural_ratio_eta": 1.0,
            "practical_range"     : fallback_range,
        }
        vmodel = GSLIB.make_variogram(
            nug=0.0, nst=1, it1=1, cc1=1.0,
            azi1=0, hmaj1=fallback_range, hmin1=fallback_range,
        )
        return vmodel, diag_fb