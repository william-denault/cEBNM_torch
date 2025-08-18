# ebnm_point_exponential.py
# Point–Exponential Empirical Bayes for Normal means:
# Prior: (1 - w) * δ_{mu}  +  w * Exp(rate=a) on theta >= 0
# Likelihood: X | theta ~ N(theta, s^2)
# Parameters optimized on unconstrained scales: alpha (w = sigmoid(alpha)),
#                                              beta  (a = exp(beta)),
#                                              mu    (free).

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.stats import norm
from scipy.optimize import minimize

# =========================
# Numeric helpers
# =========================

_LOG2PI = np.log(2.0 * np.pi)

def _log_phi(z: np.ndarray) -> np.ndarray:
    return -0.5 * z**2 - 0.5 * _LOG2PI

def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(_log_phi(z))

def _log_Phi(z: np.ndarray) -> np.ndarray:
    # stable log CDF
    return norm.logcdf(z)

def _logaddexp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # stable log(exp(x) + exp(y))
    return np.logaddexp(x, y)

def _logdiffexp(logB: np.ndarray, logA: np.ndarray) -> np.ndarray:
    # log(exp(logB) - exp(logA)), assuming logB >= logA
    return logB + np.log1p(-np.exp(logA - logB))

# =========================
# Truncated normal moments
# X ~ N(m, s^2), a < X < b
# =========================

def _etruncnorm(a: np.ndarray, b: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    E[X | a < X < b], robust to extreme tails.
    a,b,m,s broadcast; a or b can be +/- np.inf.
    """
    s = np.maximum(s, np.finfo(float).tiny)
    alpha = (a - m) / s
    beta  = (b - m) / s

    swap = beta < alpha
    alpha2 = np.where(swap, beta, alpha)
    beta2  = np.where(swap, alpha, beta)

    logPhiA = _log_Phi(alpha2)
    logPhiB = _log_Phi(beta2)
    logZ    = _logdiffexp(logPhiB, logPhiA)
    Z       = np.maximum(np.exp(logZ), np.finfo(float).tiny)

    num = _phi(alpha2) - _phi(beta2)
    t = num / Z
    # limit 0 when both num and Z ~ 0
    t = np.where((np.abs(num) <= np.finfo(float).tiny) & (Z <= np.finfo(float).tiny), 0.0, t)
    return m + s * t

def _e2truncnorm(a: np.ndarray, b: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    E[X^2 | a < X < b], X ~ N(m, s^2), robust for infinite bounds.
    """
    s = np.maximum(s, np.finfo(float).tiny)

    alpha = (a - m) / s
    beta  = (b - m) / s

    swap = beta < alpha
    alpha2 = np.where(swap, beta, alpha)
    beta2  = np.where(swap, alpha, beta)

    logPhiA = norm.logcdf(alpha2)
    logPhiB = norm.logcdf(beta2)
    # logZ = log(Phi(beta) - Phi(alpha)) (stable)
    logZ = logPhiB + np.log1p(-np.exp(logPhiA - logPhiB))
    Z = np.maximum(np.exp(logZ), np.finfo(float).tiny)

    # standard normal pdfs
    phi_a = np.exp(-0.5 * alpha2**2 - 0.5 * np.log(2*np.pi))
    phi_b = np.exp(-0.5 * beta2**2  - 0.5 * np.log(2*np.pi))

    # SAFE products: treat inf*0 -> 0 by masking infs before multiply
    alpha_safe = np.where(np.isfinite(alpha2), alpha2, 0.0)
    beta_safe  = np.where(np.isfinite(beta2),  beta2,  0.0)
    term1 = alpha_safe * phi_a - beta_safe * phi_b

    # (phi_a - phi_b)/Z with 0/0 -> 0
    num_phi = phi_a - phi_b
    term2 = num_phi / Z
    tiny = np.finfo(float).tiny
    term2 = np.where((np.abs(num_phi) <= tiny) & (Z <= tiny), 0.0, term2)

    var  = s**2 * (1.0 + term1 / Z - term2**2)

    # mean uses the same robust machinery
    mean = _etruncnorm(a, b, m, s)
    return var + mean**2


# =========================
# Convolution: Exp(a) * N(0, s^2)
# For theta >= 0 with Exp(rate=a)
# =========================

def _log_g_exp(x_minus_mu: np.ndarray, s: np.ndarray, a: float) -> np.ndarray:
    """
    log g(x) where g = Exp(rate=a) on theta>=0 convolved with N(0, s^2),
    evaluated at x - mu.
    Formula: log a + 0.5 s^2 a^2 - a (x - mu) + logPhi( (x-mu)/s - s a )
    """
    z = x_minus_mu / s - s * a
    return np.log(a) + 0.5 * (s**2) * (a**2) - a * x_minus_mu + _log_Phi(z)

# =========================
# Posterior responsibility for the non-zero component
# =========================

def wpost_pe(x: np.ndarray, s: np.ndarray, w: float, a: float, mu: float = 0.0) -> np.ndarray:
    """
    Posterior probability of the Exp component:
      wpost = w * g(x) / ((1-w)*f(x) + w*g(x))
    where f is N(mu, s^2), g is the exp-normal convolution.
    Computed in log-space for stability.
    """
    if w == 0.0:
        return np.zeros_like(x, dtype=float)
    if w == 1.0:
        return np.ones_like(x, dtype=float)

    xm = x - mu
    lf = norm.logpdf(x, loc=mu, scale=s)             # log f
    lg = _log_g_exp(xm, s, a)                        # log g

    log_num = np.log(w)     + lg
    log_den = _logaddexp(np.log(1.0 - w) + lf, log_num)
    return np.exp(log_num - log_den)

# =========================
# Posterior moments
# =========================

@dataclass
class PosteriorMeanPointExp:
    post_mean: np.ndarray
    post_mean2: np.ndarray
    post_sd: np.ndarray

def posterior_mean_pe(x: np.ndarray, s: np.ndarray, w: float, a: float, mu: float = 0.0) -> PosteriorMeanPointExp:
    """
    Compute posterior mean, second moment, and sd of Theta_total = theta + mu
    under prior (1-w) δ_mu + w * Exp(rate=a) with support theta>=0.
    """
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)

    wpost = wpost_pe(x, s, w, a, mu)

    # For the non-zero component: theta|x ~ N(m_pos, s^2) truncated to (0, +inf)
    m_pos = (x - mu) - (s**2) * a
    e1 = _etruncnorm(0.0, np.inf, m_pos, s)     # E[theta | nonzero, x]
    e2 = _e2truncnorm(0.0, np.inf, m_pos, s)    # E[theta^2 | nonzero, x]

    E_theta  = wpost * e1
    E_theta2 = wpost * e2

    # s = inf: fallback to prior moments of Exp(rate=a)
    inf_mask = np.isinf(s)
    if inf_mask.any():
        E_theta[inf_mask]  = wpost[inf_mask] * (1.0 / a)
        E_theta2[inf_mask] = wpost[inf_mask] * (2.0 / (a**2))

    # Ensure second moment >= square of mean (numerically)
    E_theta2 = np.maximum(E_theta2, E_theta**2)
    post_sd  = np.sqrt(np.maximum(0.0, E_theta2 - E_theta**2))

    # Shift back by mu:
    post_mean  = E_theta  + mu
    post_mean2 = E_theta2 + 2.0 * mu * E_theta + mu**2

    return PosteriorMeanPointExp(post_mean=post_mean, post_mean2=post_mean2, post_sd=post_sd)

# =========================
# Negative log-likelihood + gradients
# =========================

def pe_nllik(params_free: np.ndarray, x: np.ndarray, s: np.ndarray,
             fix_par: Tuple[bool, bool, bool] = (False, False, False),
             par_init_full: Optional[np.ndarray] = None,
             calc_grad: bool = True):
    """
    NLL under point-exponential model, optionally with analytic gradients.
    params_free are the *free* parameters corresponding to (~fix_par):
      full p = [alpha, beta, mu]; w = sigmoid(alpha), a = exp(beta).
    """
    # Build full parameter vector p from free + fixed
    if par_init_full is None:
        p_full = np.zeros(3, dtype=float)
    else:
        p_full = np.array(par_init_full, dtype=float)
    p_full[~np.array(fix_par, dtype=bool)] = np.asarray(params_free, dtype=float)

    alpha, beta, mu = p_full
    w = 1.0 / (1.0 + np.exp(-alpha))          # sigmoid
    a = np.exp(beta)

    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)

    # log-likelihood per-point
    lf = norm.logpdf(x, loc=mu, scale=s)
    lg = _log_g_exp(x - mu, s, a)
    llik = _logaddexp(np.log(1.0 - w) + lf, np.log(w) + lg)
    nll = -np.sum(llik)

    if not calc_grad:
        return nll

    # responsibilities (posterior over components given current params)
    # r_f + r_g = 1
    r_f = np.exp(lf - llik)    # responsibility of point-mass component
    r_g = np.exp(lg - llik)    # responsibility of exponential component

    # d nll / d alpha (via w):  d nll / d w = sum(f - g); dw/dalpha = w(1-w)
    grad_full = np.zeros_like(p_full)
    if not fix_par[0]:
        dw_dalpha = w * (1.0 - w)
        d_nll_dw  = np.sum(r_f - r_g)
        grad_full[0] = d_nll_dw * dw_dalpha

    # Helpful quantities for derivatives wrt a and mu
    z = (x - mu) / s - s * a
    # ratio phi(z)/Phi(z) computed stably in log-space; clip to avoid overflow
    log_ratio = _log_phi(z) - _log_Phi(z)
    ratio = np.exp(np.clip(log_ratio, a_min=-745, a_max=+745))  # ~exp bounds for float64

    # d lg / d a = 1/a + s^2 a - (x - mu) - s * ratio
    d_lg_da = (1.0 / a) + (s**2) * a - (x - mu) - s * ratio
    # d lg / d mu = a - (1/s) * ratio
    d_lg_dmu = a - ratio / s
    # d lf / d mu = (x - mu) / s^2
    d_lf_dmu = (x - mu) / (s**2)

    if not fix_par[1]:
        # d nll / d a = - sum r_g * d lg / d a   ;  da/dbeta = a
        d_nll_da = -np.sum(r_g * d_lg_da)
        grad_full[1] = d_nll_da * a

    if not fix_par[2]:
        # d nll / d mu = - sum [ (1-w) responsibility * d lf/dmu + w responsibility * d lg/dmu ]
        d_nll_dmu = -np.sum((1.0 - w) * (r_f * d_lf_dmu) + w * (r_g * d_lg_dmu))
        # Equivalent: -sum[ r_f * d lf/dmu + r_g * d lg/dmu ]
        # (since r_f,r_g already include mixing through llik), but both are fine.
        grad_full[2] = d_nll_dmu

    # return only free gradients
    return nll, grad_full[~np.array(fix_par, dtype=bool)]

# =========================
# Optimizer wrapper
# =========================

@dataclass
class OptimizePointExponential:
    w: float
    a: float
    mu: float
    nllik: float

def optimize_pe_nllik_with_gradient(x: np.ndarray, s: np.ndarray,
                                    par_init: Tuple[float, float, float],
                                    fix_par: Tuple[bool, bool, bool]) -> OptimizePointExponential:
    """
    Optimize (alpha,beta,mu) on unconstrained scales using L-BFGS-B with analytic gradients.
    par_init are the full parameters [alpha, beta, mu], regardless of fix_par.
    """
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)

    # Extract free subset
    p0 = np.asarray(par_init, dtype=float)[~np.array(fix_par, dtype=bool)]

    def fun(pfree):
        nll, _ = pe_nllik(pfree, x, s, fix_par=fix_par, par_init_full=np.asarray(par_init, dtype=float), calc_grad=True)
        return nll

    def jac(pfree):
        _, g = pe_nllik(pfree, x, s, fix_par=fix_par, par_init_full=np.asarray(par_init, dtype=float), calc_grad=True)
        return g

    # Reasonable bounds only for beta if you like; alpha, mu unbounded.
    bounds = []
    k = 0
    for fixed in fix_par:
        if not fixed:
            if k == 1:
                bounds.append((None, None))  # beta unrestricted; exp(beta) > 0 anyway
            else:
                bounds.append((None, None))
        k += 1

    res = minimize(fun, p0, method="L-BFGS-B", jac=jac, bounds=bounds)

    # Reconstruct full parameters
    p_full = np.array(par_init, dtype=float)
    p_full[~np.array(fix_par, dtype=bool)] = res.x
    alpha, beta, mu = p_full
    w = 1.0 / (1.0 + np.exp(-alpha))
    a = float(np.exp(beta))
    return OptimizePointExponential(w=float(w), a=a, mu=float(mu), nllik=float(res.fun))

# =========================
# One-shot solver
# =========================

@dataclass
class EBNMPointExpResult:
    post_mean: np.ndarray
    post_mean2: np.ndarray
    post_sd: np.ndarray
    scale: float
    pi: float
    log_lik: float
    mode: float

def ebnm_point_exp_solver(x: np.ndarray, s: np.ndarray,
                          opt_mu: bool = False,
                          par_init: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> EBNMPointExpResult:
    """
    Fit point–exponential EBNM and return posterior summaries.
    par_init are [alpha, beta, mu] (unconstrained scales).
    If opt_mu=False, mu is fixed at par_init[2].
    """
    fix_par = (False, False, not opt_mu)

    opt = optimize_pe_nllik_with_gradient(x, s, par_init=par_init, fix_par=fix_par)
    post = posterior_mean_pe(x, s, w=opt.w, a=opt.a, mu=opt.mu)

    # compute final log-likelihood (with fitted params)
    lf = norm.logpdf(x, loc=opt.mu, scale=s)
    lg = _log_g_exp(x - opt.mu, s, opt.a)
    llik = _logaddexp(np.log(1.0 - opt.w) + lf, np.log(opt.w) + lg).sum()

    return EBNMPointExpResult(
        post_mean = post.post_mean,
        post_mean2= post.post_mean2,
        post_sd   = post.post_sd,
        scale     = opt.a,
        pi        = opt.w,
        log_lik   = float(llik),
        mode      = opt.mu
    )
 