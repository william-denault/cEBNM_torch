import numpy as np
import scipy.stats as stats

def do_truncnorm_argchecks(a, b):
    # Ensure a and b are numpy arrays, even if they are scalars
    a = np.atleast_1d(np.asarray(a))
    b = np.atleast_1d(np.asarray(b))
    if len(a) != len(b):
        raise ValueError("truncnorm functions require that a and b have the same length.")
    if np.any(b < a):
        raise ValueError("truncnorm functions require that a <= b.")
    return a, b

def logscale_sub(logx, logy, epsilon=1e-10):
    diff = logx - logy
    if np.any(diff < 0):
        bad_idx = diff < 0
        logx[bad_idx] = logy[bad_idx]
        print(f"logscale_sub encountered negative value(s) of logx - logy (min: {np.min(diff[bad_idx]):.2e})")
    
    scale_by = logx.copy()
    scale_by[np.isinf(scale_by)] = 0
    return np.log(np.exp(logx - scale_by) - np.exp(logy - scale_by) + epsilon) + scale_by

def my_etruncnorm(a, b, mean=0, sd=1):
    a, b = do_truncnorm_argchecks(a, b)
    
    alpha = (a - mean) / sd
    beta = (b - mean) / sd
    
    flip = (alpha > 0) & (beta > 0) | (beta > np.abs(alpha))
    orig_alpha = alpha.copy()
    alpha[flip] = -beta[flip]
    beta[flip] = -orig_alpha[flip]
    
    dnorm_diff = logscale_sub(stats.norm.logpdf(beta), stats.norm.logpdf(alpha))
    pnorm_diff = logscale_sub(stats.norm.logcdf(beta), stats.norm.logcdf(alpha))
    scaled_res = -np.exp(dnorm_diff - pnorm_diff)
    
    endpts_equal = np.isinf(pnorm_diff)
    scaled_res[endpts_equal] = (alpha[endpts_equal] + beta[endpts_equal]) / 2
    
    lower_bd = np.maximum(beta + 1 / beta, (alpha + beta) / 2)
    bad_idx = (~np.isnan(beta)) & (beta < 0) & ((scaled_res < lower_bd) | (scaled_res > beta))
    scaled_res[bad_idx] = lower_bd[bad_idx]
    
    scaled_res[flip] = -scaled_res[flip]
    
    res = mean + sd * scaled_res
    
    if np.any(sd == 0):
        a = np.tile(a, len(res))
        b = np.tile(b, len(res))
        mean = np.tile(mean, len(res))
        
        sd_zero = (sd == 0)
        res[sd_zero & (b <= mean)] = b[sd_zero & (b <= mean)]
        res[sd_zero & (a >= mean)] = a[sd_zero & (a >= mean)]
        res[sd_zero & (a < mean) & (b > mean)] = mean[sd_zero & (a < mean) & (b > mean)]
    
    return res

def my_e2truncnorm(a, b, mean=0, sd=1):
    a, b = do_truncnorm_argchecks(a, b)
    
    alpha = (a - mean) / sd
    beta = (b - mean) / sd
    
    flip = (alpha > 0) & (beta > 0)
    orig_alpha = alpha.copy()
    alpha[flip] = -beta[flip]
    beta[flip] = -orig_alpha[flip]
    
    if np.any(mean != 0):
        mean = np.tile(mean, len(alpha))
        mean[flip] = -mean[flip]
    
    pnorm_diff = logscale_sub(stats.norm.logcdf(beta), stats.norm.logcdf(alpha))
    alpha_frac = alpha * np.exp(stats.norm.logpdf(alpha) - pnorm_diff)
    beta_frac = beta * np.exp(stats.norm.logpdf(beta) - pnorm_diff)
    
    # Mask invalid values
    alpha_frac = np.where(np.isnan(alpha_frac), 0, alpha_frac)
    beta_frac = np.where(np.isnan(beta_frac), 0, beta_frac)
    
    scaled_res = np.ones_like(alpha)
    scaled_res[np.isnan(flip)] = np.nan
    
    alpha_idx = np.isfinite(alpha)
    scaled_res[alpha_idx] = 1 + alpha_frac[alpha_idx]
    beta_idx = np.isfinite(beta)
    scaled_res[beta_idx] -= beta_frac[beta_idx]
    
    endpts_equal = np.isinf(pnorm_diff)
    scaled_res[endpts_equal] = ((alpha[endpts_equal] + beta[endpts_equal]) ** 2) / 4
    
    upper_bd1 = beta ** 2 + 2 * (1 + 1 / beta ** 2)
    upper_bd2 = (alpha ** 2 + alpha * beta + beta ** 2) / 3
    upper_bd1 = np.where(np.isnan(upper_bd1), np.inf, upper_bd1)
    upper_bd2 = np.where(np.isnan(upper_bd2), np.inf, upper_bd2)
    upper_bd = np.minimum(upper_bd1, upper_bd2)
    bad_idx = (~np.isnan(beta)) & (beta < 0) & ((scaled_res < beta ** 2) | (scaled_res > upper_bd))
    scaled_res[bad_idx] = upper_bd[bad_idx]
    
    res = mean ** 2 + 2 * mean * sd * my_etruncnorm(alpha, beta) + sd ** 2 * scaled_res
    
    if np.any(sd == 0):
        a = np.tile(a, len(res))
        b = np.tile(b, len(res))
        mean = np.tile(mean, len(res))
        
        sd_zero = (sd == 0)
        res[sd_zero & (b <= mean)] = b[sd_zero & (b <= mean)] ** 2
        res[sd_zero & (a >= mean)] = a[sd_zero & (a >= mean)] ** 2
        res[sd_zero & (a < mean) & (b > mean)] = mean[sd_zero & (a < mean) & (b > mean)] ** 2
    
    return res

def my_vtruncnorm(a, b, mean=0, sd=1):
    a, b = do_truncnorm_argchecks(a, b)
    
    alpha = (a - mea
