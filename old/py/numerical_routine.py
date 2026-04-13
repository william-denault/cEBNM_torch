import numpy as np
import math 
from scipy.stats import norm
from scipy.stats import truncnorm
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import logsumexp

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
 
    scaled_res = -np.exp( np.clip( dnorm_diff - pnorm_diff, None, 700))
    
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
    
    mean = np.atleast_1d(mean)
    sd   = np.atleast_1d(sd)
    alpha = (a - mean) / sd
    beta = (b - mean) / sd
    
    flip = (alpha > 0) & (beta > 0)
    orig_alpha = alpha.copy()
    alpha[flip] = -beta[flip]
    beta[flip] = -orig_alpha[flip]
    
    if np.any(mean != 0):
         
        mean  =  abs(mean)
    
    pnorm_diff = logscale_sub(stats.norm.logcdf(beta), stats.norm.logcdf(alpha))
    alpha_frac = alpha * np.exp( np.clip(stats.norm.logpdf(alpha) - pnorm_diff, None, 300))
 
    beta_frac = beta * np.exp( np.clip(stats.norm.logpdf(beta) - pnorm_diff, None, 300))
    
    # Handle inf and nan values in alpha_frac and beta_frac
    alpha_frac[np.isnan(alpha_frac) | np.isinf(alpha_frac)] = 0
    beta_frac[np.isnan(beta_frac) | np.isinf(beta_frac)] = 0
    
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



def apply_log_sum_exp(data_loglik, log_pi):
    combined_loglik = data_loglik + log_pi
    return combined_loglik - logsumexp(combined_loglik, axis=1)[:, None]

def log_sum_exp(lx, idxs=None, na_rm=False):
    # Convert input to a numpy array if it's not already
    lx = np.asarray(lx, dtype=np.float64)
    
    # If indices are provided, select the elements at those indices
    if idxs is not None:
        lx = lx[idxs]
    
    # Handle NaN values if na_rm is True
    if na_rm:
        lx = lx[~np.isnan(lx)]
    
    # To prevent overflow/underflow issues, subtract the max value from lx before exponentiating
    max_lx = np.max(lx)
    if np.isinf(max_lx):
        return max_lx  # Return max_lx if it's inf or -inf
    
    # Compute the log-sum-exp
    sum_exp = np.sum(np.exp(lx - max_lx))
    log_sum_exp_result = max_lx + np.log(sum_exp)
    
    return log_sum_exp_result

import numpy as np
from scipy.optimize import minimize

def penalized_log_likelihood(pi, L_batch, penalty, epsilon=1e-10):
    """
    Compute the penalized log likelihood function using clamping to avoid log(0) or log(negative).
    
    Parameters:
    pi (numpy.ndarray): A vector of length K corresponding to pi_k.
    L_batch (numpy.ndarray): A minibatch matrix of shape (batch_size, K) where each entry corresponds to l_kj.
    penalty (float): The penalty term.
    epsilon (float): Small constant to avoid log of zero or division by zero.
    
    Returns:
    float: The negative penalized log likelihood (for minimization purposes).
    """
    batch_size, K = L_batch.shape

    # Initialize the first summation (over j)
    first_sum = 0
    for j in range(batch_size):
        inner_sum = 0
        for k in range(K):
            inner_sum += pi[k] * L_batch[j, k]
        
        # Ensure inner_sum is not too small to avoid log(0) or log(negative)
        inner_sum = np.clip(inner_sum, epsilon, None)
        first_sum += np.log(inner_sum)
    
    # Add penalty term if applicable
    if penalty > 1:
        pi_clamped = np.clip(pi[0], epsilon, None)  # Clamp pi[0] to avoid log of zero
        penalized_log_likelihood_value = first_sum + (penalty - 1) * np.log(pi_clamped)
    else:
        penalized_log_likelihood_value = first_sum

    # Return the negative since we are minimizing
    return -penalized_log_likelihood_value

def constraint_sum_to_one(pi):
    """
    Constraint to ensure that the sum of pi equals 1 (simplex constraint).
    """
    return np.sum(pi) - 1

def sample_minibatch(L, batch_size):
    """
    Randomly sample a minibatch of rows from L.
    
    Parameters:
    L (numpy.ndarray): The full data matrix of shape (n, K).
    batch_size (int): The size of the minibatch to sample.
    
    Returns:
    L_batch (numpy.ndarray): A minibatch of shape (batch_size, K) or the entire dataset if n < batch_size.
    """
    n = L.shape[0]
    if n < batch_size:
        return L  # If the dataset is smaller than the batch size, use the entire dataset
    batch_indices = np.random.choice(n, batch_size, replace=False)
    return L[batch_indices, :]

 



def optimize_pi(L, penalty, max_iters=100, tol=1e-6, verbose= True):
    """
    EM algorithm from Stephens 2016 Biostatistics for optimizing pi subject to the simplex constraint that pi lies in the K-dimensional simplex.
    
    Parameters:
    L (numpy.ndarray): The likelihood matrix with shape (n, K) where L[j, k] corresponds to l_kj.
    penalty (float): The penalty parameter.
    max_iters (int): The maximum number of iterations for optimization.
    tol (float): The tolerance for convergence.
    
    Returns:
    numpy.ndarray: The optimized pi values as a 1D numpy array.
    """
    n, K = L.shape  # n: number of data points, K: number of components
    pi =np.exp( - np.arange(0,K) )/ np.sum( np.exp( - np.arange(0,K) ))  # Initialize pi uniformly
    vec_pen= np.ones_like(pi)
    vec_pen[0]= penalty
    # EM algorithm iterations
    for iteration in range(max_iters):
        # E-Step: calculate responsibilities (w_kj)
        w =  pi[:, np.newaxis] * L.T   +1e-32 # Element-wise multiplication (pi_k * l_kj)
        w = w / w.sum(axis=0, keepdims=True)  # Normalize by sum over k for each j

        # M-Step: update pi
        n_k = w.sum(axis=1) + vec_pen - 1  # Apply penalty (S.2.8)
        pi_new = n_k / n_k.sum()  # Normalize to make sure sum(pi) = 1 (S.2.9)

        # Check for convergence
        if np.linalg.norm(pi_new - pi) < tol:
            if verbose:
                print(f"Converged after {iteration} iterations.")
            break

        # Update pi
        pi = pi_new

    return pi

# Example usage:
# L is your likelihood matrix (shape n x K)
# penalty is a scalar penalty parameter, for example: penalty = 0.1
# pi = em_algorithm(L, penalty)
def optimize_pi_logL(logL, penalty, max_iters=100, tol=1e-6, verbose=True):
    """
    EM algorithm based on the log likelihood for optimizing pi subject to the simplex constraint 
    that pi lies in the K-dimensional simplex.

    Parameters:
    logL (numpy.ndarray): The log-likelihood matrix with shape (n, K) where logL[j, k] corresponds to log(l_kj).
    penalty (float): The penalty parameter.
    max_iters (int): The maximum number of iterations for optimization.
    tol (float): The tolerance for convergence.
    
    Returns:
    numpy.ndarray: The optimized pi values as a 1D numpy array.
    """
    n, K = logL.shape  # n: number of data points, K: number of components
    pi = np.exp(-np.arange(0, K)) / np.sum(np.exp(-np.arange(0, K)))  # Initialize pi uniformly
    vec_pen = np.ones_like(pi)
    vec_pen[0] = penalty
    epsilon = 1e-10  # Small constant to avoid log(0)
    
    # EM algorithm iterations
    for iteration in range(max_iters):
        # E-Step: calculate responsibilities (w_kj)
        log_w = np.log(pi[:, np.newaxis] + epsilon) + logL.T  # Avoid log(0) by adding epsilon
        log_w = log_w - np.max(log_w, axis=0, keepdims=True)  # For numerical stability (log-sum-exp trick)
        w = np.exp(log_w)
        w = w / w.sum(axis=0, keepdims=True)  # Normalize by sum over k for each j

        # M-Step: update pi
        n_k = w.sum(axis=1) + vec_pen - 1  # Apply penalty
        pi_new = n_k / n_k.sum()  # Normalize to make sure sum(pi) = 1

        # Check for convergence
        if np.linalg.norm(pi_new - pi) < tol:
            if verbose:
                print(f"Converged after {iteration} iterations.")
            break

        # Update pi
        pi = pi_new

    return pi
