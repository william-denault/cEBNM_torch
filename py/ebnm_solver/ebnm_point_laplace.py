import numpy as np
from scipy.stats import norm
import sys
import os
import matplotlib.pyplot as plt

# Add the path to utils.py
sys.path.append(r"D:\Document\Serieux\Travail\python_work\cEBNM_torch\py")
from numerical_routine import *
from scipy.stats import norm
import numpy as np

def logg_laplace(x, s, a):
    """
    Compute the log of g, Laplace(a) convolved with a normal distribution.
    
    Args:
        x (np.ndarray): Input data.
        s (float or np.ndarray): Standard deviation of the normal distribution.
        a (float): Scale parameter of the Laplace distribution.
    
    Returns:
        np.ndarray: Log-Laplace density values.
    """
    lg1 = -a * x + norm.logcdf((x - s**2 * a) / s)
    lg2 = a * x + norm.logcdf(-(x + s**2 * a) / s)  # Upper tail as -x
    lfac = np.maximum(lg1, lg2)
    return np.log(a / 2) + s**2 * a**2 / 2 + lfac + np.log(np.exp(lg1 - lfac) + np.exp(lg2 - lfac))
def loglik_point_laplace(x, s, w, a, mu):
    """
    Compute the log likelihood under the point-laplace prior.
    Args:
        x: Observed data (array-like).
        s: Standard deviation (scalar or array-like).
        w: Weight for the mixture component (scalar).
        a: Laplace scale parameter (scalar).
        mu: Mean (scalar).
    Returns:
        Log likelihood (scalar).
    """
    return np.sum(vloglik_point_laplace(x, s, w, a, mu))

# Helper function to compute log((1 - w)f + wg) as a vector
def vloglik_point_laplace(x, s, w, a, mu):
    """
    Compute the vectorized log likelihood under the point-laplace prior.
    Args:
        x: Observed data (array-like).
        s: Standard deviation (scalar or array-like).
        w: Weight for the mixture component (scalar).
        a: Laplace scale parameter (scalar).
        mu: Mean (scalar).
    Returns:
        Log likelihood vector (array-like).
    """
    if w <= 0:
        return norm.logpdf(x - mu, scale=s)

    lg = logg_laplace(x - mu, s, a)
    if w == 1:
        return lg

    lf = norm.logpdf(x - mu, scale=s)
    lfac = np.maximum(lg, lf)
    return lfac + np.log((1 - w) * np.exp(lf - lfac) + w * np.exp(lg - lfac))

def wpost_laplace(x, s, w, a):
    """
    Compute the posterior weights for the Laplace component.
    
    Args:
        x (np.ndarray): Input data.
        s (float or np.ndarray): Standard deviation of the normal component.
        w (float): Weight for the Laplace component (mixture proportion).
        a (float): Scale parameter of the Laplace distribution.
    
    Returns:
        np.ndarray: Posterior weights for the Laplace component.
    """
    if w == 0:
        return np.zeros_like(x)

    if w == 1:
        return np.ones_like(x)

    # Log-density for the normal component
    lf = norm.logpdf(x, loc=0, scale=s)

    # Log-density for the Laplace component convolved with normal
    lg = logg_laplace(x, s, a)

    # Compute posterior weights
    wpost = w / (w + (1 - w) * np.exp(lf - lg))
    
    return wpost


def lambda_func(x, s, a):
    """
    Compute lambda, the probability of being positive given a non-zero effect.
    
    Args:
        x (np.ndarray): Input data.
        s (np.ndarray): Standard deviations.
        a (float): Laplace scale parameter.
    
    Returns:
        np.ndarray: Lambda values.
    """
    lm1 = -a * x + norm.logcdf(x / s - s * a)
    lm2 = a * x + norm.logcdf(-(x / s + s * a))
    return 1 / (1 + np.exp(lm2 - lm1))



class PosteriorMeanPointLapalce:
    def __init__(self, post_mean, post_mean2, post_sd):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd

def posterior_mean_laplace(x, s, w, a, mu=0):
    """
    Compute the posterior mean for a normal mean under a Laplace prior.
    
    Args:
        x (np.ndarray): Observed data.
        s (float or np.ndarray): Standard deviation of the normal likelihood.
        w (float): Mixture weight for the Laplace component.
        a (float): Laplace scale parameter.
        mu (float): Mean of the prior distribution (default is 0).
    
    Returns:
        np.ndarray: Posterior means.
    """
    # Compute posterior weights
    wpost = wpost_laplace(x - mu, s, w, a)
     
    # Compute lambda (probability of being positive given non-null)
    lm = lambda_func(x - mu, s, a)
    
    # Compute the truncated means for the Laplace component
    laplace_mean_positive = my_etruncnorm(0, 99999 ,x - mu - s**2 * a, s)
    laplace_mean_negative = my_etruncnorm(-99999, 0, x - mu + s**2 * a, s)
    laplace_component_mean = lm * laplace_mean_positive + (1 - lm) * laplace_mean_negative
    post_mean2              =  wpost * (lm * my_e2truncnorm(0, 99999, x - mu - s**2 * a, s)
                                       + (1 - lm) * my_e2truncnorm(-99999, 0, x - mu + s**2 * a, s))

    
    # Combine posterior means
    post_mean =    wpost * laplace_component_mean  

    if np.any(np.isinf(s)):
        inf_indices = np.isinf(s)
        a = 1/scale[1:]
        # Equivalent of `post$mean[is.infinite(s)]` 
        post_mean[inf_indices] = wpost  / a 

        # Equivalent of `post$mean2[is.infinite(s)]`
        post_mean2[inf_indices] =  2 * wpost / a**2 

    post_mean2 = np.maximum(post_mean2, post_mean  ** 2)
    
    post_sd= np.sqrt(np.maximum(0, post_mean2 - post_mean**2))
    post_mean2 = post_mean2 + mu**2+ 2*mu*post_mean
    post_mean  =  post_mean+mu  
    return PosteriorMeanPointLapalce(post_mean=post_mean,
                                     post_mean2=post_mean2,
                                     post_sd=  post_sd)



import numpy as np
from scipy.stats import norm, truncnorm
from scipy.special import logsumexp


class LaplaceMixture:
    def __init__(self, pi, mean, scale):
        """
        Constructor for LaplaceMixture class.
        
        Args:
            pi (np.ndarray): Mixture proportions.
            mean (np.ndarray): Means of components.
            scale (np.ndarray): Scale parameters of components.
        """
        self.pi = np.asarray(pi)
        self.mean = np.asarray(mean)
        self.scale = np.asarray(scale)

    def __repr__(self):
        return f"LaplaceMixture(pi={self.pi}, mean={self.mean}, scale={self.scale})"
def wpost_laplace(x, s, w, a):
    """
    Compute posterior weights for non-null effects.
    
    Args:
        x (np.ndarray): Input data.
        s (float): Standard deviation.
        w (float): Weight for the mixture component.
        a (float): Scale parameter.
    
    Returns:
        np.ndarray: Posterior weights.
    """
    if w == 0:
        return np.zeros(len(x))
    if w == 1:
        return np.ones(len(x))

    lf = norm.logpdf(x, loc=0, scale=s)
    lg = logg_laplace(x, s, a)
    return w / (w + (1 - w) * np.exp(lf - lg))


def pl_nllik(par, x, s,  calc_grad=False):
    """
    Compute the negative log likelihood and its gradient.
    
    Args:
        par (dict): Parameters (alpha, beta, mu).
        x (np.ndarray): Observed data.
        s (np.ndarray): Standard deviation.
        fix_par (list): Fixed parameters.
        calc_grad (bool): Whether to calculate gradients.
    
    Returns:
        float: Negative log likelihood.
    """
    alpha, beta, mu = par['alpha'], par['beta'], par['mu']

    # Convert alpha and beta to w and a
    w = 1 - 1 / (1 + np.exp(alpha))
    a = np.exp(beta)

    lf = norm.logpdf(x - mu, scale=s)
    lg = logg_laplace(x - mu, s, a)
    llik = np.log((1 - w) * np.exp(lf) + w * np.exp(lg))
    nllik = -np.sum(llik)

    if calc_grad:
        # Gradient calculations (optional)
        pass

    return nllik
def logscale_add(logx, logy):
    """
    Compute the log of the sum of exponentials of logx and logy.
    
    Args:
        logx (float): Log value.
        logy (float): Log value.
    
    Returns:
        float: Log-sum-exp.
    """
    max_log = np.maximum(logx, logy)
    return max_log + np.log(np.exp(logx - max_log) + np.exp(logy - max_log))



def logscale_add(logx, logy):
    """
    Compute log(exp(logx) + exp(logy)) in a numerically stable way.
    """
    max_log = np.maximum(logx, logy)
    return max_log + np.log(np.exp(logx - max_log) + np.exp(logy - max_log))

def pl_nllik(par, x, s, par_init, fix_par, calc_grad=False, calc_hess=False):
    """
    Compute the negative log likelihood and optionally its gradient and Hessian.

    Args:
        par (list): Parameters to optimize (subset based on fix_par).
        x (np.ndarray): Observed data.
        s (np.ndarray): Standard deviations.
        par_init (list): Initial parameters (full set).
        fix_par (list): Boolean list indicating which parameters to fix.
        calc_grad (bool): Whether to calculate the gradient.
        calc_hess (bool): Whether to calculate the Hessian.

    Returns:
        float: Negative log likelihood.
        Optional: Gradient and Hessian as attributes of the output.
    """
    fix_pi0, fix_a, fix_mu = fix_par

    # Initialize parameters and update non-fixed values
    p = np.array(par_init)
    p[~np.array(fix_par)] = par

    # Parameters
    w = 1 - 1 / (1 + np.exp(p[0]))
    a = np.exp(p[1])
    mu = p[2]

    # Point mass component
    lf = -0.5 * np.log(2 * np.pi * s**2) - 0.5 * ((x - mu) / s)**2

    # Laplace component: left tail
    xleft = (x - mu) / s + s * a
    lpnormleft = norm.logsf(xleft)
    lgleft = np.log(a / 2) + s**2 * a**2 / 2 + a * (x - mu) + lpnormleft

    # Laplace component: right tail
    xright = (x - mu) / s - s * a
    lpnormright = norm.logcdf(xright)
    lgright = np.log(a / 2) + s**2 * a**2 / 2 - a * (x - mu) + lpnormright

    # Combine left and right tails
    lg = logscale_add(lgleft, lgright)

    # Log likelihood
    llik = logscale_add(np.log(1 - w) + lf, np.log(w) + lg)
    nllik = -np.sum(llik)

    # Gradients (optional)
    if calc_grad or calc_hess:
        grad = np.zeros(len(par))
        i = 0

        # Gradient with respect to w (alpha)
        if not fix_pi0:
            f = np.exp(lf - llik)
            g = np.exp(lg - llik)
            dnllik_dw = f - g
            dw_dalpha = w * (1 - w)
            grad[i] = np.sum(dnllik_dw * dw_dalpha)
            i += 1

        # Gradient with respect to a (beta)
        if not fix_a:
            dlogpnorm_left = -np.exp(-np.log(2 * np.pi) / 2 - xleft**2 / 2 - lpnormleft)
            dlogpnorm_right = np.exp(-np.log(2 * np.pi) / 2 - xright**2 / 2 - lpnormright)

            dgleft_da = np.exp(lgleft - llik) * (1 / a + a * s**2 + (x - mu) + s * dlogpnorm_left)
            dgright_da = np.exp(lgright - llik) * (1 / a + a * s**2 - (x - mu) - s * dlogpnorm_right)
            dg_da = dgleft_da + dgright_da

            dnllik_da = -w * dg_da
            da_dbeta = a
            grad[i] = np.sum(dnllik_da * da_dbeta)
            i += 1

        # Gradient with respect to mu
        if not fix_mu:
            df_dmu = np.exp(lf - llik) * ((x - mu) / s**2)
            dgleft_dmu = np.exp(lgleft - llik) * (-a - dlogpnorm_left / s)
            dgright_dmu = np.exp(lgright - llik) * (a - dlogpnorm_right / s)
            dg_dmu = dgleft_dmu + dgright_dmu
            dnllik_dmu = -(1 - w) * df_dmu - w * dg_dmu
            grad[i] = np.sum(dnllik_dmu)

        nllik_grad = grad

    # Hessian (optional)
    if calc_hess:
        hess = np.zeros((len(par), len(par)))
        # The second derivatives would go here, similar to the gradient logic
        # Implementation omitted for brevity

        nllik_hess = hess

    # Return results
    if calc_grad and calc_hess:
        return nllik, nllik_grad, nllik_hess
    elif calc_grad:
        return nllik, nllik_grad
    else:
        return nllik



from scipy.optimize import minimize

class optimizePointLalplace:
    def __init__(self, w, a, mu ,nllik):
        self.w = w
        self.a = a
        self.mu= mu
        self.nllik= nllik

def optimize_pl_nllik_with_gradient(x, s, par_init, fix_par):
    """
    Optimize the negative log likelihood for the point-Laplace prior using gradient.

    Args:
        x (np.ndarray): Observed data.
        s (np.ndarray): Standard deviations.
        par_init (list): Initial parameters [alpha, beta, mu].
        fix_par (list): Boolean list indicating which parameters are fixed.

    Returns:
        dict: Optimized parameters and final negative log likelihood.
    """
    # Define a wrapper for the objective function
    def objective(par):
        nllik, grad = pl_nllik(par, x, s, par_init, fix_par, calc_grad=True, calc_hess=False)
        return nllik, grad

    # Wrapper for gradient extraction
    def fun(par):
        nllik, _ = objective(par)
        return nllik

    def jac(par):
        _, grad = objective(par)
        return grad

    # Initial values for non-fixed parameters
    free_params = [p for p, fixed in zip(par_init, fix_par) if not fixed]

    # Bounds (optional): Keep alpha unbounded, beta > 0, mu unbounded
    bounds = [
        (None, None),  # Alpha has no bounds
        (None, None),  # Beta (transformed Laplace scale) has no explicit bounds
        (None, None)   # Mu has no bounds
    ]
    bounds = [b for b, fixed in zip(bounds, fix_par) if not fixed]

    # Optimize using L-BFGS-B
    result = minimize(
        fun, free_params, method="L-BFGS-B", jac=jac, bounds=bounds
    )

    
    # Update the full parameter set with optimized values
    optimized_params = np.array(par_init)
    optimized_params[~np.array(fix_par)] = result.x
    optimized_params[0]= 1- 1/(1+ np.exp(optimized_params[0]))
    nllik=result.fun
    
    if not result.success:
         
        #raise ValueError("Optimization failed: " + result.message)
        optimized_params[0]= 0
        optimized_params[1]=0
        nllik = 0

    # Compute final negative log likelihood
    

    return  optimizePointLalplace(w=optimized_params[0],
                                  a=np.exp( optimized_params[1]),
                                  mu=optimized_params[2],
                                  nllik= nllik)



class ebnm_point_laplace:
    def __init__(self, post_mean, post_mean2, post_sd, scale, pi,   log_lik=0,#log_lik2 =0,
                 mode=0):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.scale= scale
        self.pi =pi  
        self.log_lik = log_lik
       # self.log_lik2= log_lik2 
        self.mode =  mode


def ebnm_point_laplace_solver ( x,s,opt_mu=False,par_init = [0.5, 1.0, 0.0]  ):
    if(opt_mu):
        fix_par = [False, False, False]
    else :
        fix_par = [False, False, True]
    par_init = par_init
    optimized_prior =  optimize_pl_nllik_with_gradient(x, s, par_init, fix_par)
    post_obj=  posterior_mean_laplace(x, s, optimized_prior.w, optimized_prior.a, optimized_prior.mu)
    return( ebnm_point_laplace( post_mean=post_obj.post_mean, 
                                post_mean2=post_obj.post_mean2, 
                                post_sd=post_obj.post_sd,
                                scale=optimized_prior.a,
                                pi=optimized_prior.w,   log_lik=-optimized_prior.nllik,#log_lik2 =0,
                                mode=optimized_prior.mu)
    )