import sys
import os
import numpy as np
import math 
from scipy.stats import norm
from scipy.stats import truncnorm
import scipy.stats as stats


# Add the path to utils.py
sys.path.append(r"c:\Document\Serieux\Travail\python_work\cEBNM_torch\py")
from distribution_operation import *
from numerical_routine import *

class PosteriorMeanExp:
    def __init__(self, post_mean, post_mean2, post_sd):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd

def posterior_mean_exp(betahat, sebetahat, log_pi, scale):
    assignment = np.exp(log_pi)
    assignment = assignment / assignment.sum(axis=1, keepdims=True)
    mu = 0
    post_assign = np.zeros((betahat.shape[0], scale.shape[0]))
    
    for i in range(betahat.shape[0]):
        post_assign[i,] = wpost_exp(x=betahat[i],
                                    s=sebetahat[i], 
                                    w=assignment[i,],
                                    scale=scale) 
    
    post_mean = np.zeros(betahat.shape[0])
    post_mean2 = np.zeros(betahat.shape[0])

    for i in range(post_mean.shape[0]):
        post_mean[i] = sum(post_assign[i, 1:] * my_etruncnorm(0,
                                                              np.inf,
                                                              betahat[i] - sebetahat[i]**2 * (1/scale[1:]), 
                                                              sebetahat[i]))
        post_mean2[i] = sum(post_assign[i, 1:] * my_e2truncnorm(0,
                                                                99999, #some weird warning for inf so just use something large enough for b
                                                                betahat[i] - sebetahat[i]**2 * (1/scale[1:]), 
                                                                sebetahat[i]))
        post_mean2[i] = max(post_mean[i], post_mean2[i])
    
    if np.any(np.isinf(sebetahat)):
        inf_indices = np.isinf(sebetahat)
        a = 1/scale[1:]
        # Equivalent of `post$mean[is.infinite(s)]` 
        post_mean[inf_indices] = np.sum(post_assign[inf_indices, 1:] / a, axis=1)

        # Equivalent of `post$mean2[is.infinite(s)]`
        post_mean2[inf_indices] = np.sum(2 * post_assign[inf_indices, 1:] / a**2, axis=1)

    # Calculate `post_sd`
    post_sd = np.sqrt(np.maximum(0, post_mean2 - post_mean**2))

    # Update `post_mean2` and `post_mean`
    post_mean2 = post_mean2 + mu**2 + 2 * mu * post_mean
    post_mean = post_mean + mu

    # Return the results as an instance of PosteriorMeanExpResult
    return PosteriorMeanExp(post_mean, post_mean2, post_sd)

     
   
def wpost_exp ( x, s, w, scale):
     
    if  w[0]==1:
     out =  np.concatenate(([1]  ,np.full( scale.shape[0],[0])))
     return out
    else:
     a=1/scale[1:] 
     a = 1 / scale[1:]  # Note: slicing in Python is zero-based, so [1:] starts from the second element
     lf = norm.logpdf(x, loc=0, scale=s)
     lg = np.log(a) + s**2 * a**2 / 2 - a * x + norm.logcdf(x / s - s * a)
     log_prob = np.concatenate(([lf]  ,lg ))
     bmax=np.max(log_prob)
     log_prob = log_prob - bmax 
     wpost = w* np.exp( log_prob) / (sum(w *np.exp(log_prob)))
     return wpost    
 
 
 
class PosteriorMeanNorm:
    def __init__(self, post_mean, post_mean2, post_sd):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        
        
        
        
def posterior_mean_norm(betahat, sebetahat, log_pi, scale, location=None):
     #location ois wether a 1 d np with same length as scale array or n by length scale np array 
    if location is None:
        location = 0* scale
        
        
   
    data_loglik = get_data_loglik_normal (  betahat,sebetahat, location, scale)
    log_post_assignment = apply_log_sum_exp(data_loglik, log_pi)
    t_ind_Var =np.zeros((betahat.shape[0], scale.shape[0]))
 
   
    if(scale[0]==0):
        for i in range(t_ind_Var.shape[0]):
            t_ind_Var[i , ]= np.concatenate(
                                            ([0], 
                                             1/((1/sebetahat[i]**2)+ (1/scale[1:]**2)) )
                                            )#assume that first entry of scale is 0
    else:
        for i in range(t_ind_Var.shape[0]):
            t_ind_Var[i , ]= 1/((1/sebetahat[i]**2)+ (1/scale **2)) 
           
    
        
    temp=np.zeros((betahat.shape[0], scale.shape[0]))
    if len(location.shape)==1:
        for i in range(temp.shape[0]):
            temp[i,] = (t_ind_Var[i,]/(sebetahat[i]**2))*(betahat[i] )+ location*(1-t_ind_Var[i,]/(sebetahat[i]**2))
    elif len(location.shape)==2: 
        for i in range(temp.shape[0]):
            temp[i,] = (t_ind_Var[i,]/(sebetahat[i]**2))*(betahat[i] )+ location[i,]*(1-t_ind_Var[i,]/(sebetahat[i]**2))
    

    post_mean  = np.sum ( np.exp(log_post_assignment)* temp, axis=1)
    post_mean2 = np.sum ( np.exp(log_post_assignment)*( t_ind_Var+ temp**2), axis=1)
    post_sd    = np.sqrt(post_mean2-post_mean**2)
    return PosteriorMeanNorm(post_mean, post_mean2, post_sd)

def apply_log_sum_exp(data_loglik, assignment_loglik):
    combined_loglik = data_loglik + assignment_loglik
    
    def subtract_log_sum_exp(row):
        return row - log_sum_exp(row)
    
    # Apply the function row-wise and stack the results
    res = np.apply_along_axis(subtract_log_sum_exp, 1, combined_loglik)
    
    return res





 
import numpy as np
from scipy.stats import norm

def posterior_point_mass_normal(betahat, sebetahat, pi, mu0, mu1, sigma_0):
    """
    Compute the posterior mean and variance for a normal likelihood with point-mass-normal prior.

    Parameters:
    - betahat: observed data point
    - sebetahat : observation noise sd
    - pi: mixing proportion for the point mass (between 0 and 1)
    - mu0: point mass location
    - mu1: mean of the normal prior
    - sigma_0 : sd of the normal prior

    Returns:
    - post_mean: posterior mean
    - post_var: posterior variance
    """

    # Avoid numerical errors when sigma_0 or sebetahat are too small
    sigma_0 = max(sigma_0, 1e-8)
    sebetahat = np.maximum(sebetahat, 1e-8)  # Ensure it's not too small

    # Compute marginal likelihood for the normal prior
    marginal_likelihood = norm.pdf(betahat, loc=mu1, scale=np.sqrt(sigma_0**2 + sebetahat**2))
    
    # Compute likelihood under the point mass
    likelihood_point_mass = norm.pdf(betahat, loc=mu0, scale=sebetahat)

    # Compute denominator safely
    denominator = pi * likelihood_point_mass + (1 - pi) * marginal_likelihood
    denominator = np.maximum(denominator, 1e-12)  # Prevent division by zero

    # Compute posterior weights
    w0 = np.clip(pi * likelihood_point_mass / denominator, 0, 1)
    w1 = 1 - w0

    # Compute posterior mean and variance for the normal component
    mu_post = (mu1 / sigma_0**2 + betahat / sebetahat**2) / (1 / sigma_0**2 + 1 / sebetahat**2)
    sigma_post2 = 1 / (1 / sigma_0**2 + 1 / sebetahat**2)

    # Compute final posterior mean
    post_mean = w0 * mu0 + w1 * mu_post

    # Compute final posterior variance
    post_var = w0 * (mu0 - post_mean) ** 2 + w1 * (sigma_post2 + (mu_post - post_mean) ** 2)

    return post_mean, post_var
