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
     
    if location is None:
        location = 0* scale
        
        
   
    data_loglik = get_data_loglik_normal (  betahat,sebetahat, location, scale)
    log_post_assignment = apply_log_sum_exp(data_loglik, log_pi)
    t_ind_Var =np.zeros((betahat.shape[0], scale.shape[0]))
 
    for i in range(t_ind_Var.shape[0]):
        t_ind_Var[i , ]= np.concatenate(
                                        ([0], 
                                         1/((1/sebetahat[i]**2)+ (1/scale[1:]**2)) )
                                        )#assume that first entry of scale is 0
        
    temp=np.zeros((betahat.shape[0], scale.shape[0]))
 
    for i in range(temp.shape[0]):
            temp[i,] = (t_ind_Var[i,]/(sebetahat[i]**2))*(betahat[i] )+ location*(1-t_ind_Var[i,]/(sebetahat[i]**2))

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