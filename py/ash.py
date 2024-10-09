import numpy as np
import sys
import os
sys.path.append(r"c:\Document\Serieux\Travail\python_work\cEBNM_torch\py")
from distribution_operation import *
from utils import *
from numerical_routine import *
from posterior_computation import *

class ash_object:
    def __init__(self, post_mean, post_mean2, post_sd, scale, pi, prior, log_lik=0,log_lik2 =0):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.scale= scale
        self.pi =pi 
        self.prior= prior
        self.log_lik = log_lik
        self.log_lik2= log_lik2 


def ash ( betahat,sebetahat, prior = "norm", mult=np.sqrt(2),penalty=10,verbose= True,threshold_loglikelihood =  -300):
    
     
    if prior== "norm":
       
        scale=autoselect_scales_mix_norm(betahat  = betahat,
                                         sebetahat= sebetahat,
                                         mult=mult)
        L= get_data_loglik_normal(betahat=betahat ,
                                 sebetahat=sebetahat ,
                                 location=0*scale,
                                 scale=scale)
        optimal_pi = optimize_pi( np.exp(L),
                                 penalty=penalty,
                                 verbose=verbose) 
        out= posterior_mean_norm(betahat, sebetahat,
                                 log_pi=np.log(optimal_pi+1e-32), 
                                 scale=scale)
    if prior== "exp":
        scale=autoselect_scales_mix_exp(betahat  = betahat,
                                         sebetahat= sebetahat,
                                          mult=mult)
        L= get_data_loglik_exp(betahat=betahat ,
                                 sebetahat=sebetahat , 
                                 scale=scale)
        optimal_pi = optimize_pi( np.exp(L),
                                 penalty=penalty,
                                 verbose=verbose)  
        log_pi=  np.tile(np.log(optimal_pi+1e-32), (betahat.shape[0],1))
        
        out= posterior_mean_exp(betahat, sebetahat,
                                 log_pi=log_pi, 
                                 scale=scale)
     
    L = np.maximum(L, threshold_loglikelihood)
    
    log_lik =    np.sum(np.log(np.sum(np.exp(L)*optimal_pi, axis=1)))
    
    L_max = np.max(L, axis=1, keepdims=True)
    exp_term = np.exp(L - L_max)
    exp_term = np.maximum(exp_term, 1e-300)  # Add a small threshold to prevent extremely small values
    log_sum_exp = L_max + np.log(np.sum(exp_term, axis=1))
    log_lik2 = np.sum(log_sum_exp)

    
    
    return ash_object(post_mean  = out.post_mean,
                      post_mean2 = out.post_mean2,
                      post_sd    = out.post_sd,
                      scale      = scale,
                      pi         = optimal_pi,
                      prior      = prior ,
                      log_lik    = log_lik,
                      log_lik2   = log_lik2 )
