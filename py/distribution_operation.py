import numpy as np
import math 
from scipy.stats import norm
from scipy.stats import truncnorm
import scipy.stats as stats
#compute gaussian likelihood

def convolved_logpdf_normal(  betahat,sebetahat, location, scale):
  # Calculate the standard deviation
    sd = np.sqrt(sebetahat**2 +  scale**2)
    
    # Calculate the log probability density
    logp = norm.logpdf(betahat, loc= location, scale=sd)
    
    # Clamp the log probabilities to the range [-1e4, 1e4]
    logp = np.clip(logp, -1e4, 1e4)
    
    return logp
def get_data_loglik_normal (  betahat,sebetahat, location, scale):
    out = np.zeros( (betahat.shape[0], scale.shape[0]))
    for i in range(betahat.shape[0]):
        out[i,] =convolved_logpdf_normal(betahat=betahat[i],
                                         sebetahat=sebetahat[i], 
                                         location=location,
                                         scale=scale)
    return out