import numpy as np
import math 
from scipy.stats import norm
from scipy.stats import truncnorm
import scipy.stats as stats
#compute gaussian likelihood

def convolved_logpdf_normal(  betahat,sebetahat, location, scale):
  # Calculate the standard deviation
  #location ois wether a 1 d np with same length as scale 
    sd = np.sqrt(sebetahat**2 +  scale**2)
    
    # Calculate the log probability density
    logp = norm.logpdf(betahat, loc= location, scale=sd)
    
    # Clamp the log probabilities to the range [-1e4, 1e4]
    logp = np.clip(logp, -1e5, 1e5)
    
    return logp
def get_data_loglik_normal (  betahat,sebetahat, location, scale):
  
  
  #location is wether a 1 d np with same length as scale array or n by length scale np array 
    out = np.zeros( (betahat.shape[0], scale.shape[0]))
    if len(location.shape)==1:
      for i in range(betahat.shape[0]):
        out[i,] =convolved_logpdf_normal(betahat=betahat[i],
                                         sebetahat=sebetahat[i], 
                                         location=location,
                                         scale=scale)
    elif len(location.shape)==2:
      for i in range(betahat.shape[0]):
        out[i,] =convolved_logpdf_normal(betahat=betahat[i],
                                         sebetahat=sebetahat[i], 
                                         location=location[i,:],
                                         scale=scale)
    
    return out
  
  
def convolved_logpdf_exp(  betahat, sebetahat, scale):
    """
    Python equivalent of the R function convolved_logpdf.exp
    
    Parameters:
     
    betahat (float): The beta hat value
    sebetahat (float): The standard error
    
    Returns:
    float: The computed log probability density
    """ 
    rate = 1 / scale[1:]
       
    out_0 = np.array([ norm.logpdf(betahat, 0, sebetahat)])
    
    out_1 = (np.log(rate) + 0.5 * sebetahat**2 * rate**2 - betahat * rate + 
            norm.logcdf(betahat/sebetahat - sebetahat * rate))
    return (  np.concatenate((out_0,out_1) )  )  
        
def get_data_loglik_exp (  betahat,sebetahat,scale):
    out = np.zeros( (betahat.shape[0], scale.shape[0]))
    for i in range(betahat.shape[0]):
        out[i,] =convolved_logpdf_exp(betahat=betahat[i],
                                         sebetahat=sebetahat[i],  
                                         scale=scale)
    return out