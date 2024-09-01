def posterior_mean_exp(betahat, sebethat, log_pi, scale):
     assignment  <- np.exp(log_pi)
     assignment <- assignment /  sum(assignment)
     
     temp_array =  np.zeros ( (betahat.shape[0], scale.shape[0]))   
     for i in range(betahat.shape[0]):
          temp_array[i,] = wpost_exp ( x=betahat[i], 
                                      s=sebetahat[i],
                                      w=wassignment[i,],
                                      scale=scale)   
     
     TBEdone
     
def wpost_exp ( x, s, w, scale):
    
    if  w[0]==1:
     out =  np.concatenate(([1]  ,np.full( scale.shape[0],[0])))
     return out
    else:
     a=1/scale[1:]
     w = assignment
     a = 1 / scale[1:]  # Note: slicing in Python is zero-based, so [1:] starts from the second element
     lf = norm.logpdf(x, loc=0, scale=s)
     lg = np.log(a) + s**2 * a**2 / 2 - a * x + norm.logcdf(x / s - s * a)
     log_prob = np.concatenate(([lf]  ,lg ))
     bmax=np.max(log_prob)
     log_prob = log_prob - bmax
 
     log_prob = log_prob - bmax
     wpost = w* np.exp( log_prob) / (sum(w *np.exp(log_prob)))
     return wpost


def compute_posterior_assignment_mix_exp(betahat, sebetahat, log_pi, scale ):
     