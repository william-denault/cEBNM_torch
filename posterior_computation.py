def posterior_mean_exp(betahat, sebethat, log_pi, scale):
     assignment  <- np.exp(log_pi)
     assignment <- assignment /  sum(assignment)
     
     
     
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
  wpost <- w* exp( log_prob) / (sum(w *exp(log_prob)))


    