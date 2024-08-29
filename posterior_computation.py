def posterior_mean_exp(betahat, sebethat, logi_pi, scale):
     assignment  <- np.exp(logi_pi)
     assignment <- assignment /  sum(assignment)
     
     
     
def wpost_exp ( betahat, sebethat, assignmenti, scale):
    
    if  w[0]==1
    out = np.array( np.concatenate([1]  ,np.full( scale.shape[0])))
     return out
  # assuming a[1 ]=0
   
  a <- 1/ scale[-1]

  lf <- dnorm(x, 0, s, log = TRUE)
  lg <- log(a) + s^2 * a^2 / 2 - a * x + pnorm(x / s - s * a, log.p = TRUE)



  log_prob = c(lf, lg)
  bmax = max(log_prob)
  log_prob = log_prob - bmax
  wpost <- w* exp( log_prob) / (sum(w *exp(log_prob)))


    