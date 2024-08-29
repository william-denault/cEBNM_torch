def autoselect_scales_mix_norm(betahat, sebetahat, max_class, mult = 2):
    sigmamin = np.min(sebetahat  )/10
    if(np.all(betahat**2 <sigmamin)):
        sigmaamax = 8 * sigmaamin
    else:
         sigmaamax = np.sqrt(np.max(betahat**2 - sebetahat**2))
    
    if(mult==0):
    out = np.array ([0,sigmaamax / 2 ])
    else 
    
    

    