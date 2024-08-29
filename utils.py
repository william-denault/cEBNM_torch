def autoselect_scales_mix_norm(betahat, sebetahat, max_class=10, mult=2):
    sigmaamin = np.min(sebetahat) / 10
    if np.all(betahat**2 < sigmaamin**2):  # Fix the typo and ensure logical comparison
        sigmaamax = 8 * sigmaamin
    else:
        sigmaamax = np.sqrt(np.max(betahat**2 - sebetahat**2))
    
    if mult == 0:
        out = np.array([0, sigmaamax / 2])
    else:
        npoint = math.ceil(math.log2(sigmaamax / sigmaamin) / math.log2(mult))

        # Generate the sequence (-npoint):0 using np.arange
        sequence = np.arange(-npoint, 1)

        # Calculate the output
        out = np.concatenate(([0], (1/mult) ** (-sequence) * sigmaamax))
        
        # Check if the length of out is equal to max_class
        if len(out) != max_class:
            # Generate a sequence from min(out) to max(out) with length max_class
            out = np.linspace(np.min(out), np.max(out), num=max_class)
    
    return out
     

    
    

    