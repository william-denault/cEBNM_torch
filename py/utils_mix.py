import numpy as np
import math 
from sklearn.preprocessing import StandardScaler


def autoselect_scales_mix_norm(betahat, sebetahat, max_class=None, mult=2):
    sigmaamin = np.min(sebetahat) / 10
    if np.all(betahat**2 < sebetahat**2):  # Fix the typo and ensure logical comparison
        sigmaamax = 8 * sigmaamin
    else:
        sigmaamax = 2*np.sqrt(np.max(betahat**2 - sebetahat**2))
    
    if mult == 0:
        out = np.array([0, sigmaamax / 2])
    else:
        npoint = math.ceil(math.log2(sigmaamax / sigmaamin) / math.log2(mult))

        # Generate the sequence (-npoint):0 using np.arange
        sequence = np.arange(-npoint, 1)

        # Calculate the output
        out = np.concatenate(([0], (1/mult) ** (-sequence) * sigmaamax))
        if max_class!=None:
            # Check if the length of out is equal to max_class
            if len(out) != max_class:
            # Generate a sequence from min(out) to max(out) with length max_class
                out = np.linspace(np.min(out), np.max(out), num=max_class)
        
    
    return out
     
def autoselect_scales_mix_exp(betahat, sebetahat, max_class=None , mult=1.5,tt=1.5):
    sigmaamin = np.max( [np.min(sebetahat) / 10 , 1e-3])
    if np.all(betahat**2 < sebetahat**2):  # Fix the typo and ensure logical comparison
        sigmaamax = 8 * sigmaamin
    else:
        sigmaamax = tt*np.sqrt(np.max(betahat**2  ))
    
    if mult == 0:
        out = np.array([0, sigmaamax / 2])
    else:
        npoint = math.ceil(math.log2(sigmaamax / sigmaamin) / math.log2(mult))

        # Generate the sequence (-npoint):0 using np.arange
        sequence = np.arange(-npoint, 1)

        # Calculate the output
        out = np.concatenate(([0], (1/mult) ** (-sequence) * sigmaamax))
        if max_class!=None:
            # Check if the length of out is equal to max_class
            if len(out) != max_class:
            # Generate a sequence from min(out) to max(out) with length max_class
                out = np.linspace(np.min(out), np.max(out), num=max_class)
                if(out[2] <1e-2 ):
                 out[2: ] <- out[2: ] +1e-2
         
    
    return out

    
    
def col_scale(X, with_mean=True, with_std=True):
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler.scale_