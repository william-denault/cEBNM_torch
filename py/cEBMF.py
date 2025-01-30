import numpy as np
import sys
from fancyimpute import IterativeSVD
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.sparse.linalg import svds
from scipy.stats import norm

# Add the path to utils.py 
sys.path.append(r"D:\Document\Serieux\Travail\python_work\cEBNM_torch\py")
sys.path.append(r"D:\Document\Serieux\Travail\python_work\cEBNM_torch\py\ebnm_solver")
sys.path.append(r"D:\Document\Serieux\Travail\python_work\cEBNM_torch\py\numerical_routine")
from ash import *
from empirical_mdn import *
from ebnm_point_laplace import *
from ebnm_point_exp import *
from covaraite_moderated_generalized_binary import *
from hard_covaraite_moderated_generalized_binary import *



class PriorResult:
    """
    A wrapper class to standardize the output of prior functions.
    Ensures compatibility with cEBMF framework.
    """
    def __init__(self, post_mean, post_mean2, log_lik=0, model_param=None):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.loss = -log_lik  # Make sure loss is always defined
        self.model_param = model_param  # Some priors may not use this

# Define prior functions that return a standardized PriorResult object
def prior_norm(X, betahat, sebetahat, model_param):
    ash_obj = ash(betahat=betahat, sebetahat=sebetahat, prior="norm", verbose=False)
    return PriorResult(post_mean=ash_obj.post_mean, post_mean2=ash_obj.post_mean2, log_lik=ash_obj.log_lik)

def prior_point_laplace(X, betahat, sebetahat, model_param):
    ebnm_obj = ebnm_point_laplace_solver(x=betahat, s=sebetahat)
    return PriorResult(post_mean=ebnm_obj.post_mean, post_mean2=ebnm_obj.post_mean2, log_lik=ebnm_obj.log_lik)

def prior_point_exp(X, betahat, sebetahat, model_param):
    ebnm_obj = ebnm_point_exp_solver(x=betahat, s=sebetahat)
    return PriorResult(post_mean=ebnm_obj.post_mean, post_mean2=ebnm_obj.post_mean2, log_lik=ebnm_obj.log_lik)

def prior_emdn(X, betahat, sebetahat, model_param):
    emdn = emdn_posterior_means(X=X, betahat=betahat, sebetahat=sebetahat, model_param=model_param)
    return PriorResult(post_mean=emdn.post_mean, post_mean2=emdn.post_mean2, log_lik=-emdn.loss, model_param=emdn.model_param)

def prior_cgb(X, betahat, sebetahat, model_param):
    cgb = cgb_posterior_means(X=X, betahat=betahat, sebetahat=sebetahat, model_param=model_param)
    return PriorResult(post_mean=cgb.post_mean, post_mean2=cgb.post_mean2, log_lik=-cgb.loss, model_param=cgb.model_param)

def prior_hard_cgb(X, betahat, sebetahat, model_param):
    cgb = cgb_hard_posterior_means(X=X, betahat=betahat, sebetahat=sebetahat, model_param=model_param)
    return PriorResult(post_mean=cgb.post_mean, post_mean2=cgb.post_mean2, log_lik=-cgb.loss, model_param=cgb.model_param)

# Dictionary mapping prior names to standardized functions
prior_functions = {
    "norm": prior_norm,
    "point_Laplace": prior_point_laplace,
    "point_exp": prior_point_exp,
    "emdn": prior_emdn,
    "cgb": prior_cgb,
    "hard_cgb": prior_hard_cgb
}

# Custom get function to ensure valid function selection
def get_prior_function(prior_dict, key, default):
    """
    Custom implementation of dictionary get() method.
    
    Parameters:
        prior_dict (dict): Dictionary of prior functions.
        key (str or callable): Key (string for predefined prior or a function).
        default (callable): Default value (used if key is not found).
    
    Returns:
        callable: The corresponding function if key exists, otherwise returns default.
    """
    if key in prior_dict:
        return prior_dict[key]  # Return function from dictionary
    elif callable(key):
        return key  # If key is already a function, return it directly
    else:
        raise ValueError(f"Invalid prior function: {key}. Must be a function or a valid prior name.")



class cEBMF_object:
    def __init__(self, data, K=5, X_l=None, X_f=None, max_K=100, prior_L="norm", prior_F="norm",
                 type_noise='constant', maxit=100, param_cebmf_l=None, param_cebmf_f=None, 
                 fit_constant=True, init_type="udv_si"):
        self.data = data.astype(np.float32)
        self.K = K
        self.X_l = X_l
        self.X_f = X_f
        self.max_K = max_K
        self.type_noise = type_noise
        self.maxit = maxit
        self.kl_l          =  np.zeros_like(range(self.K))
        self.kl_f          =  np.zeros_like(range(self.K))
        self.param_cebmf_l = param_cebmf_l
        self.param_cebmf_f = param_cebmf_f
        self.fit_constant = fit_constant
        self.init_type = init_type
        self.has_nan = np.isnan(self.data).any()
        self.n_nonmissing = np.prod(data.shape) - np.sum(np.isnan(data))
        self.obj = [np.inf]
        self.model_list_L = [None] * K
        self.model_list_F = [None] * K
        # Use custom get_prior_function()
        self.prior_L = get_prior_function(prior_functions, prior_L, prior_L)
        self.prior_F = get_prior_function(prior_functions, prior_F, prior_F)

    def init_LF(self, use_nmf=False):
        if self.has_nan:
            print("Initializing using Iterative SVD due to missing values.")
            imputed_data = IterativeSVD().fit_transform(self.data)
        else:
            imputed_data = self.data

        K = min(self.K, imputed_data.shape[1])

        if use_nmf:
            imputed_data[imputed_data < 0] = 1e-6
            nmf_model = NMF(n_components=K, init='random', random_state=42, max_iter=500)
            self.L = nmf_model.fit_transform(imputed_data).astype(np.float32)
            self.F = nmf_model.components_.T.astype(np.float32)
        else:
            U, s, Vt = svds(imputed_data, k=K)
            sorted_indices = np.argsort(s)[::-1]
            U, s, Vt = U[:, sorted_indices], s[sorted_indices], Vt[sorted_indices, :]
            self.L = (U[:, :K] @ np.diag(s[:K])).astype(np.float32)
            self.F = Vt[:K, :].T.astype(np.float32)

        self.L2 = np.abs(self.L) + 1e-4
        self.F2 = np.abs(self.F) + 1e-4
        self.update_tau()


    def cal_expected_residuals(self):
    # Initialize accumulators for the results
        prod_square_firstmom = np.zeros(self.data.shape, dtype=np.float32)
        prod_sectmom = np.zeros(self.data.shape, dtype=np.float32)

    # Loop through each component to accumulate results
        for k in range(self.K):
            L_k_squared = self.L[:, k] ** 2  # Precompute L[:, k] squared
            F_k_squared = self.F[:, k] ** 2  # Precompute F[:, k] squared
        
        # Use broadcasting instead of np.outer to reduce temporary memory usage
            prod_square_firstmom += L_k_squared[:, None] * F_k_squared[None, :]
            prod_sectmom += L_k_squared[:, None] * F_k_squared[None, :]

    # Update fitted values
        self.update_fitted_val()

    # Compute residuals
        R2 = (self.data - self.Y_fit) ** 2 - prod_square_firstmom + prod_sectmom
        return R2

    def update_fitted_val(self):
        self.Y_fit = np.zeros(self.data.shape, dtype=np.float32)
        for k in range(self.K):
            self.Y_fit += np.outer(self.L[:, k], self.F[:, k])

    def cal_partial_residuals(self, k):
        """Update Y_fit based on current factors."""
    # Precompute the entire reconstruction
        full_reconstruction = self.L @ self.F.T  # Matrix multiplication

    # Subtract the contribution of the k-th factor
        k_contribution = np.outer(self.L[:, k], self.F[:, k])
        self.Rk = self.data - (full_reconstruction - k_contribution)


    def update_tau(self):
        R2 = self.cal_expected_residuals()
        if self.type_noise == 'constant':
            self.tau = np.full(self.data.shape, 1 / np.nanmean(R2), dtype=np.float32)
        elif self.type_noise == 'row_wise':
            row_means = np.nanmean(R2, axis=1).astype(np.float32)
            self.tau = np.tile(1 / row_means, (self.data.shape[1], 1)).T
        elif self.type_noise == 'column_wise':
            col_means = np.nanmean(R2, axis=0).astype(np.float32)
            self.tau = np.tile(1 / col_means, (self.data.shape[0], 1))

    

    def update_loading_factor_k(self, k):
        """Update factor loading for component k dynamically using prior functions."""
        self.cal_partial_residuals(k=k)

        # Compute estimates for L
        lhat, s_l = compute_hat_l_and_s_l(self.Rk, self.F[:, k], self.F2[:, k], self.tau, self.has_nan)

        # Use user-defined or predefined function for L
        prior_result_L = self.prior_L(self.X_l, lhat, s_l, self.model_list_L[k])
        self.model_list_L[k] = prior_result_L.model_param
        self.L[:, k] = prior_result_L.post_mean
        self.L2[:, k] = prior_result_L.post_mean2
        self.kl_l[k] = prior_result_L.loss  # Loss is already negated in PriorResult

        # Compute estimates for F
        fhat, s_f = compute_hat_f_and_s_f(self.Rk, self.L[:, k], self.L2[:, k], self.tau, self.has_nan)

        # Use user-defined or predefined function for F
        prior_result_F = self.prior_F(self.X_f, fhat, s_f, self.model_list_F[k])
        self.model_list_F[k] = prior_result_F.model_param
        self.F[:, k] = prior_result_F.post_mean
        self.F2[:, k] = prior_result_F.post_mean2
        self.kl_f[k] = prior_result_F.loss  # Loss is already negated in PriorResult



    def iter(self):
        """Run one iteration of factor updates."""
        for k in range(self.K):
            self.update_loading_factor_k(k)
        self.update_tau()
        self.cal_obj()

        
        
        
    def cal_obj(self):
        KL = sum(self.kl_f) + sum(self.kl_l)

    # Determine tau based on the type_noise setting
        if self.type_noise == 'constant': 
            tau = self.tau[0, 0]
            n_tau = 1
        elif self.type_noise == 'row_wise':
            tau = self.tau[:, 0]
            n_tau = self.data.shape[1]
        elif self.type_noise == 'column_wise':
            tau = self.tau[0, :]
            n_tau = self.data.shape[0]
        else:
            raise ValueError(f"Invalid type_noise value: {self.type_noise}")

    # Compute the objective function
        obj = KL - 0.5 * np.sum(self.n_nonmissing * (np.log(2 * np.pi) - np.log(tau + 1e-8) + n_tau))
        self.obj.append(obj)


    
     
def cEBMF(
    data, 
    K=5,
    X_l=None, 
    X_f=None, 
    max_K=100, 
    prior_L="norm",
    prior_F="norm",
    type_noise='constant',
    maxit=100,
    param_cebmf_l=None,
    param_cebmf_f=None,
    fit_constant=True,
    init_type="udv_si"
):
    """
Covariate Moderated Empirical Bayes Matrix Factorization

Parameters
----------
data : numpy.ndarray
    A numerical matrix of size (N, P) representing the data to be factorized.

X_l : numpy.ndarray
    A matrix of size (N, J) containing covariates that affect the factors in the rows of Y.

X_f : numpy.ndarray
    A matrix of size (P, T) containing covariates that affect the factors in the columns of Y.

mnreg_type : str, optional
    Specifies the underlying learner for modeling heterogeneity within each topic. Two methods are available: 
    - 'nnet': Uses a neural network model (default).
    - 'keras': Uses a Keras-based neural network.
    You can pass additional specifications for the 'nnet' model through the `param_nnet` argument and for the 
    logistic SuSiE model through the `param_susie` argument.

K : int
    The number of factors to start with.

type_noise : str, optional
    Specifies the noise structure in the data. The following options are available:
    - 'column_wise': Assumes noise is constant across columns.
    - 'constant': Assumes noise is constant throughout the entire matrix.
    - 'row_wise': Assumes noise is constant across rows.

init_type : str, optional
    Specifies the initialization method for the factorization. Available methods are:
    - 'udv_si': Default method.
    - 'random': Random initialization.
    - 'flashier': Flashier initialization method.
    - 'flashier_NMF': Non-negative Matrix Factorization initialization using Flashier.
    - 'flashier_SNM': Sparse Non-negative Matrix initialization using Flashier.
    - 'udv_si_svd': Initialization using Singular Value Decomposition.
    - 'udv': UDV decomposition initialization.
    - 'rand_udv': Random UDV decomposition initialization.

maxit : int
    The maximum number of iterations for the factorization algorithm.

tol : float
    The tolerance value for assessing convergence. The algorithm will stop if the improvement is below this threshold.

 

"""
    
    return cEBMF_object( data=data, 
                 K=K,
                 X_l=X_l, 
                 X_f=X_f, 
                 max_K=max_K, 
                 prior_L= prior_L,
                 prior_F= prior_F,
                 type_noise=type_noise ,
                 maxit= maxit ,
                 param_cebmf_l =param_cebmf_l,
                 param_cebmf_f =param_cebmf_f,
                 fit_constant= fit_constant,
                 init_type =init_type)







def compute_hat_l_and_s_l(Z, nu, omega, tau, has_nan=False):
   
    if has_nan == False:
        # Compute the numerator, broadcasting nu properly across columns (broadcast nu over axis=0)
        numerator_l_hat = np.sum(tau * Z * nu, axis=1)
        denominator_l_hat = np.sum(tau * omega, axis=1) + 1e-8
    else:
        # Fill nan values in Z with 0
        Z_filled = np.nan_to_num(Z, nan=0)
        # Set tau to 0 wherever Z has nan values
        mask_nan_Z = np.isnan(Z)
        tau_spiked = np.copy(tau)  # Make a copy of tau to modify
        tau_spiked[mask_nan_Z] = 0

        # Compute the numerator and denominator using the modified tau and Z
        numerator_l_hat = np.sum(tau_spiked * Z_filled * nu, axis=1)
        denominator_l_hat = np.sum(tau_spiked * omega, axis=1)

    # Compute l_hat
    l_hat = numerator_l_hat / (denominator_l_hat+1e-6)

    # Compute s_l
    
    s_l = (denominator_l_hat) ** (-0.5)+1e-6
    if  denominator_l_hat.any()==0:
         idx=  np.where((denominator_l_hat == 0))[0]
         s_l[idx]= 10*np.abs(denominator_l_hat[idx]) +1e-8
    if np.isinf(s_l).any():
         idx=  np.where(np.isinf(s_l))[0]
         s_l[idx]= 10*np.abs(l_hat[idx]) +1e-8
 
    return l_hat, s_l

def compute_hat_f_and_s_f(Z, nu, omega, tau, has_nan=False):
    # Create a mask that identifies non-nan entries in Z
    mask = ~np.isnan(Z)  # Assuming the mask is related to Z having no NaNs

    if not has_nan:
        # Compute the numerator, broadcasting nu properly across columns (broadcast nu over axis=0)
        numerator_f_hat = np.sum(tau * Z * mask * nu[:, np.newaxis], axis=0)  # Sum over i (axis=0)
        denominator_f_hat = np.sum(tau * mask * omega[:, np.newaxis], axis=0) +1e-6 # Sum over i (axis=0)
    else:
        # Fill nan values in Z with 0
        Z_filled = np.nan_to_num(Z, nan=0)
        # Set tau to 0 wherever Z has nan values
        mask_nan_Z = np.isnan(Z)
        tau_spiked = np.copy(tau)  # Make a copy of tau to modify
        tau_spiked[mask_nan_Z] = 0

        # Compute the numerator and denominator using the modified tau and Z
        numerator_f_hat = np.sum(tau_spiked * Z_filled * mask * nu[:, np.newaxis], axis=0)  # Sum over i (axis=0)
        denominator_f_hat = np.sum(tau_spiked * mask * omega[:, np.newaxis], axis=0)  # Sum over i (axis=0)

    # Compute f_hat
    f_hat = numerator_f_hat /( denominator_f_hat+1e-6)

    # Compute s_f
    s_f = (denominator_f_hat) ** (-0.5)+1e-6    
    if  denominator_f_hat.any()==0:
        idx=  np.where((denominator_f_hat == 0))[0]
        s_f[idx]= 10*np.abs(f_hat[idx]) +1e-8
    if np.isinf(s_f).any():
         idx=  np.where(np.isinf(s_f))[0]
         s_f[idx]= 10*np.abs(f_hat[idx]) +1e-8

    return f_hat, s_f

import numpy as np

def normal_means_loglik(x, s, Et, Et2):
    # Create a boolean index for valid entries where s is finite and greater than 0
    idx = np.isfinite(s) & (s > 0)
    
    # Filter arrays based on valid index
    x = x[idx]
    s = s[idx]
    Et = Et[idx]
    Et2 = Et2[idx]
    
    # Calculate and return the log-likelihood
    return -0.5 * np.sum(np.log(2 * np.pi * s**2) + (1 / s**2) * (Et2 - 2 * x * Et + x**2))


