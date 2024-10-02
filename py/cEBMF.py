import numpy as np
import sys 
from fancyimpute import IterativeSVD
import matplotlib.pyplot as plt
# Add the path to utils.py
sys.path.append(r"c:\Document\Serieux\Travail\python_work\cEBNM_torch\py")
from ash import *


class cEBMF_object :
    def __init__( 
                 self,
                 data, 
                 K=5,
                 X_l=None, 
                 X_f=None, 
                 max_K=100, 
                 prior_L= "norm",
                 prior_F= "norm",
                 type_noise='constant' ,
                 maxit= 100 ,
                 param_cebmf_l =None,
                 param_cebmf_f =None,
                 fit_constant= True,
                 init_type ="udv_si"):
     self.data=data 
     self.K = K 
     self.X_l =X_l
     self.X_f = X_f
     self.max_K =max_K
     self.prior_L= prior_L
     self.prior_F = prior_F
     self.type_noise = type_noise,
     self.maxit= maxit
     self.param_cebmf_l = param_cebmf_l
     self.param_cebmf_f = param_cebmf_f
     self.fit_constant =fit_constant
     self.init_type = init_type
     self.has_nan   =np.isnan(self.data).any()


    def init_LF(self):
        
         

        if self.has_nan:
         print("The array contains missing values (NaN), generate initialization using iterive svd.")
         imputed_data = IterativeSVD().fit_transform(self.data)
          
         #imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
         #imputed_data = imputer.fit_transform(self.data)
         
         U, s, Vt = np.linalg.svd(imputed_data, full_matrices=False)
        else:
         print("The array does not contain any missing values.")
         U, s, Vt = np.linalg.svd(self.data, full_matrices=False)
        
       
        K = np.min([self.K, U.shape[1]])
        # Keep only the top K singular values/vectors
        U_k = U[:, :K]
        D_k = np.diag(s[:K])
        V_k = Vt[:K, :]
        self.L = np.matmul(U_k,  D_k)
        self.F =V_k.T
        self.L2 = self.L**2
        self.F2 = self.F**2
        self.update_tau()
    
    def cal_expected_residuals(self): 
        if self.K == 1:
            # When K=1, directly calculate outer product for first and second moments
            prod_square_firstmom = np.outer(self.L[:, 0] ** 2, self.F[:, 0] ** 2)
            prod_sectmom = np.outer(self.L2[:, 0], self.F2[:, 0])
        else:
            # When K>1, sum the outer products across all K components
            prod_square_firstmom = np.sum(
                [np.outer(self.L[:, k] ** 2, self.F[:, k].T ** 2) for k in range(self.K)], axis=0
            )
            prod_sectmom = np.sum(
                [np.outer(self.L2[:, k], self.F2[:, k]) for k in range(self.K)], axis=0
            )

        self.update_fitted_val()  # Update fitted values Y_fit

        # Compute R2 as per the formula
        R2 = (self.data- self.Y_fit) ** 2 - prod_square_firstmom + prod_sectmom
        return R2

    def update_fitted_val(self):
        """Update Y_fit based on current factors."""
        self.Y_fit= np.sum( [np.outer(   self.L[:, k]  ,  self.F[:, k]    ) for k in range( self.K)], axis=0)

    def cal_partial_residuals(self,k):
        """Update Y_fit based on current factors."""
        idx_loop = set(range(self.K))-{k}
        self.Rk= self.data - np.sum( [np.outer(   self.L[:, j]  ,  self.F[:, j]    ) for j in  idx_loop], axis=0)

    def update_tau(self):
        """Update tau based on the noise structure."""
        R2 = self.cal_expected_residuals()
        if self.type_noise[0] == 'constant': 
            self.tau = np.full(self.data.shape, 1 / np.nanmean(R2))
        elif self.type_noise[0] == 'row_wise':
            row_means = np.nanmean(R2, axis=1)  # Mean across rows
            self.tau = np.tile(1 / row_means, (self.data.shape[1], 1)).T
        elif self.type_noise[0] == 'column_wise':
            col_means = np.nanmean(R2, axis=0)  # Mean across columns
            self.tau = np.tile(1 / col_means, (self.data.shape[0], 1))
            
            
            
    def update_loading_factor_k(self, k):
        self.cal_partial_residuals(k=k)
        lhat , s_l  = compute_hat_l_and_s_l(Z = self.Rk,
                                                            nu = self.F[:,k] ,
                                                            omega= self.F2[:,k], 
                                                            tau= self.tau,
                                                            has_nan=self.has_nan)
     
        ash_obj = ash(betahat   =lhat,
                      sebetahat =s_l ,
                      prior     = self.prior_L,
                      verbose=False
                      )
        self.L  [:,k] =ash_obj.post_mean
        self.L2 [:,k] =ash_obj.post_mean2
        
        fhat , s_f  = compute_hat_f_and_s_f(Z = self.Rk,
                                                            nu = self.L[:,k] ,
                                                            omega= self.L2[:,k], 
                                                            tau= self.tau  ,
                                                            has_nan=self.has_nan)
        ash_obj = ash(betahat   = fhat, 
                      sebetahat = s_f ,
                      prior     = self.prior_F,
                      verbose=False
                      )
        self.F  [:,k] =ash_obj.post_mean
        self.F2 [:,k] =ash_obj.post_mean2
    
    def iter (self):
        for k in range(self.K):
            self.update_loading_factor_k(k=k)
        self.update_tau()
    

def cEBMF( data, 
                 K=5,
                 X_l=None, 
                 X_f=None, 
                 max_K=100, 
                 prior_L= "norm",
                 prior_F= "norm",
                 type_noise='constant' ,
                 maxit= 100 ,
                 param_cebmf_l =None,
                 param_cebmf_f =None,
                 fit_constant= True,
                 init_type ="udv_si"):
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
        denominator_l_hat = np.sum(tau * omega, axis=1) + 1e-32
    else:
        # Fill nan values in Z with 0
        Z_filled = np.nan_to_num(Z, nan=0)
        # Set tau to 0 wherever Z has nan values
        mask_nan_Z = np.isnan(Z)
        tau_spiked = np.copy(tau)  # Make a copy of tau to modify
        tau_spiked[mask_nan_Z] = 0

        # Compute the numerator and denominator using the modified tau and Z
        numerator_l_hat = np.sum(tau_spiked * Z_filled * nu, axis=1)
        denominator_l_hat = np.sum(tau_spiked * omega, axis=1) + 1e-32

    # Compute l_hat
    l_hat = numerator_l_hat / denominator_l_hat

    # Compute s_l
    s_l = (denominator_l_hat) ** (-0.5)
    
    return l_hat, s_l

def compute_hat_f_and_s_f(Z, nu, omega, tau, has_nan=False):
    # Create a mask that identifies non-nan entries in Z
    mask = ~np.isnan(Z)  # Assuming the mask is related to Z having no NaNs

    if not has_nan:
        # Compute the numerator, broadcasting nu properly across columns (broadcast nu over axis=0)
        numerator_f_hat = np.sum(tau * Z * mask * nu[:, np.newaxis], axis=0)  # Sum over i (axis=0)
        denominator_f_hat = np.sum(tau * mask * omega[:, np.newaxis], axis=0) + 1e-32  # Sum over i (axis=0)
    else:
        # Fill nan values in Z with 0
        Z_filled = np.nan_to_num(Z, nan=0)
        # Set tau to 0 wherever Z has nan values
        mask_nan_Z = np.isnan(Z)
        tau_spiked = np.copy(tau)  # Make a copy of tau to modify
        tau_spiked[mask_nan_Z] = 0

        # Compute the numerator and denominator using the modified tau and Z
        numerator_f_hat = np.sum(tau_spiked * Z_filled * mask * nu[:, np.newaxis], axis=0)  # Sum over i (axis=0)
        denominator_f_hat = np.sum(tau_spiked * mask * omega[:, np.newaxis], axis=0) + 1e-32  # Sum over i (axis=0)

    # Compute f_hat
    f_hat = numerator_f_hat / denominator_f_hat

    # Compute s_f
    s_f = (denominator_f_hat) ** (-0.5)

    return f_hat, s_f
