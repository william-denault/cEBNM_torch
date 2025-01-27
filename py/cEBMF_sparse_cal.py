import numpy as np
from scipy.sparse import csr_matrix, issparse
from fancyimpute import IterativeSVD
from sklearn.decomposition import NMF

class cEBMF_object:
    def __init__(
        self,
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
        init_type="udv_si",
    ):
        # Convert dense input to sparse if applicable
        if not issparse(data):
            data = csr_matrix(data)
        self.data = data  # Sparse matrix
        self.K = K
        self.X_l = X_l
        self.X_f = X_f
        self.max_K = max_K
        self.prior_L = prior_L
        self.prior_F = prior_F
        self.type_noise = type_noise
        self.maxit = maxit
        self.param_cebmf_l = param_cebmf_l
        self.param_cebmf_f = param_cebmf_f
        self.kl_l = np.zeros(K, dtype=np.float32)
        self.kl_f = np.zeros(K, dtype=np.float32)
        self.fit_constant = fit_constant
        self.init_type = init_type
        self.has_nan = np.isnan(self.data.toarray()).any()
        self.n_nonmissing = self.data.size - self.data.nnz  # Non-missing entries
        self.obj = [np.inf]
        self.model_list_L = [None] * K
        self.model_list_F = [None] * K

    def init_LF(self, use_nmf=False):
        if self.has_nan:
            print("Sparse data with missing values detected. Converting to dense for imputation.")
            imputed_data = self.data.toarray()  # Convert to dense for imputation
            imputed_data = IterativeSVD().fit_transform(imputed_data)
        else:
            imputed_data = self.data.toarray() if issparse(self.data) else self.data

        if use_nmf:
            nmf_model = NMF(n_components=self.K, init="random", random_state=42, max_iter=500)
            self.L = nmf_model.fit_transform(imputed_data).astype(np.float32)
            self.F = nmf_model.components_.T.astype(np.float32)
        else:
            U, s, Vt = np.linalg.svd(imputed_data, full_matrices=False)
            K = min(self.K, len(s))
            U_k = U[:, :K]
            D_k = np.diag(s[:K])
            V_k = Vt[:K, :]
            self.L = (U_k @ D_k).astype(np.float32)
            self.F = V_k.T.astype(np.float32)

        self.L2 = self.L**2
        self.F2 = self.F**2
        self.update_tau()

    def update_fitted_val(self):
        self.Y_fit = csr_matrix(self.data.shape, dtype=np.float32)  # Initialize sparse matrix
        for k in range(self.K):
            # Sparse outer product using scipy.sparse
            self.Y_fit += csr_matrix(self.L[:, k][:, np.newaxis] @ self.F[:, k][np.newaxis, :])

    def cal_expected_residuals(self):
        prod_square_firstmom = csr_matrix(self.data.shape, dtype=np.float32)
        prod_sectmom = csr_matrix(self.data.shape, dtype=np.float32)

        for k in range(self.K):
            l_k_square = csr_matrix(self.L[:, k][:, np.newaxis] ** 2)
            f_k_square = csr_matrix(self.F[:, k][np.newaxis, :] ** 2)
            prod_square_firstmom += l_k_square @ f_k_square
            prod_sectmom += l_k_square @ f_k_square

        self.update_fitted_val()
        R2 = (self.data - self.Y_fit).multiply(self.data - self.Y_fit) - prod_square_firstmom + prod_sectmom
        return R2

    def cal_partial_residuals(self, k):
        idx_loop = set(range(self.K)) - {k}
        self.Rk = self.data.copy()
        for j in idx_loop:
            self.Rk -= csr_matrix(self.L[:, j][:, np.newaxis] @ self.F[:, j][np.newaxis, :])

    def update_tau(self):
        R2 = self.cal_expected_residuals()
        if self.type_noise == 'constant':
            mean_R2 = np.mean(R2.data)  # Sparse-aware mean calculation
            self.tau = np.full(self.data.shape, 1 / (mean_R2 + 1e-8), dtype=np.float32)
        elif self.type_noise == 'row_wise':
            row_means = np.array(R2.mean(axis=1)).flatten() + 1e-8
            self.tau = np.tile(1 / row_means, (self.data.shape[1], 1)).T
        elif self.type_noise == 'column_wise':
            col_means = np.array(R2.mean(axis=0)).flatten() + 1e-8
            self.tau = np.tile(1 / col_means, (self.data.shape[0], 1))

    def update_loading_factor_k(self, k):
        self.cal_partial_residuals(k=k)

        nu = self.F[:, k]  # Dense vector
        omega = self.F2[:, k]  # Dense vector

        lhat, s_l = compute_hat_l_and_s_l(
            Z=self.Rk,
            nu=nu,
            omega=omega,
            tau=self.tau,
            has_nan=self.has_nan
        )

        if self.prior_L == "norm" or self.prior_L == "exp":
            ash_obj = ash(
                betahat=lhat,
                sebetahat=s_l,
                prior=self.prior_L,
                verbose=False
            )
            self.L[:, k] = ash_obj.post_mean
            self.L2[:, k] = ash_obj.post_mean2

            self.kl_l[k] = ash_obj.log_lik - normal_means_loglik(
                x=lhat,
                s=s_l,
                Et=ash_obj.post_mean,
                Et2=ash_obj.post_mean2
            )

    def iter(self):
        for k in range(self.K):
            self.update_loading_factor_k(k=k)
        self.update_tau()
        self.cal_obj()

    def cal_obj(self):
        KL = sum(self.kl_f) + sum(self.kl_l)
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
    init_type="udv_si",
):
    return cEBMF_object(
        data=data,
        K=K,
        X_l=X_l,
        X_f=X_f,
        max_K=max_K,
        prior_L=prior_L,
        prior_F=prior_F,
        type_noise=type_noise,
        maxit=maxit,
        param_cebmf_l=param_cebmf_l,
        param_cebmf_f=param_cebmf_f,
        fit_constant=fit_constant,
        init_type=init_type,
    )

def compute_hat_l_and_s_l(Z, nu, omega, tau, has_nan=False):
    if has_nan:
        Z = Z.toarray()  # Convert to dense if NaN handling is required

    numerator_l_hat = Z @ nu
    denominator_l_hat = Z @ omega + 1e-8

    l_hat = numerator_l_hat / (denominator_l_hat + 1e-6)
    s_l = (1 / np.sqrt(denominator_l_hat + 1e-6))

    return l_hat, s_l

def compute_hat_f_and_s_f(Z, nu, omega, tau, has_nan=False):
    if has_nan:
        Z = Z.toarray()  # Convert to dense if NaN handling is required

    numerator_f_hat = Z.T @ nu
    denominator_f_hat = Z.T @ omega + 1e-8

    f_hat = numerator_f_hat / (denominator_f_hat + 1e-6)
    s_f = (1 / np.sqrt(denominator_f_hat + 1e-6))

    return f_hat, s_f

def normal_means_loglik(x, s, Et, Et2):
    idx = np.isfinite(s) & (s > 0)
    x = x[idx]
    s = s[idx]
    Et = Et[idx]
    Et2 = Et2[idx]

    return -0.5 * np.sum(np.log(2 * np.pi * s**2) + (1 / s**2) * (Et2 - 2 * x * Et + x**2))
