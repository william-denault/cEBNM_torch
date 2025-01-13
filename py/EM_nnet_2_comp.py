import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm


class EMAlgorithmWithNNet:
    def __init__(self, x, betahat, sd_noise, hidden_dim=16, lr=0.01, max_iter=100, tol=1e-6):
        """
        EM algorithm for fitting a mixture model with covariate-dependent mixture proportions using a neural network.

        Parameters:
        - x: Covariates (2D array, shape [n_samples, n_features]).
        - betahat: Observed data (1D array, shape [n_samples]).
        - sd_noise: Fixed observation noise standard deviation.
        - hidden_dim: Hidden layer size for the neural network.
        - lr: Learning rate for the neural network optimizer.
        - max_iter: Maximum number of EM iterations.
        - tol: Tolerance for convergence.
        """
        self.x = torch.tensor(x, dtype=torch.float32)  # Covariates
        self.betahat = betahat
        self.sd_noise = sd_noise
        self.max_iter = max_iter
        self.tol = tol

        # Initialize Gaussian parameters
        self.mu_2 = np.mean(betahat)  # Initial guess for Gaussian mean
        self.sigma_2_sq = np.var(betahat)  # Initial guess for Gaussian variance

        # Neural network for \(\pi(x)\)
        self.pi_net = nn.Sequential(
            nn.Linear(x.shape[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Ensure output is in (0, 1)
        )
        self.optimizer = optim.Adam(self.pi_net.parameters(), lr=lr)

    def e_step(self):
        """
        E-step: Compute posterior probabilities (responsibilities) q_i.
        """
        # Predict \(\pi(x)\) using the neural network
        self.pi_net.eval()
        with torch.no_grad():
            pi_2 = self.pi_net(self.x).squeeze().numpy()
        pi_1 = 1 - pi_2

        # Total variance for the Gaussian component
        total_var_2 = self.sigma_2_sq + self.sd_noise**2

        # Compute likelihoods
        p1 = norm.pdf(self.betahat, 0, self.sd_noise)  # Point mass likelihood
        p2 = norm.pdf(self.betahat, self.mu_2, np.sqrt(total_var_2))  # Gaussian likelihood

        # Posterior probabilities (responsibilities)
        q = (pi_2 * p2) / (pi_1 * p1 + pi_2 * p2 + 1e-8)  # Add epsilon for stability
        return q, pi_1, pi_2

    def m_step(self, q):
        """
        M-step: Update parameters using current responsibilities.
        """
        # Update neural network for \(\pi(x)\) using current responsibilities
        self.pi_net.train()
        q_tensor = torch.tensor(q, dtype=torch.float32)  # Ensure same shape as pi_2
        for _ in range(100):  # Fixed number of optimization steps
            self.optimizer.zero_grad()
            pi_2 = self.pi_net(self.x).squeeze()
            loss = nn.BCELoss()(pi_2, q_tensor)  # Binary cross-entropy loss
            loss.backward()
            self.optimizer.step()

        # Update mu_2 (Gaussian mean)
        self.mu_2 = np.sum(q * self.betahat) / np.sum(q)

        # Update sigma_2_sq (Gaussian variance)
        total_var = np.sum(q * (self.betahat - self.mu_2) ** 2) / np.sum(q)
        self.sigma_2_sq = max(total_var - self.sd_noise**2, 1e-8)  # Ensure non-negative variance

    def run(self):
        """
        Run the EM algorithm until convergence.

        Returns:
        - pi_net: Trained neural network for \(\pi(x)\).
        - mu_2: Estimated mean of the Gaussian component.
        - sigma_2_sq: Estimated variance of the Gaussian component.
        """
        for iteration in range(self.max_iter):
            # E-step
            q, pi_1, pi_2 = self.e_step()

            # Save old parameters to check for convergence
            old_params = np.array([self.mu_2, self.sigma_2_sq])

            # M-step
            self.m_step(q)

            # Check convergence
            new_params = np.array([self.mu_2, self.sigma_2_sq])
            if np.linalg.norm(new_params - old_params) < self.tol:
                print(f"Converged in {iteration + 1} iterations.")
                break
        else:
            print("Maximum iterations reached without convergence.")

        return self.pi_net, self.mu_2, self.sigma_2_sq