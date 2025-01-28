import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(r"D:\Document\Serieux\Travail\python_work\cEBNM_torch\py")
from posterior_computation import *
# Dataset for regression with observation noise
class DensityRegressionDataset(Dataset):
    def __init__(self, X, betahat, sebetahat):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.betahat = torch.tensor(betahat, dtype=torch.float32)
        self.sebetahat = torch.tensor(sebetahat, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.betahat[idx], self.sebetahat[idx]

# Two-Component Mixture Density Network
class MDN(nn.Module):
    def __init__(self, input_dim):
        super(MDN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output: logit for pi_2
        )
        self.mu_1 = 0.0  # Fixed mean for component 1
        self.mu_2 = nn.Parameter(torch.tensor(0.0))  # Learnable mean for component 2

    def forward(self, x):
        logit_pi2 = self.fc(x).squeeze()
        pi_2 = torch.sigmoid(logit_pi2)
        pi_1 = 1 - pi_2
        return pi_1, pi_2, self.mu_2

# Loss function for the MDN
def mdn_loss(pi_1, pi_2, mu_2, sigma_2_sq, targets, sd_noise):
    mu_1 = torch.tensor(0.0)
    sigma_1_sq_total = sd_noise**2
    sigma_2_sq_total = sigma_2_sq + sd_noise**2

    p1 = (1 / torch.sqrt(2 * torch.pi * sigma_1_sq_total)) * torch.exp(-0.5 * ((targets - mu_1) ** 2) / sigma_1_sq_total)
    p2 = (1 / torch.sqrt(2 * torch.pi * sigma_2_sq_total)) * torch.exp(-0.5 * ((targets - mu_2) ** 2) / sigma_2_sq_total)

    mixture_pdf = pi_1 * p1 + pi_2 * p2
    return -torch.mean(torch.log(mixture_pdf + 1e-8))

# Compute responsibilities
def compute_responsibilities(pi_1, pi_2, mu_2, sigma_2_sq, targets, sd_noise):
    sigma_1_sq_total = sd_noise**2
    sigma_2_sq_total = sigma_2_sq + sd_noise**2

    p1 = (1 / torch.sqrt(2 * torch.pi * sigma_1_sq_total)) * torch.exp(-0.5 * (targets**2) / sigma_1_sq_total)
    p2 = (1 / torch.sqrt(2 * torch.pi * sigma_2_sq_total)) * torch.exp(-0.5 * ((targets - mu_2) ** 2) / sigma_2_sq_total)

    return (pi_2 * p2) / (pi_1 * p1 + pi_2 * p2)

# Perform the M-step to update sigma_2^2
def m_step_sigma2(gamma_2, mu_2, targets, sd_noise):
    residuals_sq = (targets - mu_2) ** 2
    sigma_0_sq = sd_noise**2
    numerator = torch.sum(gamma_2 * (residuals_sq - sigma_0_sq))
    denominator = torch.sum(gamma_2)
    return torch.clamp(numerator / denominator, min=1e-6)

# Class to store the results
class CgbPosteriorMeans:
    def __init__(self, post_mean, post_mean2, post_sd, pi=0, loss=0, model_param=None, mu_1=0, sigma_0=1):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.loss = loss
        self.mu_1 = mu_1
        self.sigma_0 = sigma_0
        self.model_param = model_param
        self.pi = pi

# Train the MDN model and compute posterior means
def cgb_posterior_means(X, betahat, sebetahat, n_epochs=1000, batch_size=1024, lr=0.01):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    input_dim = X_scaled.shape[1]
    model = MDN(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    sigma_2_sqe = torch.tensor(1, requires_grad=False)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        for X_batch, y_batch, noise_batch in train_loader:
            pi_1, pi_2, mu_2e = model(X_batch)

            gamma_2 = compute_responsibilities(
                pi_1=pi_1, pi_2=pi_2, mu_2=mu_2e, sigma_2_sq=sigma_2_sqe, targets=y_batch, sd_noise=noise_batch)

            with torch.no_grad():
                sigma_2_sqe = m_step_sigma2(gamma_2, mu_2e, y_batch, noise_batch)

            loss = mdn_loss(pi_1, pi_2, mu_2e, sigma_2_sqe, y_batch, noise_batch)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Sigma_2^2: {sigma_2_sqe.item()}")

    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        pi_1, pi_2, mu_2e = model(X_tensor)

    pi_np = pi_2.detach().numpy()
    mu_2e = mu_2e.detach().numpy()
    sigma_prior = np.sqrt(sigma_2_sqe.detach().numpy())

    post_mean = np.zeros_like(betahat)
    post_var = np.zeros_like(betahat)

    for i in range(len(betahat)):
        post_mean[i], post_var[i] = posterior_point_mass_normal(
            betahat=betahat[i],
            sebetahat=sebetahat[i],
            pi=(1 - pi_np[i]),
            mu0=0,
            mu1=mu_2e,
            sigma_0=sigma_prior
        )
    post_mean2 = post_var + post_mean**2

    return CgbPosteriorMeans(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=np.sqrt(post_var),
        pi=pi_np,
        mu_1=mu_2e,
        sigma_0=sigma_prior,
        loss=loss.item(),
        model_param=model.state_dict()
    )