import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from posterior_computation import *

# Dataset for regression with observation noise
class DensityRegressionDataset(Dataset):
    def __init__(self, X, betahat, sebetahat):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.betahat = torch.tensor(betahat, dtype=torch.float32)
        self.sebetahat = torch.tensor(sebetahat, dtype=torch.float32)  # Noise level for each observation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.betahat[idx], self.sebetahat[idx]  # Return the noise_std (sebetahat) as well

# Two-Component Mixture Density Network
class cov_GB(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=4):
        super(cov_GB, self).__init__()
        # Input layer
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        # Hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        # Output layer for logits of pi_2
        self.fc_out = nn.Linear(hidden_dim, 1)
        # Learnable parameter for the second component
        self.mu_2 = nn.Parameter(torch.tensor(0.0))  # Learnable mean for component 2

    def forward(self, x):
        # Input layer
        x = torch.relu(self.fc_in(x))
        # Hidden layers
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        # Output layer for logits
        logit_pi2 = self.fc_out(x).squeeze()
        pi_2 = torch.sigmoid(logit_pi2)
        pi_1 = 1 - pi_2
        return pi_1, pi_2, self.mu_2


# Loss function for two-component MDN
def two_component_mdn_loss(pi_1, pi_2, mu_2, sigma_2_sq, targets, sd_noise):
    mu_1 = torch.tensor(0.0)
    sigma_1_sq_total = sd_noise**2
    sigma_2_sq_total = sigma_2_sq + sd_noise**2

    p1 = (1 / torch.sqrt(2 * torch.pi * sigma_1_sq_total)) * torch.exp(-0.5 * ((targets - mu_1) ** 2) / sigma_1_sq_total)
    p2 = (1 / torch.sqrt(2 * torch.pi * sigma_2_sq_total)) * torch.exp(-0.5 * ((targets - mu_2) ** 2) / sigma_2_sq_total)

    mixture_pdf = pi_1 * p1 + pi_2 * p2
    return -torch.mean(torch.log(mixture_pdf + 1e-8))

# Compute responsibilities (gamma values)
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

# Train the two-component MDN
def cgb_posterior_means(X, betahat, sebetahat, n_epochs=50, n_layers=4, hidden_dim=64, batch_size=128, lr=0.001, model_param=None):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_scaled.shape[1]
    model = cov_GB(input_dim, hidden_dim, n_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sigma_2_sqe = torch.tensor(.50, requires_grad=False)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        full_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            for inputs, targets, noise_std in full_dataloader:
                pi_1, pi_2, mu_2e = model(inputs)
                gamma_2 = compute_responsibilities(
                pi_1=pi_1, pi_2=pi_2, mu_2=mu_2e,
                sigma_2_sq=sigma_2_sqe,
                targets=targets,  # Use scaled targets
                sd_noise=noise_std  # Use scaled noise_std
                )
            sigma_2_sqe=  m_step_sigma2(gamma_2=gamma_2,
                                         mu_2= mu_2e, targets=targets, sd_noise=noise_std)
            
        for inputs, targets, noise_std in dataloader:
            optimizer.zero_grad()
            pi_1, pi_2, mu_2e = model(inputs)

             

            loss = two_component_mdn_loss(pi_1, pi_2, mu_2e, sigma_2_sqe, targets, noise_std)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
       #print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss / len(dataloader):.4f}")
 
    model.eval()
    with torch.no_grad():
        full_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for inputs, targets, noise_std in full_dataloader:
            pi_1, pi_2, mu_2e = model(inputs)

    


        
    pi_np = pi_2.detach().numpy()
    mu_2e = mu_2e.detach().numpy()
    sigma_prior = np.sqrt(sigma_2_sqe.detach().numpy())

    post_mean = np.zeros_like(betahat)
    post_var = np.zeros_like(betahat)

    for i in range(len(betahat)):
        post_mean[i], post_var[i] = posterior_point_mass_normal(
            betahat=betahat[i],
            sebetahat=sebetahat[i],
            pi=(1 - pi_np[i]),  # Scalar pi for each observation
            mu0=0,  # Fixed mean for the point mass
            mu1=mu_2e,  # Global parameter
            sigma_0=sigma_prior  # Global parameter
        )
    post_mean2 = post_var + post_mean**2
    model_param = model.state_dict()

    return CgbPosteriorMeans(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=np.sqrt(post_var),
        pi=pi_np,
        mu_1=mu_2e,
        sigma_0=sigma_prior,
        loss=running_loss,
        model_param=model_param
    )