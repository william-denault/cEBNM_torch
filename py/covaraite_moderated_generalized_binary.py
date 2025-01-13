import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

from posterior_computation import *

# Dataset for regression with observation noise
class DensityRegressionDataset(Dataset):
    def __init__(self, X,  betahat, sebetahat):
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
        # Output layers for Gaussian parameters
        self.pi = nn.Linear(hidden_dim, 1)  # Single output for pi_2 (pi_1 = 1 - pi_2)
        self.mu_2 = nn.Parameter(torch.tensor(0.0))  # Mean of the second component
        self.log_sigma_2 = nn.Parameter(torch.tensor(0.0))  # Log of standard deviation of the second component

    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        pi_2 = torch.sigmoid(self.pi(x))  # Ensure pi_2 is between 0 and 1
        mu_2 = self.mu_2 
        log_sigma_2 = self.log_sigma_2   # Stability for variance
        return pi_2, mu_2, log_sigma_2


# Loss function for two-component MDN
def two_component_mdn_loss(pi_2, mu_2, log_sigma_2, betahat, sebetahat):
    # Fixed parameters for the first component
    mu_1 = torch.tensor(0.0, dtype=torch.float32)
    
    sebetahat = sebetahat.unsqueeze(1)
    sigma_1_sq = torch.sqrt(0**2 + sebetahat**2)
    # Parameters for the second component
    sigma_2 = torch.sqrt(torch.exp(log_sigma_2)**2 + sebetahat**2)

    # Compute Gaussian PDFs
    p1 = (1 / torch.sqrt(2 * torch.pi * sigma_1_sq)) * torch.exp(-0.5 * (( betahat - mu_1) ** 2) / sigma_1_sq)
    p2 = (1 / torch.sqrt(2 * torch.pi * sigma_2 ** 2)) * torch.exp(-0.5 * (( betahat - mu_2) ** 2) / (sigma_2 ** 2))

    # Mixture model likelihood
    mixture_pdf = (1 - pi_2) * p1 + pi_2 * p2

    # Negative log-likelihood
    nll = -torch.mean(torch.log(mixture_pdf + 1e-8))  # Add epsilon for numerical stability
    return nll


# Class to store the results
class CgbPosteriorMeans:
    def __init__(self, post_mean, post_mean2, post_sd,pi=0, loss=0,model_param=None, mu_1=0, sigma_0=1):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.loss = loss
        self.mu_1= mu_1
        self.sigma_0= sigma_0
        self.model_param= model_param
        self.pi=  pi
        self.loss=loss

# Train the two-component MDN
def cgb_posterior_means(X, betahat, sebetahat, n_epochs=20, n_layers=4, hidden_dim=64, batch_size=128, lr=0.001, model_param=None):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # Standardize input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create dataset and dataloader
    dataset = DensityRegressionDataset(X_scaled,   betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    input_dim = X_scaled.shape[1]
    model = cov_GB(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if model_param is not None:
        model.load_state_dict(model_param)
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, noise_std in dataloader:
            optimizer.zero_grad()
            pi_2, mu_2, log_sigma_2 = model(inputs)
            loss = two_component_mdn_loss(pi_2, mu_2, log_sigma_2, targets, noise_std)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss / len(dataloader):.4f}')

    model.eval()
    with torch.no_grad():
        pi_2, mu_2, log_sigma_2 = model(torch.tensor(X_scaled, dtype=torch.float32))

    pi_np = pi_2.detach().numpy()
    mu_2= mu_2.detach().numpy()
    sigma_prior = np.exp( log_sigma_2.detach().numpy())


    post_mean = np.zeros_like(betahat)
    post_var = np.zeros_like(betahat)

    for i in range(len(betahat)):
        post_mean[i], post_var[i] = posterior_point_mass_normal(
            betahat=betahat[i],
            sebetahat=sebetahat[i],
            pi=( pi_np[i]),  # Scalar pi for each observation
            mu0=0,  # Fixed mean for the point mass
            mu1=mu_2,  # Global parameter
            sigma_0=sigma_prior  # Global parameter
            )
    post_mean2= post_var+ post_mean**2
    model_param= model.state_dict()
    # Return all three arrays: posterior mean, mean^2, and standard deviation
    return CgbPosteriorMeans(post_mean=post_mean,
                             post_mean2=post_mean2, 
                             post_sd= np.sqrt(post_var),
                             pi= pi_np,
                             mu_1=mu_2,
                             sigma_0=sigma_prior,
                             loss= running_loss,model_param=model_param)
