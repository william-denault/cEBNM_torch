import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import *
from numerical_routine import *
from distribution_operation import *
from posterior_computation import *
from ash import *

# Define dataset class that includes observation noise
class DensityRegressionDataset(Dataset):
    def __init__(self, X, y, noise_std):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.noise_std = torch.tensor(noise_std, dtype=torch.float32)  # Noise level for each observation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.noise_std[idx]  # Return the noise_std as well

# Mixture Density Network
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_gaussians):
        super(MDN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi = nn.Linear(hidden_dim, n_gaussians)  # Mixing coefficients (weights)
        self.mu = nn.Linear(hidden_dim, n_gaussians)  # Means of Gaussians
        self.log_sigma = nn.Linear(hidden_dim, n_gaussians)  # Log of standard deviations

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        pi = torch.softmax(self.pi(x), dim=1)  # Softmax for mixture weights
        mu = self.mu(x)  # Mean of each Gaussian
        log_sigma = self.log_sigma(x)  # Log standard deviation for stability
        return pi, mu, log_sigma

# Negative log-likelihood loss for MDN with varying observation noise
def mdn_loss_with_varying_noise(pi, mu, log_sigma, y, obs_noise_std):
    sigma = torch.exp(log_sigma)  # Model predicted std (Gaussian std)
    obs_noise_std = obs_noise_std.unsqueeze(1)  # Match the dimensions for broadcasting
    total_sigma = torch.sqrt(sigma**2 + obs_noise_std**2)  # Combine with varying observation noise
    m = torch.distributions.Normal(mu, total_sigma)
    probs = m.log_prob(y.unsqueeze(1))  # Log probability of y under each Gaussian
    log_probs = probs + torch.log(pi + 1e-8)  # Log-prob weighted by pi
    nll = -torch.logsumexp(log_probs, dim=1)  # Logsumexp for numerical stability
    return nll.mean()



class EmdnPosteriorMeanNorm:
    def __init__(self, post_mean, post_mean2, post_sd):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
# Main function to train the model and compute posterior means
# Main function to train the model and compute posterior means, mean^2, and standard deviations
def emdn_posterior_means(X, y_obs, obs_noise_std, n_epochs=200, n_gaussians=5, hidden_dim=64, batch_size=128,lr=0.001):
    # Standardize X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create dataset and dataloader
    dataset = DensityRegressionDataset(X_scaled, y_obs, obs_noise_std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    input_dim = X_scaled.shape[1]  # Number of input features
    model = MDN(input_dim=input_dim, hidden_dim=hidden_dim, n_gaussians=n_gaussians)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, noise_std in dataloader:
            optimizer.zero_grad()
            pi, mu, log_sigma = model(inputs)
            loss = mdn_loss_with_varying_noise(pi, mu, log_sigma, targets, noise_std)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(dataloader):.4f}')

    # Once trained, generate posterior means for the entire dataset
    model.eval()
    with torch.no_grad():
        # Convert the entire dataset into a batch for prediction
        train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for X_batch, _, _ in train_loader:
            pi, mu, log_sigma = model(X_batch)  # Predict pi, mu, and log_sigma

    # Convert predictions to numpy arrays
    pi_np = pi.numpy()
    mu_np = mu.numpy()
    log_sigma_np = log_sigma.numpy()

    # Initialize arrays to store the results
    post_mean = np.zeros(len(y_obs))
    post_mean2 = np.zeros(len(y_obs))
    post_sd = np.zeros(len(y_obs))

    # Estimate posterior means for each observation
    for i in range(len(y_obs)):
        result = posterior_mean_norm(
            betahat=np.array([y_obs[i]]),
            sebetahat=np.array([obs_noise_std[i]]),
            log_pi=np.log(pi_np[i, :]),
            location=mu_np[i, :],
            scale=np.sqrt(np.exp(log_sigma_np[i, :]) ** 2)
        )
        post_mean[i] = result.post_mean
        post_mean2[i] = result.post_mean2
        post_sd[i] = result.post_sd

    # Return all three arrays: posterior mean, mean^2, and standard deviation
    return EmdnPosteriorMeanNorm( post_mean, post_mean2, post_sd)
