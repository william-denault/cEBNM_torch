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

# Mixture Density Network with modularity
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, n_layers=4):
        super(MDN, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hidden_dim, 1)  # Output logit for pi_2
        self.mu_1 = 0.0  # Fixed mean for component 1
        self.mu_2 = nn.Parameter(torch.tensor(0.0))  # Learnable mean for component 2

    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        logit_pi2 = self.fc_out(x).squeeze()
        pi_2 = torch.sigmoid(logit_pi2)
        pi_1 = 1 - pi_2
        return pi_1, pi_2, self.mu_2

# Loss function for the MDN
def mdn_loss(pi_1, pi_2, mu_2, omega, targets, sd_noise):
    mu_1 = torch.tensor(0.0)
    sigma_1_sq_total = sd_noise**2
    sigma_2_sq=omega*torch.abs(mu_2)
    sigma_2_sq_total = sigma_2_sq + sd_noise**2
    p1 = (1 / torch.sqrt(2 * torch.pi * sigma_1_sq_total)) * torch.exp(-0.5 * ((targets - mu_1) ** 2) / sigma_1_sq_total)
    p2 = (1 / torch.sqrt(2 * torch.pi * sigma_2_sq_total)) * torch.exp(-0.5 * ((targets - mu_2) ** 2) / sigma_2_sq_total)
    mixture_pdf = pi_1 * p1 + pi_2 * p2
    return -torch.mean(torch.log(mixture_pdf + 1e-8))

 

 

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
def cgb_hard_posterior_means(X, betahat, sebetahat, n_epochs=100,  hidden_dim=64, n_layers=4,  batch_size=1024, lr=0.01, model_param=None,omega=0.02):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    input_dim = X_scaled.shape[1]
    model = MDN(input_dim= input_dim,
                  hidden_dim= hidden_dim, 
                  n_layers= n_layers)
    if model_param is not None:
        model.load_state_dict(model_param)
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        for X_batch, y_batch, noise_batch in train_loader:
            pi_1, pi_2, mu_2e = model(X_batch)
           
            loss = mdn_loss(pi_1, pi_2, mu_2e, omega, y_batch, noise_batch)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()} ")

    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        pi_1, pi_2, mu_2e = model(X_tensor)

    pi_np = pi_2.detach().numpy()
    mu_2e = mu_2e.detach().numpy()
    sigma_prior = omega*np.abs(mu_2e)    
    post_mean, post_var = np.zeros_like(betahat), np.zeros_like(betahat)
    for i in range(len(betahat)):
        post_mean[i], post_var[i] = posterior_point_mass_normal(betahat[i], sebetahat[i], 1 - pi_np[i], 0, mu_2e, sigma_prior)
    post_mean2 = post_var + post_mean**2

    return CgbPosteriorMeans(post_mean = post_mean,
                             post_mean2 = post_mean2, 
                             post_sd= np.sqrt(post_var),
                             pi=  pi_np,
                             loss= loss.item(), 
                             model_param=model.state_dict(),
                             mu_1= mu_2e,
                             sigma_0= sigma_prior)
