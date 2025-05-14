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
    def __init__(self, X, betahat, sebetahat):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.betahat = torch.tensor(betahat, dtype=torch.float32)
        self.sebetahat = torch.tensor(sebetahat, dtype=torch.float32)  # Noise level for each observation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.betahat[idx], self.sebetahat[idx]  # Return the noise_std (sebetahat) as well

# Mixture Density Network
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_gaussians, n_layers=4):
        super(MDN, self).__init__()
        
        # Input layer
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        
        # Output layers for the Gaussian parameters
        self.pi = nn.Linear(hidden_dim, n_gaussians)  # Mixing coefficients (weights)
        self.mu = nn.Linear(hidden_dim, n_gaussians)  # Means of Gaussians
        self.log_sigma = nn.Linear(hidden_dim, n_gaussians)  # Log of standard deviations

    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        
        # Passing through each hidden layer
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        
        # Outputs
        pi = torch.softmax(self.pi(x), dim=1)  # Softmax for mixture weights
        mu = self.mu(x)  # Mean of each Gaussian
        log_sigma = self.log_sigma(x)  # Log standard deviation for stability
        
        return pi, mu, log_sigma

# Negative log-likelihood loss for MDN with varying observation noise
def mdn_loss_with_varying_noise(pi, mu, log_sigma, betahat, sebetahat):
    sigma =torch.exp(log_sigma)  # Model predicted std (Gaussian std)
    sebetahat = sebetahat.unsqueeze(1)  # Match the dimensions for broadcasting
    total_sigma = torch.sqrt(sigma**2 + sebetahat**2)  # Combine with varying observation noise
    m = torch.distributions.Normal(mu, total_sigma)
    probs = m.log_prob(betahat.unsqueeze(1))  # Log probability of betahat under each Gaussian
    log_probs = probs + torch.log(pi )  # Log-prob weighted by pi
    nll = -torch.logsumexp(log_probs, dim=1)  # Logsumexp for numerical stability
    return nll.mean()


# Class to store the results
class EmdnPosteriorMeanNorm:
    def __init__(self, post_mean, post_mean2, post_sd,location ,pi_np, scale, loss=0,model_param=None):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.location =location  
        self.pi_np=pi_np
        self.scale= scale
        self.loss = loss
        self.model_param= model_param

# Main function to train the model and compute posterior means, mean^2, and standard deviations
def emdn_posterior_means(X, betahat, sebetahat, n_epochs=50 ,n_layers=4, n_gaussians=5, hidden_dim=64, batch_size=1024, lr=0.001, model_param=None):
    # Standardize X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create dataset and dataloader
    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    input_dim = X_scaled.shape[1]  # Number of input features
    model = MDN(input_dim=input_dim, hidden_dim=hidden_dim, n_gaussians=n_gaussians,n_layers=n_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if model_param is not None:
        model.load_state_dict(model_param)

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
            #final_loss = mdn_loss_with_varying_noise(pi, mu, log_sigma, betahat_batch, sebetahat_batch).item()


    # Convert predictions to numpy arrays
    pi_np = pi.numpy()
    mu_np = mu.numpy()
    log_sigma_np = log_sigma.numpy()

    # Initialize arrays to store the results
    post_mean = np.zeros(len(betahat))
    post_mean2 = np.zeros(len(betahat))
    post_sd = np.zeros(len(betahat))

    # Estimate posterior means for each observation
    for i in range(len(betahat)):
        result = posterior_mean_norm(
            betahat=np.array([betahat[i]]),
            sebetahat=np.array([sebetahat[i]]),
            log_pi=np.log(pi_np[i, :]),
            location=mu_np[i, :],
            scale=np.sqrt(np.exp(log_sigma_np[i, :]) ** 2 )
        )
        post_mean[i] = result.post_mean
        post_mean2[i] = result.post_mean2
        post_sd[i] = result.post_sd

    model_param= model.state_dict()
    # Return all three arrays: posterior mean, mean^2, and standard deviation
    return EmdnPosteriorMeanNorm(post_mean,
                                  post_mean2,
                                    post_sd, 
                                    location = mu_np,
                                    pi_np = pi_np,
                                    scale = np.sqrt(np.exp(log_sigma_np ) ** 2 ),
                                    loss= running_loss,model_param=model_param)
