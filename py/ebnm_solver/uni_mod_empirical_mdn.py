# Define dataset class that includes observation noise
 

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


import os
import sys
sys.path.append(r"D:\Document\Serieux\Travail\python_work\cEBNM_torch\py")
# Import utils.py directly
from utils import *
from numerical_routine import *
from distribution_operation import *
from posterior_computation import *

 
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

# Define the MeanNet model
class MeanNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(MeanNet, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input layer
        x = self.relu(self.input_layer(x))
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        
        # Output layer
        x = self.output_layer(x)
        return x

# Define the CashNet model
class CashNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_layers):
        super(CashNet, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input layer
        x = self.relu(self.input_layer(x))
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        
        # Output layer
        x = self.softmax(self.output_layer(x))
        return x

# Custom loss function
def pen_loglik_loss(pred_pi, marginal_log_lik, penalty=1.1, epsilon=1e-10):
    L_batch = torch.exp(marginal_log_lik)
    inner_sum = torch.sum(pred_pi * L_batch, dim=1)
    inner_sum = torch.clamp(inner_sum, min=epsilon)
    first_sum = torch.sum(torch.log(inner_sum))

    if penalty > 1:
        pi_clamped = torch.clamp(torch.sum(pred_pi[:, 0]), min=epsilon)
        penalized_log_likelihood_value = first_sum + (penalty - 1) * torch.log(pi_clamped)
    else:
        penalized_log_likelihood_value = first_sum

    return -penalized_log_likelihood_value




# Class to store the results
class Unimod_Emdn_PosteriorMeanNorm:
    def __init__(self, post_mean, post_mean2, post_sd, loss=0,model_param=None):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.loss = loss
        self.model_param= model_param


# Main function to train the model and compute posterior means, mean^2, and standard deviations
def unimod_emdn_posterior_means(X, betahat, sebetahat, n_epochs=100 ,n_layers=4,  num_classes=20, hidden_dim=64, batch_size=128, lr=0.001, model_param=None,penalty=1.5):
    # Standardize X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scale = autoselect_scales_mix_norm(betahat=betahat,
                                   sebetahat=sebetahat,
                                       max_class=num_classes ) 
    # Create dataset and dataloader
    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = X_scaled.shape[1]
    model_mean = MeanNet(input_dim=input_dim, hidden_dim=hidden_dim,  n_layers=n_layers)
    model_cash = CashNet(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes,n_layers=n_layers)
    optimizer_mean = optim.Adam(model_mean.parameters(), lr=lr)
    optimizer_cash = optim.Adam(model_cash.parameters(), lr=lr)

    mean_criterion = nn.MSELoss()
    # Training loop
    # Training loop
    for epoch in range(n_epochs):
        total_mean_loss = 0
        total_cash_loss = 0
        running_loss = 0.0
        for inputs, targets, noise_std in dataloader:
            optimizer_cash.zero_grad()
            mean_predictions =  model_mean(inputs).squeeze(-1)
            outputs = model_cash(inputs) 
            residuals =targets.detach().numpy()-mean_predictions.detach().numpy()
            batch_loglik = get_data_loglik_normal(betahat=residuals,
                                              sebetahat=noise_std,
                                              location=0*scale,
                                              scale=scale)
            optimizer_cash.zero_grad()
            outputs = model_cash(inputs)
        
            cash_loss = pen_loglik_loss(pred_pi=  outputs, 
                                 marginal_log_lik=  torch.tensor(batch_loglik),
                                 penalty=penalty)
            cash_loss.backward()
            optimizer_cash.step()
            total_cash_loss += cash_loss.item()



            optimizer_mean.zero_grad()
            #mean_predictions =  model_mean(inputs).squeeze(-1)
            mean_loss = mean_criterion(mean_predictions, targets)
            mean_loss.backward()
            optimizer_mean.step()
            total_mean_loss += mean_loss.item()
 
           
 
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Mean Loss: {total_mean_loss / len(dataloader):.4f}, Variance Loss: {total_cash_loss / len(dataloader):.4f}")
            # After training the model, compute the posterior mean
    model_mean.eval()
    model_cash.eval()

    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for X_batch, _, _ in train_loader:
            all_pi_values = model_cash(X_batch)
            all_location  = model_mean(X_batch)

    all_pi_values_np = all_pi_values.detach().numpy()
    all_location  =  all_location.detach().numpy()
    # Initialize arrays to store the results
    post_mean = np.zeros(len(betahat))
    post_mean2 = np.zeros(len(betahat))
    post_sd = np.zeros(len(betahat))

    # Estimate posterior means for each observation
    for i in range(len(betahat)):
        result = posterior_mean_norm(
        betahat=np.array([betahat[i]]),
        sebetahat=np.array([sebetahat[i]]),
        log_pi=np.log(all_pi_values_np[i, :]),
        location =  all_location[i],
        scale= scale  # Assuming this is available from earlier in your code
        )
        post_mean[i] = result.post_mean
        post_mean2[i] = result.post_mean2
        post_sd[i] = result.post_sd

    
    return  Unimod_Emdn_PosteriorMeanNorm(post_mean, post_mean2, post_sd, loss= running_loss,model_param=model_param)