import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader as GeoDataLoader

from posterior_computation import *  # Assume this contains posterior_mean_norm

# Dataset class for graph-based side info
class GraphDensityRegressionDataset(Dataset):
    def __init__(self, node_features, edge_index, betahat, sebetahat):
        self.data = Data(x=torch.tensor(node_features, dtype=torch.float32),
                         edge_index=torch.tensor(edge_index, dtype=torch.long))
        self.betahat = torch.tensor(betahat, dtype=torch.float32)
        self.sebetahat = torch.tensor(sebetahat, dtype=torch.float32)

    def __len__(self):
        return len(self.betahat)

    def __getitem__(self, idx):
        return self.data.x[idx], self.data.edge_index, self.betahat[idx], self.sebetahat[idx]

# Graph-based MDN using GCN layers
class GraphMDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_gaussians, n_layers=2):
        super(GraphMDN, self).__init__()
        self.convs = nn.ModuleList([GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(n_layers)])
        self.pi = nn.Linear(hidden_dim, n_gaussians)
        self.mu = nn.Linear(hidden_dim, n_gaussians)
        self.log_sigma = nn.Linear(hidden_dim, n_gaussians)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        pi = torch.softmax(self.pi(x), dim=1)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        return pi, mu, log_sigma

# MDN loss supporting observation noise
def mdn_loss_with_varying_noise(pi, mu, log_sigma, betahat, sebetahat):
    sigma = torch.exp(log_sigma)
    total_sigma = torch.sqrt(sigma**2 + sebetahat.unsqueeze(1)**2)
    m = torch.distributions.Normal(mu, total_sigma)
    log_probs = m.log_prob(betahat.unsqueeze(1)) + torch.log(pi)
    nll = -torch.logsumexp(log_probs, dim=1)
    return nll.mean()

# Posterior result class
class EmdnPosteriorMeanNorm:
    def __init__(self, post_mean, post_mean2, post_sd, loss=0, model_param=None):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.loss = loss
        self.model_param = model_param

# Main training + inference function for spatial transcriptomic graphs

def emdn_posterior_means_graph(node_features, edge_index, betahat, sebetahat, n_epochs=50, n_gaussians=5,
                                hidden_dim=64, batch_size=1024, lr=0.001, model_param=None, n_layers=2):
    dataset = GraphDensityRegressionDataset(node_features, edge_index, betahat, sebetahat)
    dataloader = GeoDataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = node_features.shape[1]
    model = GraphMDN(input_dim, hidden_dim, n_gaussians, n_layers=n_layers)
    if model_param is not None:
        model.load_state_dict(model_param)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for x, edge_idx, y, noise in dataloader:
            optimizer.zero_grad()
            pi, mu, log_sigma = model(x, edge_idx)
            loss = mdn_loss_with_varying_noise(pi, mu, log_sigma, y, noise)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(dataloader):.4f}")

    model.eval()
    with torch.no_grad():
        all_pi, all_mu, all_log_sigma = [], [], []
        for x, edge_idx, _, _ in GeoDataLoader(dataset, batch_size=len(dataset)):
            pi, mu, log_sigma = model(x, edge_idx)
            all_pi = pi.numpy()
            all_mu = mu.numpy()
            all_log_sigma = log_sigma.numpy()

    post_mean = np.zeros(len(betahat))
    post_mean2 = np.zeros(len(betahat))
    post_sd = np.zeros(len(betahat))

    for i in range(len(betahat)):
        result = posterior_mean_norm(
            betahat=np.array([betahat[i]]),
            sebetahat=np.array([sebetahat[i]]),
            log_pi=np.log(all_pi[i, :]),
            location=all_mu[i, :],
            scale=np.sqrt(np.exp(all_log_sigma[i, :])**2)
        )
        post_mean[i] = result.post_mean
        post_mean2[i] = result.post_mean2
        post_sd[i] = result.post_sd

    return EmdnPosteriorMeanNorm(post_mean, post_mean2, post_sd, loss=running_loss, model_param=model.state_dict())
