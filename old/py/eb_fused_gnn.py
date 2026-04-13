import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader
import numpy as np

class ManualStyleGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, n_layers):
        super().__init__()
        self.input_layer = nn.Linear(node_feat_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        return self.output_layer(h).squeeze(-1)

class EBGNNDenoiser(nn.Module):
    def __init__(self, gnn_model, lambda1=0.1, lambda2=0.1):
        super().__init__()
        self.gnn = gnn_model
        self.lambda1 = nn.Parameter(torch.tensor(lambda1))
        self.lambda2 = nn.Parameter(torch.tensor(lambda2))

    def marginal_log_likelihood(self, beta_hat, se, X, edge_index):
        mu = self.gnn(X, edge_index).flatten()
        sigma = se
        lambda1, lambda2 = self.lambda1, self.lambda2

        log_lap1 = -torch.abs(beta_hat) / lambda1 - torch.log(2 * lambda1)
        log_lap2 = -torch.abs(beta_hat - mu) / lambda2 - torch.log(2 * lambda2)
        prior_logprob = torch.logaddexp(log_lap1, log_lap2)

        log_lik = -0.5 * ((beta_hat - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * torch.log(torch.tensor(2 * np.pi))
        return -torch.mean(prior_logprob + log_lik)

    def posterior_mean(self, beta_hat, se, X, edge_index):
        mu = self.gnn(X, edge_index).detach().flatten()
        lambda1, lambda2 = self.lambda1, self.lambda2

        w1 = torch.exp(-torch.abs(beta_hat) / lambda1) / (2 * lambda1)
        w2 = torch.exp(-torch.abs(beta_hat - mu) / lambda2) / (2 * lambda2)
        norm = w1 + w2
        return (w2 * mu) / norm, w1.detach().numpy(), w2.detach().numpy(), mu.detach().numpy()

class EGNNFusedPosteriorResult:
    def __init__(self, post_mean, prior_weights_1, prior_weights_2, location, model_param, loss):
        self.post_mean = post_mean
        self.w1 = prior_weights_1
        self.w2 = prior_weights_2
        self.location = location
        self.model_param = model_param
        self.loss = loss

def egnnfused_posterior_means(X, beta_hat, se, edge_index, n_layers=2, hidden_dim=64,
                               n_epochs=100, lr=1e-3, batch_size=None, lambda1=0.1, lambda2=0.1,
                               model_param=None):

    X = torch.tensor(X, dtype=torch.float32)
    beta_hat = torch.tensor(beta_hat, dtype=torch.float32)
    se = torch.tensor(se, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    gnn_model = ManualStyleGNN(node_feat_dim=X.shape[1], hidden_dim=hidden_dim, n_layers=n_layers)
    eb_model = EBGNNDenoiser(gnn_model=gnn_model, lambda1=lambda1, lambda2=lambda2)

    if model_param is not None:
        eb_model.load_state_dict(model_param)

    optimizer = torch.optim.Adam(eb_model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        eb_model.train()
        optimizer.zero_grad()
        loss = eb_model.marginal_log_likelihood(beta_hat, se, X, edge_index)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    eb_model.eval()
    with torch.no_grad():
        post_mean, w1, w2, location = eb_model.posterior_mean(beta_hat, se, X, edge_index)
        result = EGNNFusedPosteriorResult(
            post_mean=post_mean.numpy(),
            prior_weights_1=w1,
            prior_weights_2=w2,
            location=location,
            model_param=eb_model.state_dict(),
            loss=loss.item()
        )
        return result
