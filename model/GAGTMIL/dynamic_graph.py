# models/ADGAMIL/dynamic_graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbors=5):
        super(DynamicGraphEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors

        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
  
        B, N, D = x.size()  
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)  
        S = torch.bmm(x_norm, x_norm.transpose(1, 2))  
        mask = torch.eye(N, device=x.device).bool().unsqueeze(0).expand(B, -1, -1)
        S.masked_fill_(mask, float('-inf'))
        values, indices = torch.topk(S, k=self.num_neighbors, dim=2)  # values: (B, N, num_neighbors), indices: (B, N, num_neighbors)
        weights = F.softmax(values, dim=2)  # (B, N, num_neighbors)
        neighbor_features = torch.stack([x[b, indices[b], :] for b in range(B)], dim=0)  # (B, N, num_neighbors, D)
        aggregate_neighbors = torch.sum(weights.unsqueeze(-1) * neighbor_features, dim=2)  # (B, N, D)
        h_combined = x + aggregate_neighbors  # (B, N, D)

         h_combined = h_combined.view(B * N, D)
        h_combined = F.relu(self.conv1(h_combined))  # (B*N, hidden_dim)
        h_combined = F.relu(self.conv2(h_combined))  # (B*N, hidden_dim)
        graph_features = h_combined.view(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        return graph_features
