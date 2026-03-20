import hashlib
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN_AST_Analyzer(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(GNN_AST_Analyzer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

class TheProphet:
    def __init__(self, in_channels: int = 16, hidden_channels: int = 32):
        self.model = GNN_AST_Analyzer(in_channels, hidden_channels)
        self.model.eval()
        self.in_channels = in_channels
        
    def generate_attention_mask(self, G: nx.DiGraph, k: int = 5) -> str:
        node_mapping = {n: i for i, n in enumerate(G.nodes)}
        
        x_features = torch.zeros((len(G.nodes), self.in_channels), dtype=torch.float)
        for node, data in G.nodes(data=True):
            idx = node_mapping[node]
            h = int(hashlib.md5(data['type'].encode('utf-8')).hexdigest()[:8], 16)
            for i in range(self.in_channels):
                x_features[idx, i] = float((h >> i) & 1)

        edges = [[node_mapping[u], node_mapping[v]] for u, v in G.edges]
        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        with torch.no_grad():
            scores = self.model(x_features, edge_index).squeeze(-1)

        k_eff = min(k, len(scores))
        if k_eff == 0:
            return "No anomalies detected."
            
        top_k_indices = torch.topk(scores, k_eff).indices
        
        reverse_mapping = {v: k for k, v in node_mapping.items()}
        anomalies = []
        for idx in top_k_indices.tolist():
            orig_node = reverse_mapping[idx]
            data = G.nodes[orig_node]
            if data.get('lineno', -1) > 0:
                anomalies.append(f"Line {data['lineno']} | Type: {data['type']} | Anomaly Score: {scores[idx].item():.4f}")
                
        return "\n".join(anomalies)
