import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, TransformerConv
from torch_geometric.data import Data
import numpy as np
import os


# Advanced GNN for PCB layout generation.
class AdvancedPCBGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, output_dim=128, num_layers=6):
        super(AdvancedPCBGNN, self).__init__()
        self.input_dim, self.hidden_dim, self.output_dim, self.num_layers = (
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.attention_layers = nn.ModuleList(
            [
                GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1)
                for _ in range(num_layers // 2)
            ]
        )
        self.conv_layers = nn.ModuleList(
            [
                TransformerConv(hidden_dim, hidden_dim // 4, heads=4, dropout=0.1)
                for _ in range(num_layers // 2)
            ]
        )
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.position_refiner = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),
        )
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid(),
        )
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
        )
        self._init_weights()

    # Initializes model weights.
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # Forward pass of the GNN.
    def forward(self, x, edge_index, batch=None, positions=None):
        h = self.node_encoder(x)
        for attn_layer in self.attention_layers:
            h = F.relu(h + attn_layer(h, edge_index))
        for conv_layer in self.conv_layers:
            h = F.relu(h + conv_layer(h, edge_index))
        global_ctx = (
            global_mean_pool(h, batch)
            if batch is not None
            else torch.mean(h, dim=0, keepdim=True)
        )
        h_combined = torch.cat(
            [h, global_ctx[batch if batch is not None else 0].expand(h.size(0), -1)],
            dim=1,
        )
        return {
            "node_embeddings": h,
            "edge_predictions": self._predict_edges(h_combined, edge_index, positions),
            "position_deltas": self._refine_positions(h, positions),
            "reconstructed_features": self.feature_reconstructor(h),
        }

    # Predicts edge probability between all node pairs.
    def _predict_edges(self, h, edge_index, positions):
        num_nodes = h.size(0)
        row = (
            torch.arange(num_nodes, device=h.device)
            .view(-1, 1)
            .repeat(1, num_nodes)
            .view(-1)
        )
        col = (
            torch.arange(num_nodes, device=h.device)
            .view(1, -1)
            .repeat(num_nodes, 1)
            .view(-1)
        )
        mask = row != col
        row, col = row[mask], col[mask]
        src_h, tgt_h = h[row], h[col]
        rel_pos = (
            positions[col] - positions[row]
            if positions is not None
            and row.max() < positions.size(0)
            and col.max() < positions.size(0)
            else torch.zeros(src_h.size(0), 2, device=h.device)
        )
        edge_features = torch.cat(
            [src_h[:, : self.hidden_dim], tgt_h[:, : self.hidden_dim], rel_pos], dim=1
        )
        return torch.stack([row, col]), self.edge_predictor(edge_features).squeeze(-1)

    # Refines node positions based on their embeddings.
    def _refine_positions(self, h, positions):
        combined_features = torch.cat(
            [
                h[:, : self.hidden_dim],
                (
                    positions
                    if positions is not None
                    else torch.zeros(h.size(0), 2, device=h.device)
                ),
            ],
            dim=1,
        )
        return self.position_refiner(combined_features)


# Generates and refines a PCB layout using the GNN.
class PCBLayoutGenerator(nn.Module):
    def __init__(self, gnn_model, refinement_steps=5):
        super(PCBLayoutGenerator, self).__init__()
        self.gnn = gnn_model
        self.refinement_steps = refinement_steps
        self.layout_refiner = nn.Sequential(
            nn.Linear(256 + 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

    # Generates and refines a PCB layout.
    def forward(self, spec_vector, num_nodes, initial_positions=None):
        spec_tensor = (
            torch.tensor(spec_vector, dtype=torch.float)
            .unsqueeze(0)
            .expand(num_nodes, -1)
        )
        x = (
            torch.cat(
                [
                    torch.randn(num_nodes, self.gnn.input_dim - len(spec_vector)),
                    spec_tensor,
                ],
                dim=1,
            )
            if self.gnn.input_dim > len(spec_vector)
            else spec_tensor
        )
        edge_index = (
            torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
            if num_nodes > 1
            else torch.empty((2, 0), dtype=torch.long)
        )
        edge_index = (
            torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
            if num_nodes > 1
            else edge_index
        )
        positions = (
            initial_positions.clone()
            if initial_positions is not None
            else torch.randn(num_nodes, 2) * 10
        )
        for _ in range(self.refinement_steps):
            outputs = self.gnn(x, edge_index, positions=positions)
            positions += (
                self.layout_refiner(
                    torch.cat([outputs["node_embeddings"], positions], dim=1)
                )
                * 0.5
            )
            positions = torch.clamp(positions, min=-50, max=50)
        outputs = self.gnn(x, edge_index, positions=positions)
        return {"final_positions": positions, **outputs}


# Saves a trained model to a file.
def save_model(model, filepath):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "output_dim": model.output_dim,
                "num_layers": model.num_layers,
            },
        },
        filepath,
    )


# Loads a trained model from a file.
def load_model(filepath):
    if not os.path.exists(filepath):
        return AdvancedPCBGNN(input_dim=5, hidden_dim=256, output_dim=128, num_layers=6)
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    config = checkpoint["model_config"]
    model = AdvancedPCBGNN(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
