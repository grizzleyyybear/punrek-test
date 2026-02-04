import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import networkx as nx
from torch_geometric.data import Data
from model import AdvancedPCBGNN, PCBLayoutGenerator
from dataset_parser import load_dataset
from graph_utils import graph_to_pyg_data
import numpy as np
from tqdm import tqdm
import os


# Custom PyTorch Dataset for loading and processing PCB graph data.
class PCBGraphDataset(Dataset):
    def __init__(self, graphs):
        self.pyg_data_list = [
            graph_to_pyg_data(g)
            for g in graphs
            if "position" in next(iter(g.nodes(data=True)))
        ]

    def __len__(self):
        return len(self.pyg_data_list)

    def __getitem__(self, idx):
        return self.pyg_data_list[idx]


# Composite loss function for training the PCB GNN.
class AdvancedPCBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta

    def forward(self, outputs, data):
        loss = 0.0
        if "reconstructed_features" in outputs and hasattr(data, "x"):
            loss += self.alpha * self.mse_loss(
                outputs["reconstructed_features"], data.x
            )
        if "position_deltas" in outputs and hasattr(data, "pos"):
            loss += self.beta * self.mse_loss(
                data.pos + outputs["position_deltas"], data.pos
            )
        if "edge_predictions" in outputs and hasattr(data, "edge_index"):
            edge_idx, edge_probs = outputs["edge_predictions"]
            gt_edges = {
                (min(src, tgt), max(src, tgt))
                for src, tgt in data.edge_index.t().tolist()
            }
            targets = torch.tensor(
                [
                    (
                        1.0
                        if (min(int(s), int(t)), max(int(s), int(t))) in gt_edges
                        else 0.0
                    )
                    for s, t in edge_idx.t().tolist()
                ],
                device=edge_probs.device,
            )
            loss += self.gamma * self.bce_loss(edge_probs, targets)
        if "node_embeddings" in outputs:
            loss += self.delta * torch.mean(torch.abs(outputs["node_embeddings"]))
        return loss


# Main function for training the GNN model.
def train_gnn_model(
    dataset_path,
    model_save_path="pcb_gnn.pt",
    epochs=200,
    batch_size=2,
    lr=0.001,
    validate_every=10,
):
    print("Loading PCB dataset...")
    graphs = load_dataset(dataset_path) or create_synthetic_dataset(50)
    print(f"Loaded {len(graphs)} PCB layouts.")
    dataset = PCBGraphDataset(graphs)
    if not dataset:
        raise ValueError("Dataset is empty after processing.")
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    model = AdvancedPCBGNN(
        input_dim=dataset[0].x.shape[1], hidden_dim=256, output_dim=128, num_layers=6
    )
    criterion = AdvancedPCBLoss(alpha=1.0, beta=1.0, gamma=2.0, delta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=10
    )
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_train_loss = sum(
            run_epoch(model, loader, criterion, optimizer, is_training=True)
            for loader in [train_loader]
        )
        avg_train_loss = total_train_loss / len(train_loader)
        if epoch % validate_every == 0 or epoch == epochs - 1:
            model.eval()
            total_val_loss = sum(
                run_epoch(model, loader, criterion) for loader in [val_loader]
            )
            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            print(
                f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_model(
                    model,
                    optimizer,
                    epoch,
                    avg_train_loss,
                    avg_val_loss,
                    model_save_path.replace(".pt", "_best.pt"),
                )
        else:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
    save_model(model, optimizer, epoch, avg_train_loss, None, model_save_path)
    print(f"Final model saved to {model_save_path}")
    return model


# Runs a single epoch of training or validation.
def run_epoch(model, loader, criterion, optimizer=None, is_training=False):
    total_loss = 0
    for batch in tqdm(loader, desc="Training" if is_training else "Validation"):
        if is_training:
            optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, batch.batch, batch.pos)
        loss = criterion(outputs, batch)
        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
    return total_loss


# Saves the model state to a file.
def save_model(model, optimizer, epoch, train_loss, val_loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_config": {
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "output_dim": model.output_dim,
                "num_layers": model.num_layers,
            },
        },
        path,
    )
    print(f"Model saved to {path}")


# Creates a synthetic dataset of PCB-like graphs for demonstration purposes.
def create_synthetic_dataset(size=50):
    return [nx.erdos_renyi_graph(np.random.randint(5, 15), p=0.3) for _ in range(size)]


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Advanced PCB Graph Neural Network"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="datasets/", help="Path to PCB dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="pcb_gnn.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    args = parser.parse_args()
    print("Starting Advanced GNN training...")
    train_gnn_model(
        dataset_path=args.dataset_path,
        model_save_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
    )
    print("Training completed!")
