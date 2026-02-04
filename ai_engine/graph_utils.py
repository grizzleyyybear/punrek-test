import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Tuple


# Converts a NetworkX graph to a PyTorch Geometric Data object.
def graph_to_pyg_data(graph: nx.Graph) -> Data:
    node_features = []
    node_mapping = {
        node_id: i for i, (node_id, data) in enumerate(graph.nodes(data=True))
    }
    for node_id, data in graph.nodes(data=True):
        signal_type_onehot = [0] * 5
        signal_type = data.get("signal_type", "signal").lower()
        if "power" in signal_type:
            signal_type_onehot[1] = 1
        elif "ground" in signal_type or "gnd" in signal_type:
            signal_type_onehot[2] = 1
        elif "clock" in signal_type:
            signal_type_onehot[3] = 1
        else:
            signal_type_onehot[0] = 1
        features = [
            data.get("type", "component") == "component",
            data.get("position", [0, 0])[0],
            data.get("position", [0, 0])[1],
        ] + signal_type_onehot
        node_features.append(features)
    x = torch.tensor(node_features, dtype=torch.float)
    edge_indices, edge_attrs = [], []
    for source, target, data in graph.edges(data=True):
        src_idx, tgt_idx = node_mapping[source], node_mapping[target]
        edge_indices.extend([[src_idx, tgt_idx], [tgt_idx, src_idx]])
        edge_attr = [
            data.get("length", 0),
            data.get("width", 0.1),
            data.get("layer", 0),
            1 if data.get("critical", False) else 0,
        ]
        edge_attrs.extend([edge_attr, edge_attr])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    pos = torch.tensor(
        [data.get("position", [0, 0]) for _, data in graph.nodes(data=True)],
        dtype=torch.float,
    )
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


# Converts a PyTorch Geometric Data object back into a NetworkX graph.
def pyg_data_to_graph(data: Data) -> nx.Graph:
    graph = nx.Graph()
    signal_type_map = ["signal", "power", "ground", "clock", "other"]
    for i in range(data.x.size(0)):
        features = data.x[i].tolist()
        signal_type_idx = np.argmax(features[3:8])
        signal_type = (
            signal_type_map[signal_type_idx]
            if signal_type_idx < len(signal_type_map)
            else "signal"
        )
        graph.add_node(
            i,
            type="component" if features[0] > 0.5 else "pad",
            position=[features[1], features[2]],
            signal_type=signal_type,
        )
    added_edges = set()
    for i in range(data.edge_index.size(1)):
        src, tgt = int(data.edge_index[0, i]), int(data.edge_index[1, i])
        edge_key = tuple(sorted([src, tgt]))
        if edge_key in added_edges:
            continue
        added_edges.add(edge_key)
        attrs = (
            data.edge_attr[i].tolist() if hasattr(data, "edge_attr") else [0, 0.1, 0, 0]
        )
        graph.add_edge(
            src,
            tgt,
            length=attrs[0],
            width=attrs[1],
            layer=attrs[2],
            critical=attrs[3] > 0.5,
        )
    return graph


# Calculates a variety of metrics for a given PCB graph.
def get_graph_metrics(graph: nx.Graph) -> Dict:
    metrics = {
        "node_count": len(graph.nodes()),
        "edge_count": len(graph.edges()),
        "is_connected": nx.is_connected(graph),
        "diameter": (
            nx.diameter(graph)
            if nx.is_connected(graph) and len(graph.nodes()) > 1
            else float("inf")
        ),
        "avg_trace_length": (
            sum(data.get("length", 0) for _, _, data in graph.edges(data=True))
            / len(graph.edges())
            if graph.edges()
            else 0
        ),
        "component_counts": {
            t: sum(1 for _, data in graph.nodes(data=True) if data.get("type") == t)
            for t in set(data.get("type") for _, data in graph.nodes(data=True))
        },
        "signal_type_distribution": {
            t: sum(
                1 for _, data in graph.nodes(data=True) if data.get("signal_type") == t
            )
            for t in set(data.get("signal_type") for _, data in graph.nodes(data=True))
        },
        "density": nx.density(graph),
    }
    return metrics


# Normalizes the coordinates of the graph's nodes to fit within a standard range.
def normalize_graph(graph: nx.Graph) -> nx.Graph:
    positions = np.array(
        [data["position"] for _, data in graph.nodes(data=True) if "position" in data]
    )
    if not positions.any():
        return graph
    min_pos, max_pos = positions.min(axis=0), positions.max(axis=0)
    range_pos = max_pos - min_pos
    range_pos[range_pos == 0] = 1
    normalized_positions = (positions - min_pos) / range_pos * 100
    new_graph = graph.copy()
    node_ids = [
        node_id for node_id, data in graph.nodes(data=True) if "position" in data
    ]
    for i, node_id in enumerate(node_ids):
        new_graph.nodes[node_id]["position"] = normalized_positions[i].tolist()
    return new_graph


# Validates that the graph represents a valid PCB layout.
def validate_pcb_graph(graph: nx.Graph) -> Tuple[bool, List[str]]:
    issues = []
    if any(graph.degree(node) == 0 for node in graph.nodes()):
        issues.append(
            f"Found floating nodes: {[node for node in graph.nodes() if graph.degree(node) == 0]}"
        )
    if any(u == v for u, v in graph.edges()):
        issues.append(f"Found self-loops: {list(nx.selfloop_edges(graph))}")
    for u, v, data in graph.edges(data=True):
        if data.get("length", 0) < 0:
            issues.append(f"Negative length on edge {u}-{v}: {data['length']}")
        if data.get("width", 0) <= 0:
            issues.append(f"Non-positive width on edge {u}-{v}: {data['width']}")
    return not issues, issues


# Adds additional computed features to the graph's nodes and edges.
def enrich_graph_features(graph: nx.Graph) -> nx.Graph:
    enriched_graph = graph.copy()
    is_connected = nx.is_connected(graph)
    if is_connected:
        betweenness = nx.betweenness_centrality(graph)
        closeness = nx.closeness_centrality(graph)
        eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
        for node in graph.nodes():
            enriched_graph.nodes[node]["betweenness_centrality"] = betweenness.get(
                node, 0
            )
            enriched_graph.nodes[node]["closeness_centrality"] = closeness.get(node, 0)
            enriched_graph.nodes[node]["eigenvector_centrality"] = eigenvector.get(
                node, 0
            )
    for u, v, data in enriched_graph.edges(data=True):
        pos_u, pos_v = enriched_graph.nodes[u].get(
            "position", [0, 0]
        ), enriched_graph.nodes[v].get("position", [0, 0])
        euclidean_dist = np.sqrt(
            (pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2
        )
        enriched_graph[u][v]["euclidean_length"] = euclidean_dist
        enriched_graph[u][v]["length_discrepancy"] = abs(
            data.get("length", euclidean_dist) - euclidean_dist
        )
    return enriched_graph
