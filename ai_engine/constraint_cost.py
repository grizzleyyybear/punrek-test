import numpy as np
from typing import Dict, List
import networkx as nx


# Calculates the total constraint violation cost for a PCB layout.
def calculate_constraint_cost(graph: nx.Graph, spec: Dict = None) -> float:
    if spec is None:
        spec = {}
    costs = {}
    costs["trace_length"] = calculate_trace_length_cost(
        graph, spec.get("max_trace_length", 50.0)
    )
    costs["crossings"] = calculate_crossing_cost(graph)
    costs["floating_nodes"] = calculate_floating_nodes_cost(graph)
    costs["power_efficiency"] = calculate_power_path_cost(graph)
    total_cost = sum(costs.values())
    return total_cost, costs


# Calculates cost for trace lengths exceeding the maximum limit.
def calculate_trace_length_cost(graph: nx.Graph, max_length: float) -> float:
    cost = 0.0
    for _, _, data in graph.edges(data=True):
        length = data.get("length", 0)
        if length > max_length:
            cost += (length - max_length) ** 2
    return cost


# Approximates the cost associated with trace crossings.
def calculate_crossing_cost(graph: nx.Graph) -> float:
    edge_count = len(graph.edges())
    node_count = len(graph.nodes())
    if node_count > 1:
        expected_crossings = (edge_count**2) / (node_count * 2)
        return expected_crossings * 10.0
    return 0.0


# Calculates the cost for nodes with no connections.
def calculate_floating_nodes_cost(graph: nx.Graph) -> float:
    floating_count = sum(1 for node in graph.nodes() if graph.degree(node) == 0)
    return floating_count * 100.0


# Calculates the cost for inefficient power distribution paths.
def calculate_power_path_cost(graph: nx.Graph) -> float:
    cost = 0.0
    power_nodes = [
        n
        for n, data in graph.nodes(data=True)
        if data.get("signal_type", "").lower() in ["power", "vcc", "vdd"]
    ]
    if len(power_nodes) < 2:
        return 0.0
    subgraph = graph.subgraph(power_nodes)
    if not nx.is_connected(subgraph):
        components = list(nx.connected_components(subgraph))
        cost += len(components) * 50.0
    if len(power_nodes) > 1:
        try:
            avg_path_length = nx.average_shortest_path_length(subgraph)
            cost += avg_path_length * 5.0
        except nx.NetworkXError:
            pass
    return cost


# Calculates security-related costs, such as short circuits.
def calculate_security_cost(graph: nx.Graph) -> float:
    cost = 0.0
    power_nodes = [
        n
        for n, data in graph.nodes(data=True)
        if data.get("signal_type", "").lower() in ["power", "vcc", "vdd"]
    ]
    ground_nodes = [
        n
        for n, data in graph.nodes(data=True)
        if data.get("signal_type", "").lower() in ["ground", "gnd"]
    ]
    for p_node in power_nodes:
        for g_node in ground_nodes:
            if graph.has_edge(p_node, g_node):
                cost += 1000.0
    return cost


# Validates that the generated graph meets all specified design constraints.
def validate_constraints(graph: nx.Graph, spec: Dict) -> Dict[str, bool]:
    results = {}
    max_length = spec.get("max_trace_length", 50.0)
    results["max_trace_length"] = all(
        data.get("length", 0) <= max_length for _, _, data in graph.edges(data=True)
    )
    results["no_floating_nodes"] = all(graph.degree(node) > 0 for node in graph.nodes())
    results["is_connected"] = nx.is_connected(graph) if len(graph.nodes()) > 1 else True
    max_layers = spec.get("layers", 2)
    used_layers = {data.get("layer", 0) for _, _, data in graph.edges(data=True)}
    results["layer_usage"] = len(used_layers) <= max_layers
    return results


# Example usage
if __name__ == "__main__":
    g = nx.Graph()
    g.add_node(0, signal_type="power", position=[0, 0])
    g.add_node(1, signal_type="ground", position=[10, 0])
    g.add_node(2, signal_type="signal", position=[5, 5])
    g.add_edge(0, 2, length=5.0, width=0.2, layer=0)
    g.add_edge(1, 2, length=8.0, width=0.2, layer=0)
    spec = {"max_trace_length": 10.0, "layers": 2}
    total_cost, costs = calculate_constraint_cost(g, spec)
    print(f"Total constraint cost: {total_cost}")
    print(f"Individual costs: {costs}")
    validation_results = validate_constraints(g, spec)
    print(f"Validation results: {validation_results}")
