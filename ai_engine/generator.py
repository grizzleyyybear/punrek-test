import os
import networkx as nx
from ai_engine.pso_optimizer import PSOOptimizer
import numpy as np
from typing import Dict, Tuple


class _NoOpLayoutGenerator:
    """Fallback layout generator used when AI model dependencies are unavailable."""

    def __init__(self, model):
        self.model = model


# Main class for generating PCB layouts.
class PCBGenerator:
    def __init__(self, model_path="pcb_gnn.pt"):
        self.model = None
        self.layout_generator = _NoOpLayoutGenerator(self.model)

        try:
            from ai_engine.model import PCBLayoutGenerator, load_model

            self.model = (
                load_model(model_path)
                if self._model_exists(model_path)
                else self._create_dummy_model()
            )
            self.layout_generator = PCBLayoutGenerator(self.model)
        except Exception:
            # Gracefully fallback when optional ML dependencies are unavailable
            # (for example, in lightweight serverless environments).
            self.model = None
            self.layout_generator = _NoOpLayoutGenerator(self.model)

        self.pso_optimizer = PSOOptimizer(
            num_particles=int(os.getenv("PSO_PARTICLES", "8")),
            max_iterations=int(os.getenv("PSO_ITERATIONS", "30")),
        )

    # Checks if the GNN model file exists.
    def _model_exists(self, path: str) -> bool:
        import os

        return os.path.exists(path)

    # Creates a dummy GNN model if the actual model is not available.
    def _create_dummy_model(self):
        from ai_engine.model import AdvancedPCBGNN

        return AdvancedPCBGNN(input_dim=5, hidden_dim=256, output_dim=128, num_layers=6)

    # Generates a PCB layout from a set of specifications.
    def generate_layout(self, spec: Dict) -> Tuple[Dict, Dict]:
        spec_vector = self._spec_to_vector(spec)
        num_components = spec.get("component_count", 10)
        initial_layout = self._generate_initial_layout_advanced(
            spec_vector, num_components
        )
        optimized_layout = self.pso_optimizer.optimize_layout(initial_layout, spec)
        pcb_graph_dict = self._graph_to_dict(optimized_layout)
        metrics = self._calculate_metrics(pcb_graph_dict, spec)
        return pcb_graph_dict, metrics

    # Converts the specification dictionary into a normalized feature vector.
    def _spec_to_vector(self, spec: Dict) -> list:
        vector = [
            spec.get("component_count", 10) / 100.0,
            spec.get("max_trace_length", 50.0) / 100.0,
            spec.get("layers", 2) / 4.0,
            len(spec.get("power_domains", [])) / 5.0,
            len(spec.get("signal_types", [])) / 10.0,
        ]
        return vector

    # Generates a basic initial layout.
    def _generate_initial_layout_advanced(
        self, spec_vector: list, num_components: int
    ) -> nx.Graph:
        graph = nx.Graph()
        grid_size = int(np.ceil(np.sqrt(num_components)))
        for i in range(num_components):
            row, col = i // grid_size, i % grid_size
            x, y = col * 10 + np.random.uniform(-1, 1), row * 10 + np.random.uniform(
                -1, 1
            )
            signal_types = spec_vector[3:]
            signal_type = "signal"
            if i % 3 == 0:
                signal_type = "power" if signal_types[0] > 0.1 else "signal"
            elif i % 3 == 1:
                signal_type = "ground" if signal_types[1] > 0.1 else "signal"
            graph.add_node(
                i, type="component", position=[x, y], signal_type=signal_type
            )
        for i in range(num_components - 1):
            if np.random.random() > 0.3:
                target = np.random.randint(0, num_components)
                if target != i:
                    dist = np.sqrt(
                        (
                            graph.nodes[i]["position"][0]
                            - graph.nodes[target]["position"][0]
                        )
                        ** 2
                        + (
                            graph.nodes[i]["position"][1]
                            - graph.nodes[target]["position"][1]
                        )
                        ** 2
                    )
                    graph.add_edge(i, target, length=dist, width=0.2, layer=0)
        return graph

    # Ensures the graph is fully connected.
    def _ensure_connectivity(self, graph: nx.Graph):
        if not nx.is_connected(graph):
            components = list(nx.connected_components(graph))
            for i in range(len(components) - 1):
                comp1_nodes, comp2_nodes = list(components[i]), list(components[i + 1])
                min_dist = float("inf")
                closest_pair = (None, None)
                for n1 in comp1_nodes:
                    for n2 in comp2_nodes:
                        pos1, pos2 = (
                            graph.nodes[n1]["position"],
                            graph.nodes[n2]["position"],
                        )
                        dist = np.sqrt(
                            (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
                        )
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (n1, n2)
                if closest_pair[0] is not None:
                    graph.add_edge(
                        closest_pair[0],
                        closest_pair[1],
                        length=min_dist,
                        width=0.2,
                        layer=0,
                    )

    # Converts a NetworkX graph into a dictionary format.
    def _graph_to_dict(self, graph: nx.Graph) -> Dict:
        return {
            "nodes": {
                str(node_id): {
                    "type": data.get("type", "component"),
                    "position": data.get("position", [0, 0]),
                    "signal_type": data.get("signal_type", "signal"),
                    "properties": {
                        k: v
                        for k, v in data.items()
                        if k not in ["type", "position", "signal_type"]
                    },
                }
                for node_id, data in graph.nodes(data=True)
            },
            "edges": {
                f"{src}-{tgt}": {
                    "source": str(src),
                    "target": str(tgt),
                    "length": data.get("length", 0),
                    "width": data.get("width", 0.2),
                    "layer": data.get("layer", 0),
                    "properties": {
                        k: v
                        for k, v in data.items()
                        if k not in ["length", "width", "layer"]
                    },
                }
                for src, tgt, data in graph.edges(data=True)
            },
        }

    # Calculates a variety of metrics to evaluate the quality of the generated layout.
    def _calculate_metrics(self, pcb_graph: Dict, spec: Dict) -> Dict:
        node_count, edge_count = len(pcb_graph["nodes"]), len(pcb_graph["edges"])
        total_trace_length = sum(edge["length"] for edge in pcb_graph["edges"].values())
        avg_trace_length = total_trace_length / edge_count if edge_count > 0 else 0
        max_trace_violations = sum(
            1
            for edge in pcb_graph["edges"].values()
            if edge["length"] > spec.get("max_trace_length", 50)
        )
        temp_graph = nx.Graph()
        for node_id in pcb_graph["nodes"]:
            temp_graph.add_node(node_id)
        for edge_data in pcb_graph["edges"].values():
            temp_graph.add_edge(edge_data["source"], edge_data["target"])
        is_connected = nx.is_connected(temp_graph) if node_count > 1 else True
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "total_trace_length": total_trace_length,
            "avg_trace_length": avg_trace_length,
            "max_trace_length": max(
                (edge["length"] for edge in pcb_graph["edges"].values()), default=0
            ),
            "max_trace_violations": max_trace_violations,
            "constraint_satisfaction_rate": 1.0
            - (max_trace_violations / edge_count if edge_count > 0 else 0),
            "component_density": (
                node_count / (spec.get("area", 100) or 100) if node_count > 0 else 0
            ),
            "connectivity": is_connected,
            "efficiency_score": (edge_count / node_count) if node_count > 0 else 0,
        }


# Example usage
if __name__ == "__main__":
    generator = PCBGenerator()
    test_spec = {
        "component_count": 8,
        "max_trace_length": 20.0,
        "layers": 2,
        "power_domains": ["3.3V", "5V"],
        "signal_types": ["digital", "analog"],
        "constraints": {"min_clearance": 0.2},
        "area": 100,
    }
    pcb_graph, metrics = generator.generate_layout(test_spec)
    print("Generated PCB layout:")
    print(f"Nodes: {len(pcb_graph['nodes'])}")
    print(f"Edges: {len(pcb_graph['edges'])}")
    print(f"Metrics: {metrics}")
