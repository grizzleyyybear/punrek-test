import networkx as nx
from typing import List, Dict, Tuple


class SecurityRules:
    def __init__(self):
        pass

    def check_short_circuits(self, graph: nx.Graph) -> List[Dict]:
        """
        Check for direct short circuits between power and ground
        """
        short_circuits = []

        # Find power and ground nodes
        power_nodes = [
            n
            for n, data in graph.nodes(data=True)
            if data.get("signal_type", "").lower() in ["power", "vcc", "vdd", "supply"]
        ]
        ground_nodes = [
            n
            for n, data in graph.nodes(data=True)
            if data.get("signal_type", "").lower() in ["ground", "gnd", "earth"]
        ]

        # Check for direct connections between power and ground
        for p_node in power_nodes:
            for g_node in ground_nodes:
                if graph.has_edge(p_node, g_node):
                    # Check if there's any protection (like resistors)
                    edge_data = graph[p_node][g_node]
                    width = edge_data.get("width", 0.2)

                    # Very wide traces might indicate intentional power connections
                    # but narrow ones could be problematic
                    if width < 0.5:  # Arbitrary threshold
                        short_circuits.append(
                            {
                                "nodes": [p_node, g_node],
                                "edge_data": edge_data,
                                "reason": "Direct connection between power and ground",
                            }
                        )

        return short_circuits

    def check_loops(self, graph: nx.Graph) -> List[List]:
        """
        Check for unintended electrical loops that could cause oscillation
        """
        loops = []

        # Find simple cycles (basic loops)
        try:
            cycles = list(nx.simple_cycles(graph.to_directed()))
            # Since we're dealing with undirected graphs conceptually,
            # we look for fundamental cycles
            if graph.number_of_nodes() > 0:
                try:
                    fundamental_cycles = nx.cycle_basis(graph)
                    for cycle in fundamental_cycles:
                        if len(cycle) > 2:  # Ignore simple bidirectional edges
                            loops.append(cycle)
                except:
                    pass
        except:
            # Handle disconnected graphs
            for component in nx.connected_components(graph):
                subgraph = graph.subgraph(component)
                if subgraph.number_of_edges() > subgraph.number_of_nodes():
                    try:
                        cycles = nx.cycle_basis(subgraph)
                        for cycle in cycles:
                            if len(cycle) > 2:
                                loops.append(cycle)
                    except:
                        continue

        return loops

    def check_antenna_traces(self, graph: nx.Graph) -> List:
        """
        Check for antenna traces (unterminated high-frequency traces)
        """
        antennas = []

        for node in graph.nodes():
            degree = graph.degree(node)
            node_data = graph.nodes[node]

            # Look for nodes that are only connected to one other node
            # and are of signal type (potential antenna)
            if degree == 1 and node_data.get("signal_type", "").lower() == "signal":
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    neighbor_data = graph.nodes[neighbors[0]]
                    # If connected to another signal node without proper termination
                    if neighbor_data.get("signal_type", "").lower() == "signal":
                        antennas.append(node)

        return antennas

    def check_floating_grounds(self, graph: nx.Graph) -> List:
        """
        Check for floating ground connections
        """
        floating_grounds = []

        ground_nodes = [
            n
            for n, data in graph.nodes(data=True)
            if data.get("signal_type", "").lower() in ["ground", "gnd"]
        ]

        for g_node in ground_nodes:
            # A ground node is floating if it's not connected to other ground nodes
            # or if it's isolated from the main ground plane
            neighbors = list(graph.neighbors(g_node))
            neighbor_grounds = [
                n
                for n in neighbors
                if graph.nodes[n].get("signal_type", "").lower() in ["ground", "gnd"]
            ]

            # If ground node has no other ground connections, it might be floating
            if len(neighbor_grounds) == 0 and len(neighbors) > 0:
                floating_grounds.append(g_node)

        return floating_grounds

    def check_signal_integrity(self, graph: nx.Graph) -> List[Dict]:
        """
        Check for potential signal integrity issues
        """
        issues = []

        for node1, node2, data in graph.edges(data=True):
            # Check trace width for current carrying capacity
            width = data.get("width", 0.2)
            signal_type = graph.nodes[node1].get("signal_type", "signal")

            if signal_type.lower() in ["power", "vcc", "vdd"] and width < 0.5:
                issues.append(
                    {
                        "type": "narrow_power_trace",
                        "nodes": [node1, node2],
                        "width": width,
                        "recommended_width": 0.5,
                        "description": f"Power trace too narrow ({width}mm), recommend >= 0.5mm",
                    }
                )

            # Check trace length for high-speed signals
            length = data.get("length", 0)
            if (
                signal_type.lower() == "high_speed" and length > 25
            ):  # arbitrary threshold
                issues.append(
                    {
                        "type": "long_high_speed_trace",
                        "nodes": [node1, node2],
                        "length": length,
                        "description": f"High-speed trace too long ({length}mm), may cause signal degradation",
                    }
                )

        return issues

    def check_thermal_issues(self, graph: nx.Graph) -> List[Dict]:
        """
        Check for potential thermal issues
        """
        issues = []

        # Check for dense component placement
        positions = [
            (n, data["position"])
            for n, data in graph.nodes(data=True)
            if data.get("type") == "component"
        ]

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                node1, pos1 = positions[i]
                node2, pos2 = positions[j]

                dist = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
                if dist < 2.0:  # Less than 2mm apart
                    issues.append(
                        {
                            "type": "thermal_crowding",
                            "nodes": [node1, node2],
                            "distance": dist,
                            "description": f"Components too close ({dist:.2f}mm), potential thermal issues",
                        }
                    )

        return issues

    def run_all_checks(self, graph: nx.Graph) -> Dict:
        """
        Run all security checks and return comprehensive results
        """
        results = {
            "short_circuits": self.check_short_circuits(graph),
            "loops": self.check_loops(graph),
            "antenna_traces": self.check_antenna_traces(graph),
            "floating_grounds": self.check_floating_grounds(graph),
            "signal_integrity": self.check_signal_integrity(graph),
            "thermal_issues": self.check_thermal_issues(graph),
        }

        return results


if __name__ == "__main__":
    # Test the security rules
    import networkx as nx

    # Create a test graph
    g = nx.Graph()
    g.add_node(0, signal_type="power", position=[0, 0])
    g.add_node(1, signal_type="ground", position=[10, 0])
    g.add_node(2, signal_type="signal", position=[5, 5])
    g.add_node(3, signal_type="signal", position=[15, 5])

    g.add_edge(0, 1, width=0.8, length=10.0)  # Wide power-ground connection
    g.add_edge(1, 2, width=0.2, length=7.07)
    g.add_edge(2, 3, width=0.2, length=11.18)

    rules = SecurityRules()
    results = rules.run_all_checks(g)

    print("Security Rule Results:")
    for check, findings in results.items():
        if findings:
            print(f"{check}: {findings}")
        else:
            print(f"{check}: OK")
