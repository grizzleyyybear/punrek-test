import networkx as nx
from security.security_rules import SecurityRules
from typing import Dict, List, Tuple


class SecurityAnalyzer:
    def __init__(self):
        self.rules = SecurityRules()

    def analyze(self, pcb_graph: Dict) -> Tuple[List[Dict], float]:
        """
        Analyze the PCB graph for security vulnerabilities
        Returns: (list of vulnerabilities, overall security score)
        """
        # Convert dict to NetworkX graph
        graph = self._dict_to_graph(pcb_graph)

        # Run all security checks
        vulnerabilities = []

        # Short circuit detection
        short_circuits = self.rules.check_short_circuits(graph)
        for sc in short_circuits:
            vulnerabilities.append(
                {
                    "type": "short_circuit",
                    "severity": "critical",
                    "description": f'Short circuit detected between {sc["nodes"][0]} and {sc["nodes"][1]}',
                    "location": sc["nodes"],
                    "details": sc,
                }
            )

        # Loop detection
        loops = self.rules.check_loops(graph)
        for loop in loops:
            vulnerabilities.append(
                {
                    "type": "loop",
                    "severity": "high",
                    "description": f"Unintended loop detected: {loop}",
                    "location": loop,
                    "details": {"nodes_in_loop": loop},
                }
            )

        # Antenna trace detection
        antennas = self.rules.check_antenna_traces(graph)
        for antenna in antennas:
            vulnerabilities.append(
                {
                    "type": "antenna_trace",
                    "severity": "medium",
                    "description": f"Antenna trace detected at node {antenna}",
                    "location": [antenna],
                    "details": {"node": antenna},
                }
            )

        # Floating ground detection
        floating_grounds = self.rules.check_floating_grounds(graph)
        for fg in floating_grounds:
            vulnerabilities.append(
                {
                    "type": "floating_ground",
                    "severity": "high",
                    "description": f"Floating ground detected at node {fg}",
                    "location": [fg],
                    "details": {"node": fg},
                }
            )

        # Calculate security score (0-1 scale, 1 being completely secure)
        max_severity_scores = {"critical": 0.1, "high": 0.3, "medium": 0.6, "low": 0.8}

        if not vulnerabilities:
            score = 1.0
        else:
            severity_multipliers = [
                max_severity_scores[v["severity"]] for v in vulnerabilities
            ]
            score = (
                sum(severity_multipliers) / len(vulnerabilities)
                if severity_multipliers
                else 0.5
            )
            # Lower the score based on number of vulnerabilities
            score *= 1.0 - min(len(vulnerabilities) * 0.1, 0.5)  # Up to 50% reduction

        score = max(0.0, min(1.0, score))  # Clamp between 0 and 1

        return vulnerabilities, score

    def _dict_to_graph(self, pcb_graph: Dict) -> nx.Graph:
        """Convert PCB graph dictionary to NetworkX graph"""
        graph = nx.Graph()

        # Add nodes
        for node_id, node_data in pcb_graph["nodes"].items():
            graph.add_node(int(node_id), **node_data)

        # Add edges
        for edge_id, edge_data in pcb_graph["edges"].items():
            source = int(edge_data["source"])
            target = int(edge_data["target"])
            graph.add_edge(source, target, **edge_data)

        return graph

    def get_security_report(self, pcb_graph: Dict) -> Dict:
        """Generate a detailed security report"""
        vulnerabilities, score = self.analyze(pcb_graph)

        report = {
            "overall_score": score,
            "total_vulnerabilities": len(vulnerabilities),
            "vulnerability_summary": {
                "critical": len(
                    [v for v in vulnerabilities if v["severity"] == "critical"]
                ),
                "high": len([v for v in vulnerabilities if v["severity"] == "high"]),
                "medium": len(
                    [v for v in vulnerabilities if v["severity"] == "medium"]
                ),
                "low": len([v for v in vulnerabilities if v["severity"] == "low"]),
            },
            "vulnerabilities": vulnerabilities,
            "recommendations": self._generate_recommendations(vulnerabilities),
        }

        return report

    def _generate_recommendations(self, vulnerabilities: List[Dict]) -> List[str]:
        """Generate recommendations based on vulnerabilities found"""
        recommendations = []

        if any(v["type"] == "short_circuit" for v in vulnerabilities):
            recommendations.append("Add proper isolation between power and ground nets")

        if any(v["type"] == "loop" for v in vulnerabilities):
            recommendations.append("Review and remove unintended feedback loops")

        if any(v["type"] == "antenna_trace" for v in vulnerabilities):
            recommendations.append(
                "Connect antenna traces to appropriate termination points"
            )

        if any(v["type"] == "floating_ground" for v in vulnerabilities):
            recommendations.append(
                "Ensure all ground connections are properly tied to common ground"
            )

        if not recommendations:
            recommendations.append("No critical security vulnerabilities detected")

        return recommendations


if __name__ == "__main__":
    # Test the security analyzer
    analyzer = SecurityAnalyzer()

    # Create a test graph with potential vulnerabilities
    test_graph = {
        "nodes": {
            "0": {"type": "component", "signal_type": "power", "position": [0, 0]},
            "1": {"type": "component", "signal_type": "ground", "position": [10, 0]},
            "2": {"type": "component", "signal_type": "signal", "position": [5, 5]},
            "3": {"type": "pad", "signal_type": "signal", "position": [15, 5]},
        },
        "edges": {
            "0-1": {
                "source": "0",
                "target": "1",
                "length": 10.0,
                "width": 0.2,
                "layer": 0,
            },  # Potential short circuit
            "1-2": {
                "source": "1",
                "target": "2",
                "length": 7.07,
                "width": 0.2,
                "layer": 0,
            },
            "2-3": {
                "source": "2",
                "target": "3",
                "length": 11.18,
                "width": 0.2,
                "layer": 0,
            },
        },
    }

    vulnerabilities, score = analyzer.analyze(test_graph)
    print(f"Security Score: {score:.2f}")
    print(f"Vulnerabilities Found: {len(vulnerabilities)}")

    for v in vulnerabilities:
        print(f"- {v['type']}: {v['description']} [{v['severity']}]")
