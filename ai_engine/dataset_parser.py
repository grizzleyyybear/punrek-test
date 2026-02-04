import xml.etree.ElementTree as ET
import networkx as nx
import json
import re
from typing import Dict, List, Tuple
import numpy as np
import os


# Parses KiCad PCB files into a graph representation.
class KiCadParser:
    def __init__(self):
        self.graph = nx.Graph()

    # Parses a .kicad_pcb file and converts it into a graph.
    def parse_kicad_file(self, filepath: str) -> nx.Graph:
        self.graph = nx.Graph()
        with open(filepath, "r") as f:
            content = f.read()
        components = self._extract_components(content)
        tracks = self._extract_tracks(content)
        self._build_graph(components, tracks)
        return self.graph

    # Extracts component information from the KiCad file content.
    def _extract_components(self, content: str) -> List[Dict]:
        components = []
        comp_pattern = r'\(footprint\s+"([^"]+)"\s+\(layer\s+"[^"]+"\)\s+\(at\s+([-\d\.]+)\s+([-\d\.]+)(?:\s+[-\d\.]+)?\)'
        matches = re.findall(comp_pattern, content)
        if not matches:
            comp_pattern_alt = r'\(footprint\s+"([^"]+)"\s+\(at\s+([-\d\.]+)\s+([-\d\.]+)(?:\s+[-\d\.]+)?\)'
            matches = re.findall(comp_pattern_alt, content)
        for i, match in enumerate(matches):
            if len(match) >= 3:
                footprint, x, y = match[0], match[1], match[2]
                components.append(
                    {
                        "id": f"comp_{i}",
                        "footprint": footprint,
                        "position": [float(x), float(y)],
                        "type": "component",
                        "signal_type": "signal",
                    }
                )
        return components

    # Extracts track (segment) information from the KiCad file content.
    def _extract_tracks(self, content: str) -> List[Dict]:
        tracks = []
        track_pattern = r"\(segment\s+\(start\s+([-\d\.]+)\s+([-\d\.]+)\)\s+\(end\s+([-\d\.]+)\s+([-\d\.]+)\)\s+\(width\s+([-\d\.]+)\)"
        matches = re.findall(track_pattern, content)
        for start_x, start_y, end_x, end_y, width in matches:
            tracks.append(
                {
                    "start": [float(start_x), float(start_y)],
                    "end": [float(end_x), float(end_y)],
                    "width": float(width),
                    "length": self._calculate_distance(
                        float(start_x), float(start_y), float(end_x), float(end_y)
                    ),
                }
            )
        return tracks

    # Calculates the Euclidean distance between two points.
    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # Builds the graph from the parsed component and track data.
    def _build_graph(self, components: List[Dict], tracks: List[Dict]):
        for i, comp in enumerate(components):
            self.graph.add_node(f"comp_{i}", **comp)
        for i, track in enumerate(tracks):
            start_node = self._find_closest_component(track["start"])
            end_node = self._find_closest_component(track["end"])
            if start_node and end_node and start_node != end_node:
                self.graph.add_edge(
                    start_node,
                    end_node,
                    id=f"track_{i}",
                    length=track["length"],
                    width=track["width"],
                )

    # Finds the closest component to a given point in the layout.
    def _find_closest_component(self, point: List[float]) -> str:
        min_dist = float("inf")
        closest_comp = None
        for node_id, data in self.graph.nodes(data=True):
            if "position" in data:
                pos = data["position"]
                dist = self._calculate_distance(pos[0], pos[1], point[0], point[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_comp = node_id
        return closest_comp


# Loads a dataset of PCB graphs from a directory of .kicad_pcb files.
def load_dataset(dataset_path: str) -> List[nx.Graph]:
    graphs = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".kicad_pcb"):
            filepath = os.path.join(dataset_path, filename)
            parser = KiCadParser()
            graph = parser.parse_kicad_file(filepath)
            graphs.append(graph)
    return graphs


# Example usage
if __name__ == "__main__":
    parser = KiCadParser()
    # graph = parser.parse_kicad_file("sample.kicad_pcb")
    # if graph:
    #     print(f"Parsed graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
