import os
import shutil
from pathlib import Path
import requests
import zipfile
import tarfile
import sys
import inspect

# Add the parent directory to the path so we can import from ai_engine
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from ai_engine.dataset_parser import KiCadParser
import networkx as nx


def collect_sample_dataset(output_dir="datasets/sample_pcb_boards"):
    """
    Collect a sample dataset of PCB boards for training
    In a real implementation, this would download from public repositories
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create sample PCB files for demonstration
    sample_files = [
        "sample_board_1.kicad_pcb",
        "sample_board_2.kicad_pcb",
        "sample_board_3.kicad_pcb",
    ]

    for filename in sample_files:
        filepath = os.path.join(output_dir, filename)
        create_sample_kicad_file(filepath)

    print(f"Created sample dataset in {output_dir}")
    return output_dir


def create_sample_kicad_file(filepath):
    """
    Create a sample KiCad PCB file for testing with proper format for parser
    """
    content = """(kicad_pcb (version 20211014) (generator pcbnew)
  (general
    (thickness 1.6)
  )

  (paper "A4")
  
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (40 "Dwgs.User" user "User.Drawings")
    (41 "Cmts.User" user "User.Comments")
    (42 "Eco1.User" user "User.Eco1")
    (43 "Eco2.User" user "User.Eco2")
    (44 "Edge.Cuts" user)
    (45 "Margin" user)
    (46 "B.CrtYd" user "B.Courtyard")
    (47 "F.CrtYd" user "F.Courtyard")
    (48 "B.Fab" user)
    (49 "F.Fab" user)
  )

  (setup
    (pad_to_mask_clearance 0.0)
  )

  (net 0 "")
  (net 1 "VCC")
  (net 2 "GND")
  (net 3 "SIG1")
  (net 4 "SIG2")

  (footprint "Resistor_SMD:R_0805_2012Metric" (layer "F.Cu") (at 0 0))
  (footprint "Capacitor_SMD:C_0805_2012Metric" (layer "F.Cu") (at 10 0))
  (footprint "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm" (layer "F.Cu") (at 20 0))

  (segment
    (start 0 0)
    (end 10 0)
    (width 0.2)
    (layer "F.Cu")
    (net 3)
  )
  (segment
    (start 10 0)
    (end 20 0)
    (width 0.2)
    (layer "F.Cu")
    (net 3)
  )
  (segment
    (start 20 0)
    (end 30 0)
    (width 0.2)
    (layer "F.Cu")
    (net 4)
  )
)
"""

    with open(filepath, "w") as f:
        f.write(content)


def validate_dataset(dataset_path):
    """
    Validate the collected dataset
    """
    parser = KiCadParser()
    valid_files = []
    invalid_files = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".kicad_pcb"):
            filepath = os.path.join(dataset_path, filename)
            try:
                graph = parser.parse_kicad_file(filepath)
                if len(graph.nodes()) > 0 and len(graph.edges()) > 0:
                    valid_files.append(filename)
                else:
                    print(
                        f"File {filename} parsed but has {len(graph.nodes())} nodes and {len(graph.edges())} edges"
                    )
                    invalid_files.append(filename)
            except Exception as e:
                print(f"Error parsing {filename}: {str(e)}")
                invalid_files.append(filename)

    print(f"Dataset validation complete:")
    print(f"Valid files: {len(valid_files)}")
    print(f"Invalid files: {len(invalid_files)}")

    return valid_files, invalid_files


def analyze_dataset(dataset_path):
    """
    Analyze the dataset statistics
    """
    parser = KiCadParser()
    stats = {
        "total_files": 0,
        "total_components": 0,
        "total_connections": 0,
        "avg_components_per_board": 0,
        "avg_connections_per_board": 0,
    }

    for filename in os.listdir(dataset_path):
        if filename.endswith(".kicad_pcb"):
            filepath = os.path.join(dataset_path, filename)
            try:
                graph = parser.parse_kicad_file(filepath)
                stats["total_files"] += 1
                stats["total_components"] += len(
                    [
                        n
                        for n, d in graph.nodes(data=True)
                        if d.get("type") == "component"
                    ]
                )
                stats["total_connections"] += len(graph.edges())
            except:
                continue

    if stats["total_files"] > 0:
        stats["avg_components_per_board"] = (
            stats["total_components"] / stats["total_files"]
        )
        stats["avg_connections_per_board"] = (
            stats["total_connections"] / stats["total_files"]
        )

    print("Dataset Analysis:")
    print(f"Total PCB files: {stats['total_files']}")
    print(f"Total components: {stats['total_components']}")
    print(f"Total connections: {stats['total_connections']}")
    print(f"Avg components per board: {stats['avg_components_per_board']:.2f}")
    print(f"Avg connections per board: {stats['avg_connections_per_board']:.2f}")

    return stats


if __name__ == "__main__":
    print("Collecting sample PCB dataset...")
    dataset_path = collect_sample_dataset()

    print("\nValidating dataset...")
    valid_files, invalid_files = validate_dataset(dataset_path)

    print("\nAnalyzing dataset...")
    stats = analyze_dataset(dataset_path)
