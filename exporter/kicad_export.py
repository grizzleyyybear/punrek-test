import json
import os
from typing import Dict


class KiCadExporter:
    def __init__(self):
        pass

    def export_to_kicad(self, pcb_graph: Dict, filepath: str) -> bool:
        """
        Export PCB graph to KiCad format (.kicad_pcb)
        """
        try:
            with open(filepath, "w") as f:
                f.write(self._generate_kicad_content(pcb_graph))
            return True
        except Exception as e:
            print(f"Export failed: {str(e)}")
            return False

    def _generate_kicad_content(self, pcb_graph: Dict) -> str:
        """
        Generate KiCad PCB file content from graph
        """
        content = []
        content.append("(kicad_pcb (version 20211014) (generator pcbnew)")
        content.append("")

        # Add general section
        content.append("  (general")
        content.append("    (thickness 1.6)")
        content.append("  )")
        content.append("")

        # Add paper settings
        content.append('  (paper "A4")')
        content.append("")

        # Add layers
        content.append("  (layers")
        content.append('    (0 "F.Cu" signal)')
        content.append('    (31 "B.Cu" signal)')
        content.append('    (32 "B.Adhes" user "B.Adhesive")')
        content.append('    (33 "F.Adhes" user "F.Adhesive")')
        content.append('    (34 "B.Paste" user)')
        content.append('    (35 "F.Paste" user)')
        content.append('    (36 "B.SilkS" user "B.Silkscreen")')
        content.append('    (37 "F.SilkS" user "F.Silkscreen")')
        content.append('    (38 "B.Mask" user)')
        content.append('    (39 "F.Mask" user)')
        content.append('    (40 "Dwgs.User" user "User.Drawings")')
        content.append('    (41 "Cmts.User" user "User.Comments")')
        content.append('    (42 "Eco1.User" user "User.Eco1")')
        content.append('    (43 "Eco2.User" user "User.Eco2")')
        content.append('    (44 "Edge.Cuts" user)')
        content.append('    (45 "Margin" user)')
        content.append('    (46 "B.CrtYd" user "B.Courtyard")')
        content.append('    (47 "F.CrtYd" user "F.Courtyard")')
        content.append('    (48 "B.Fab" user)')
        content.append('    (49 "F.Fab" user)')
        content.append("  )")
        content.append("")

        # Add setup
        content.append("  (setup")
        content.append("    (pad_to_mask_clearance 0.0)")
        content.append("    (pcbplotparams")
        content.append("      (layerselection 0x010fc_ffffffff)")
        content.append("      (plot_on_all_layers_selection 0x00000_000000)")
        content.append("      (disableapertmacros false)")
        content.append("      (usegerberextensions false)")
        content.append("      (usegerberattributes true)")
        content.append("      (usegerberadvancedattributes true)")
        content.append("      (creategerberjobfile true)")
        content.append("      (svguseinch false)")
        content.append("      (svgprecision 6)")
        content.append("      (excludeedgelayer true)")
        content.append("      (plotframeref false)")
        content.append("      (viasonmask false)")
        content.append("      (mode 1)")
        content.append("      (useauxorigin false)")
        content.append("      (hpglpennumber 1)")
        content.append("      (hpglpenspeed 20)")
        content.append("      (hpglpendiameter 15.00)")
        content.append("      (psnegative false)")
        content.append("      (psa4output false)")
        content.append("      (plotreference true)")
        content.append("      (plotvalue true)")
        content.append("      (plotinvisibletext false)")
        content.append("      (sketchpadsonfab false)")
        content.append("      (subtractmaskfromsilk false)")
        content.append("      (outputformat 1)")
        content.append("      (mirror false)")
        content.append("      (drillshape 1)")
        content.append("      (scaleselection 1)")
        content.append('      (outputdirectory "")')
        content.append("    )")
        content.append("  )")
        content.append("")

        # Add nets
        content.append('  (net 0 "")')
        net_counter = 1
        for edge_id, edge_data in pcb_graph["edges"].items():
            content.append(f'  (net {net_counter} "Net-(N{net_counter})")')
            net_counter += 1
        content.append("")

        # Add components (simplified)
        for node_id, node_data in pcb_graph["nodes"].items():
            x, y = node_data["position"]
            component_type = node_data.get("type", "component")
            content.append(
                f'  (footprint "{component_type}_{node_id}" (layer "F.Cu") (at {x} {y}))'
            )
        content.append("")

        # Add tracks (edges)
        for edge_id, edge_data in pcb_graph["edges"].items():
            source = int(edge_data["source"])
            target = int(edge_data["target"])
            source_pos = pcb_graph["nodes"][str(source)]["position"]
            target_pos = pcb_graph["nodes"][str(target)]["position"]
            width = edge_data.get("width", 0.2)
            layer = "F.Cu" if edge_data.get("layer", 0) == 0 else "B.Cu"

            content.append("  (segment")
            content.append(f"    (start {source_pos[0]} {source_pos[1]})")
            content.append(f"    (end {target_pos[0]} {target_pos[1]})")
            content.append(f"    (width {width})")
            content.append(f'    (layer "{layer}")')
            content.append("  )")

        content.append("")
        content.append(")")

        return "\n".join(content)

    def validate_kicad_export(self, filepath: str) -> bool:
        """
        Validate that the exported file is a valid KiCad file
        """
        try:
            with open(filepath, "r") as f:
                content = f.read()

            # Basic validation checks
            if "(kicad_pcb" not in content:
                return False
            if "(version" not in content:
                return False
            if "(generator" not in content:
                return False

            return True
        except:
            return False


if __name__ == "__main__":
    # Test the exporter
    exporter = KiCadExporter()

    # Create a test graph
    test_graph = {
        "nodes": {
            "0": {"type": "R", "position": [0, 0], "signal_type": "power"},
            "1": {"type": "C", "position": [10, 0], "signal_type": "ground"},
            "2": {"type": "U", "position": [5, 5], "signal_type": "signal"},
        },
        "edges": {
            "0-1": {
                "source": "0",
                "target": "1",
                "length": 10.0,
                "width": 0.2,
                "layer": 0,
            },
            "1-2": {
                "source": "1",
                "target": "2",
                "length": 7.07,
                "width": 0.2,
                "layer": 0,
            },
        },
    }

    success = exporter.export_to_kicad(test_graph, "test_output.kicad_pcb")
    if success:
        is_valid = exporter.validate_kicad_export("test_output.kicad_pcb")
        print(f"Export successful: {success}, Valid: {is_valid}")
    else:
        print("Export failed")
