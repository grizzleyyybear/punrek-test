#!/usr/bin/env python3
"""
Complete PunRek workflow example
"""

import requests
import json


def complete_pcb_design_workflow():
    base_url = "http://localhost:8000/api/v1"

    print("🚀 Starting complete PCB design workflow...")

    # Step 1: Generate layout
    print("\n1️⃣  Generating PCB layout...")
    layout_payload = {
        "specification": {
            "component_count": 10,
            "max_trace_length": 30.0,
            "layers": 2,
            "power_domains": ["3.3V", "5V"],
            "signal_types": ["digital", "analog", "clock"],
            "constraints": {"min_clearance": 0.2},
        }
    }

    response = requests.post(f"{base_url}/generate_layout", json=layout_payload)
    if response.status_code == 200:
        layout_result = response.json()
        print(
            f"✅ Layout generated: {len(layout_result['pcb_graph']['nodes'])} nodes, {len(layout_result['pcb_graph']['edges'])} edges"
        )

        # Step 2: Security analysis
        print("\n2️⃣  Running security analysis...")
        security_payload = {"pcb_graph": layout_result["pcb_graph"]}
        sec_response = requests.post(
            f"{base_url}/analyze_security", json=security_payload
        )

        if sec_response.status_code == 200:
            sec_result = sec_response.json()
            print(
                f"✅ Security score: {sec_result['overall_score']:.2f}, Vulnerabilities: {len(sec_result['vulnerabilities'])}"
            )

            # Step 3: Export to KiCad
            print("\n3️⃣  Exporting to KiCad...")
            export_payload = {
                "pcb_graph": layout_result["pcb_graph"],
                "filename": "generated_board.kicad_pcb",
            }
            exp_response = requests.post(
                f"{base_url}/export_kicad", json=export_payload
            )

            if exp_response.status_code == 200:
                exp_result = exp_response.json()
                print(f"✅ Exported to: {exp_result['filepath']}")
                print("\n🎉 Complete workflow finished successfully!")
            else:
                print(f"❌ Export failed: {exp_response.text}")
        else:
            print(f"❌ Security analysis failed: {sec_response.text}")
    else:
        print(f"❌ Layout generation failed: {response.text}")


if __name__ == "__main__":
    complete_pcb_design_workflow()
