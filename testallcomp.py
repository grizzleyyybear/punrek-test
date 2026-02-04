#!/usr/bin/env python3
"""
Comprehensive test script for PunRek EDA Platform
Tests all components in order of dependency
"""

import os
import sys
import traceback
import tempfile
import json


def print_section(title):
    print(f"\n{'='*60}")
    print(f"TESTING: {title}")
    print(f"{'='*60}")


def test_imports():
    """Test basic imports"""
    print_section("Basic Imports")

    modules_to_test = [
        ("ai_engine.model", "AdvancedPCBGNN"),
        ("ai_engine.generator", "PCBGenerator"),
        ("ai_engine.constraint_cost", "calculate_constraint_cost"),
        ("security.analyzer", "SecurityAnalyzer"),
        ("exporter.kicad_export", "KiCadExporter"),
        ("backend.api.schemas", "LayoutRequest"),
        ("backend.api.routes", "router"),
    ]

    passed = 0
    total = len(modules_to_test)

    for module_path, class_name in modules_to_test:
        try:
            # Import module
            module = __import__(module_path, fromlist=[class_name])
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                print(f"✓ {module_path}: {class_name} imported successfully")
                passed += 1
            else:
                print(f"✗ {module_path}: {class_name} not found")
        except Exception as e:
            print(f"✗ {module_path}: Import failed - {e}")
            traceback.print_exc()

    return passed, total


def test_model_creation():
    """Test GNN model creation"""
    print_section("GNN Model Creation")

    try:
        from ai_engine.model import AdvancedPCBGNN
        import torch

        model = AdvancedPCBGNN(
            input_dim=5, hidden_dim=256, output_dim=128, num_layers=6
        )
        print("✓ AdvancedPCBGNN created successfully")

        # Test forward pass with dummy data
        x = torch.randn(5, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        positions = torch.randn(5, 2)

        outputs = model(x, edge_index, positions=positions)
        print("✓ Model forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_generator():
    """Test PCB generator"""
    print_section("PCB Generator")

    try:
        from ai_engine.generator import PCBGenerator

        # Create generator (uses dummy model if no trained model)
        gen = PCBGenerator()
        print("✓ PCBGenerator initialized")

        # Test with simple spec
        test_spec = {
            "component_count": 4,
            "max_trace_length": 15.0,
            "layers": 2,
            "power_domains": ["3.3V"],
            "signal_types": ["digital"],
            "constraints": {"min_clearance": 0.2},
            "area": 50,
        }

        pcb_graph, metrics = gen.generate_layout(test_spec)
        print(
            f"✓ Layout generated: {len(pcb_graph['nodes'])} nodes, {len(pcb_graph['edges'])} edges"
        )
        print(f"✓ Metrics: {metrics}")
        return True
    except Exception as e:
        print(f"✗ Generator test failed: {e}")
        traceback.print_exc()
        return False


def test_constraint_cost():
    """Test constraint cost calculator"""
    print_section("Constraint Cost Calculator")

    try:
        from ai_engine.constraint_cost import calculate_constraint_cost
        import networkx as nx

        # Create simple graph
        g = nx.Graph()
        g.add_node(0, signal_type="power", position=[0, 0])
        g.add_node(1, signal_type="ground", position=[10, 0])
        g.add_node(2, signal_type="signal", position=[5, 5])
        g.add_edge(0, 2, length=5.0, width=0.2, layer=0)
        g.add_edge(1, 2, length=8.0, width=0.2, layer=0)

        total_cost, costs = calculate_constraint_cost(g, {"max_trace_length": 10.0})
        print(f"✓ Constraint cost: {total_cost}, breakdown: {costs}")
        return True
    except Exception as e:
        print(f"✗ Constraint cost test failed: {e}")
        traceback.print_exc()
        return False


def test_security_analyzer():
    """Test security analyzer"""
    print_section("Security Analyzer")

    try:
        from security.analyzer import SecurityAnalyzer

        # Create test graph dict
        test_graph = {
            "nodes": {
                "0": {"type": "component", "signal_type": "power", "position": [0, 0]},
                "1": {
                    "type": "component",
                    "signal_type": "ground",
                    "position": [10, 0],
                },
                "2": {"type": "component", "signal_type": "signal", "position": [5, 5]},
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

        analyzer = SecurityAnalyzer()
        vulnerabilities, score = analyzer.analyze(test_graph)
        print(f"✓ Security analysis: {len(vulnerabilities)} issues, score: {score:.2f}")
        return True
    except Exception as e:
        print(f"✗ Security analyzer test failed: {e}")
        traceback.print_exc()
        return False


def test_kicad_exporter():
    """Test KiCad exporter"""
    print_section("KiCad Exporter")

    try:
        from exporter.kicad_export import KiCadExporter

        # Create test graph
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

        exporter = KiCadExporter()

        # Test export to temporary file
        with tempfile.NamedTemporaryFile(suffix=".kicad_pcb", delete=False) as tmp:
            success = exporter.export_to_kicad(test_graph, tmp.name)
            if success:
                print(f"✓ KiCad export successful to: {tmp.name}")

                # Verify file content
                with open(tmp.name, "r") as f:
                    content = f.read()
                    if "(kicad_pcb" in content and "segment" in content:
                        print("✓ File contains valid KiCad format")
                    else:
                        print("✗ File format issue")

                # Clean up
                os.unlink(tmp.name)
            else:
                print("✗ KiCad export failed")
            return success
    except Exception as e:
        print(f"✗ KiCad exporter test failed: {e}")
        traceback.print_exc()
        return False


def test_backend_api():
    """Test backend API endpoints (without running server)"""
    print_section("Backend API Endpoints")

    try:
        # Test schema imports
        from backend.api.schemas import (
            LayoutRequest,
            SecurityAnalysisRequest,
            ExportRequest,
        )
        from backend.api.routes import router

        # Test request creation
        layout_req = LayoutRequest(
            specification={
                "component_count": 4,
                "max_trace_length": 15.0,
                "layers": 2,
                "power_domains": ["3.3V"],
                "signal_types": ["digital"],
                "constraints": {"min_clearance": 0.2},
            }
        )
        print("✓ API schemas imported and instantiated")

        # Test route registration
        routes = [route.path for route in router.routes]
        print(f"✓ Routes registered: {routes}")

        return True
    except Exception as e:
        print(f"✗ Backend API test failed: {e}")
        traceback.print_exc()
        return False


def test_frontend_files():
    """Test frontend files exist and are readable"""
    print_section("Frontend Files")

    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "src")
    files_to_check = [
        "App.jsx",
        "components/SpecForm.jsx",
        "components/LayoutViewer.jsx",
    ]

    passed = 0
    total = len(files_to_check)

    for file_path in files_to_check:
        full_path = os.path.join(frontend_path, file_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, "r") as f:
                    content = f.read(100)  # Read first 100 chars
                print(f"✓ {file_path}: exists and readable")
                passed += 1
            except Exception as e:
                print(f"✗ {file_path}: exists but read error - {e}")
        else:
            print(f"✗ {file_path}: not found")

    return passed, total


def main():
    print("PUNREK EDA PLATFORM - COMPREHENSIVE TEST SCRIPT")
    print("=" * 80)

    # Test imports first
    import_passed, import_total = test_imports()

    # Test individual components
    tests = [
        ("GNN Model", test_model_creation),
        ("PCB Generator", test_generator),
        ("Constraint Cost", test_constraint_cost),
        ("Security Analyzer", test_security_analyzer),
        ("KiCad Exporter", test_kicad_exporter),
        ("Backend API", test_backend_api),
        ("Frontend Files", test_frontend_files),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                passed, total = result
                success_rate = f"{passed}/{total}"
                results.append((name, success_rate, passed == total))
            else:
                results.append((name, "PASS" if result else "FAIL", result))
        except Exception as e:
            print(f"Error in {name} test: {e}")
            results.append((name, "ERROR", False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed_count = 0
    total_count = len(results)

    for name, status, success in results:
        status_str = "✓ PASS" if success else "✗ FAIL"
        print(f"{status_str:8} {name:<20}: {status}")
        if success:
            passed_count += 1

    print(f"\nOverall: {passed_count}/{total_count} components working")

    if passed_count == total_count:
        print("\n🎉 ALL COMPONENTS WORKING! PunRek is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} components need attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
