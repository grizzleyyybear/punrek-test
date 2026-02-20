import os
import tempfile
from fastapi import APIRouter, HTTPException
from ai_engine.generator import PCBGenerator
from .schemas import (
    ExportRequest,
    ExportResponse,
    LayoutRequest,
    LayoutResponse,
    SecurityAnalysisRequest,
    SecurityAnalysisResponse,
)
from exporter.kicad_export import KiCadExporter
from security.analyzer import SecurityAnalyzer

# Initialize API router and core components
router = APIRouter()
generator = PCBGenerator()
security_analyzer = SecurityAnalyzer()
exporter = KiCadExporter()


# Endpoint to generate a PCB layout
@router.post("/generate_layout", response_model=LayoutResponse)
async def generate_layout(request: LayoutRequest):
    try:
        pcb_graph, metrics = generator.generate_layout(request.specification.dict())
        return LayoutResponse(
            success=True,
            pcb_graph=pcb_graph,
            message="Layout generated successfully",
            metrics=metrics,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layout generation failed: {e}")


# Endpoint to analyze the security of a PCB layout
@router.post("/analyze_security", response_model=SecurityAnalysisResponse)
async def analyze_security(request: SecurityAnalysisRequest):
    try:
        vulnerabilities, score = security_analyzer.analyze(request.pcb_graph)
        return SecurityAnalysisResponse(
            vulnerabilities=vulnerabilities,
            overall_score=score,
            passed=score >= 0.8,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Security analysis failed: {e}")


# Endpoint to export a PCB layout to a KiCad file
@router.post("/export_kicad", response_model=ExportResponse)
async def export_kicad(request: ExportRequest):
    try:
        export_dir = os.path.join(tempfile.gettempdir(), "exports")
        os.makedirs(export_dir, exist_ok=True)
        filepath = os.path.join(export_dir, request.filename)
        success = exporter.export_to_kicad(request.pcb_graph, filepath)
        if success:
            return ExportResponse(
                success=True,
                filepath=filepath,
                message="Successfully exported to KiCad format",
            )
        raise HTTPException(status_code=500, detail="Export failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")
