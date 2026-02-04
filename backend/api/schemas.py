from typing import Dict, List, Optional
from pydantic import BaseModel


# Schema for PCB design specifications
class PCBSpecification(BaseModel):
    component_count: int
    max_trace_length: float
    layers: int
    power_domains: List[str]
    signal_types: List[str]
    constraints: Dict[str, float]


# Schema for layout generation requests
class LayoutRequest(BaseModel):
    specification: PCBSpecification
    seed: Optional[int] = None


# Schema for security analysis requests
class SecurityAnalysisRequest(BaseModel):
    pcb_graph: Dict


# Schema for export requests
class ExportRequest(BaseModel):
    pcb_graph: Dict
    filename: str


# Schema for layout generation responses
class LayoutResponse(BaseModel):
    success: bool
    pcb_graph: Dict
    message: str
    metrics: Dict


# Schema for security analysis responses
class SecurityAnalysisResponse(BaseModel):
    vulnerabilities: List[Dict]
    overall_score: float
    passed: bool


# Schema for export responses
class ExportResponse(BaseModel):
    success: bool
    filepath: str
    message: str
