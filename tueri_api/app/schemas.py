from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScanPromptRequest(BaseModel):
    prompt: str = Field(title="Prompt")
    scanners_suppress: List[str] = Field(title="Scanners to suppress", default=[])
    scanner_params: Optional[Dict[str, Dict[str, Any]]] = Field(
        title="Scanner parameter overrides",
        description="Override scanner parameters per request. Format: {'ScannerName': {'param': value}}",
        default=None
    )


class ScanPromptResponse(BaseModel):
    is_valid: bool = Field(title="Whether the prompt is safe")
    scanners: Dict[str, float] = Field(title="Risk scores of individual scanners")


class AnalyzePromptRequest(ScanPromptRequest):
    pass


class AnalyzePromptResponse(ScanPromptResponse):
    sanitized_prompt: str = Field(title="Sanitized prompt")


class ScanOutputRequest(BaseModel):
    prompt: str = Field(title="Prompt")
    output: str = Field(title="Model output")
    scanners_suppress: List[str] = Field(title="Scanners to suppress", default=[])
    scanner_params: Optional[Dict[str, Dict[str, Any]]] = Field(
        title="Scanner parameter overrides",
        description="Override scanner parameters per request. Format: {'ScannerName': {'param': value}}",
        default=None
    )


class ScanOutputResponse(BaseModel):
    is_valid: bool = Field(title="Whether the output is safe")
    scanners: Dict[str, float] = Field(title="Risk scores of individual scanners")


class AnalyzeOutputRequest(ScanOutputRequest):
    pass


class AnalyzeOutputResponse(ScanOutputResponse):
    sanitized_output: str = Field(title="Sanitized output")
