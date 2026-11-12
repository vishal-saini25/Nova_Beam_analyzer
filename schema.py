"""Pydantic models and deterministic checks for LLM-extracted beam data."""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ExtractedProblem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_type: Literal["beam_analysis", "needs_extension", "unsupported"]
    config: dict[str, Any] | None = None
    unknowns_requested: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


def parse_extraction(payload: str) -> ExtractedProblem:
    """Parse a JSON string and validate its outer extraction schema."""
    return ExtractedProblem.model_validate_json(payload)


def validate_config_shape(config: dict[str, Any] | None) -> list[str]:
    """Return required fields missing before calling the numerical solver."""
    if config is None:
        return ["A complete beam configuration could not be extracted."]

    missing: list[str] = []
    if config.get("L") is None:
        missing.append("Total beam length L (in metres).")
    if not isinstance(config.get("supports"), list) or not config["supports"]:
        missing.append("Support types and their positions from the left end.")
    section = config.get("section")
    if not isinstance(section, dict) or not section.get("type"):
        missing.append("Section dimensions / moment of inertia, or flexural rigidity EI.")
    elif str(section.get("type", "")).lower() not in {"ei", "4"} and config.get("E_GPa") is None:
        missing.append("Young's modulus E (or provide EI directly).")
    return missing


__all__ = ["ExtractedProblem", "ValidationError", "parse_extraction", "validate_config_shape"]
