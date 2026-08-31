from typing import List, Literal

from pydantic import BaseModel, Field


# ============================================================
# Strength
# ============================================================

class Strength(BaseModel):

    category: str = Field(
        description="UI/UX category of the strength."
    )

    description: str = Field(
        description="Description of the positive aspect."
    )

    evidence: str = Field(
        description="Evidence from the provided UI data."
    )


# ============================================================
# UI/UX Issue
# ============================================================

class UIUXIssue(BaseModel):

    id: int = Field(
        description="Unique issue ID."
    )

    category: str = Field(
        description="UI/UX issue category."
    )

    severity: Literal[
        "low",
        "medium",
        "high"
    ] = Field(
        description="Issue severity."
    )

    affected_elements: List[int] = Field(
        description="IDs of affected UI elements."
    )

    description: str = Field(
        description="Description of the UI/UX problem."
    )

    evidence: str = Field(
        description="Evidence supporting the issue."
    )

    recommendation: str = Field(
        description="Recommended improvement."
    )


# ============================================================
# Complete UI/UX Analysis
# ============================================================

class UIUXAnalysis(BaseModel):

    overall_score: float = Field(
        description="Overall UI/UX score from 0 to 100."
    )

    strengths: List[Strength] = Field(
        description="Evidence-based UI/UX strengths."
    )

    issues: List[UIUXIssue] = Field(
        description="Evidence-based UI/UX issues."
    )

    summary: str = Field(
        description="Overall summary of the UI/UX evaluation."
    )