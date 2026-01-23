"""
Base Schema Models for Graph Operators

Provides shared enums, time models, and filters used across all operators.
These are the foundational types that ensure consistency and type safety.
"""

from enum import Enum
from typing import Optional, Union, Annotated, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import date
import re


# ============================================================================
# DIMENSION ENUMS
# ============================================================================

class DimensionType(str, Enum):
    """
    Categorical dimensions available for analysis.
    These are the primary analytical axes in the incident data.
    """
    DOMAIN = "domain"
    CATEGORY = "category"
    SUBCATEGORY = "subcategory"
    CLASSIFICATION = "classification"
    STAGE = "stage"
    SEVERITY = "severity"
    HARM = "harm"


# ============================================================================
# TIME ENUMS AND MODELS
# ============================================================================

class TimeMode(str, Enum):
    """
    Time partitioning strategies.
    
    - SINGLE: One time window (produces 1 bucket)
    - MULTI: Multiple discrete time windows (produces N buckets for comparison)
    - BINARY_SPLIT: Before/After a specific date (produces 2 buckets)
    """
    SINGLE = "single"
    MULTI = "multi"
    BINARY_SPLIT = "binary_split"


class TimeWindowType(str, Enum):
    """
    Granularity of time windows.
    """
    YEAR = "year"
    SEASON = "season"
    MONTH = "month"
    RANGE = "range"


# ============================================================================
# TIME WINDOW MODELS (Discriminated Union)
# ============================================================================

class TimeWindowYear(BaseModel):
    """
    Year-based time window.
    Example: { "type": "year", "value": 2025 }
    """
    type: Literal[TimeWindowType.YEAR] = Field(default=TimeWindowType.YEAR)
    value: int = Field(..., ge=2000, le=2100, description="Year (e.g., 2025)")
    
    @field_validator('value')
    @classmethod
    def validate_year(cls, v: int) -> int:
        if v < 2000 or v > 2100:
            raise ValueError("Year must be between 2000 and 2100")
        return v
    
    def get_label(self) -> str:
        """Get display label for this time window"""
        return str(self.value)


class TimeWindowSeason(BaseModel):
    """
    Season-based time window (quarters or trimesters).
    Examples:
    - Quarter: { "type": "season", "value": "2025-Q1" }
    - Trimester: { "type": "season", "value": "2025-T1" }
    """
    type: Literal[TimeWindowType.SEASON] = Field(default=TimeWindowType.SEASON)
    value: str = Field(
        ...,
        pattern=r"^\d{4}-(Q[1-4]|T[1-3])$",
        description="Season in format YYYY-QN or YYYY-TN"
    )
    
    @field_validator('value')
    @classmethod
    def validate_season(cls, v: str) -> str:
        if not re.match(r"^\d{4}-(Q[1-4]|T[1-3])$", v):
            raise ValueError(
                "Season must be in format YYYY-QN (quarters) or YYYY-TN (trimesters). "
                "Examples: '2025-Q1', '2025-T2'"
            )
        return v
    
    def get_label(self) -> str:
        """Get display label for this time window"""
        return self.value


class TimeWindowMonth(BaseModel):
    """
    Month-based time window.
    Example: { "type": "month", "value": "2025-03" }
    """
    type: Literal[TimeWindowType.MONTH] = Field(default=TimeWindowType.MONTH)
    value: str = Field(
        ...,
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
        description="Month in format YYYY-MM"
    )
    
    @field_validator('value')
    @classmethod
    def validate_month(cls, v: str) -> str:
        if not re.match(r"^\d{4}-(0[1-9]|1[0-2])$", v):
            raise ValueError(
                "Month must be in format YYYY-MM. Example: '2025-03'"
            )
        return v
    
    def get_label(self) -> str:
        """Get display label for this time window"""
        return self.value


class TimeWindowRange(BaseModel):
    """
    Custom date range time window.
    Example: { "type": "range", "from_date": "2025-01-01", "to_date": "2025-06-30" }
    """
    type: Literal[TimeWindowType.RANGE] = Field(default=TimeWindowType.RANGE)
    from_date: date = Field(..., description="Start date (inclusive)")
    to_date: date = Field(..., description="End date (inclusive)")
    
    @model_validator(mode='after')
    def validate_date_range(self):
        """Ensure from_date is before to_date"""
        if self.from_date > self.to_date:
            raise ValueError("from_date must be before or equal to to_date")
        return self
    
    def get_label(self) -> str:
        """Get display label for this time window"""
        return f"{self.from_date.isoformat()} to {self.to_date.isoformat()}"


# Discriminated union of all time window types
TimeWindow = Annotated[
    Union[TimeWindowYear, TimeWindowSeason, TimeWindowMonth, TimeWindowRange],
    Field(discriminator="type")
]


# ============================================================================
# FILTERS
# ============================================================================

class OperatorFilters(BaseModel):
    """
    Orthogonal filters applied before any operator execution.
    These are dimension-independent constraints.
    """
    org_unit_id: Optional[int] = Field(None, description="Organization unit filter")
    administration_id: Optional[int] = Field(None, description="Administration filter")
    department_id: Optional[int] = Field(None, description="Department filter")
    section_id: Optional[int] = Field(None, description="Section filter")
    
    # Additional domain filters
    domain: Optional[str] = Field(None, description="Domain filter")
    category: Optional[str] = Field(None, description="Category filter")
    subcategory: Optional[str] = Field(None, description="Subcategory filter")
    classification: Optional[str] = Field(None, description="Classification filter")
    stage: Optional[str] = Field(None, description="Stage filter")
    severity: Optional[str] = Field(None, description="Severity filter")
    harm: Optional[str] = Field(None, description="Harm filter")
    
    model_config = {"extra": "forbid"}  # Don't allow unexpected fields
