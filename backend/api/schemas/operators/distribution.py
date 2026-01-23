"""
Distribution Operator Schema

Implements: Univariate Categorical Distribution Operator with Time Partitioning
Internal Name: DIST_1D_TIME_PARTITIONED

This operator computes P(D=v | T) for a selected dimension D across time partition(s) T.

Mathematical Definition:
- Given dimension D ∈ {domain, category, subcategory, classification, stage, severity, harm}
- Given time selection T producing subsets {R_T1, R_T2, ..., R_Tk}
- For each subset R_Ti and each value v in D:
    * count(v) = |{r ∈ R_Ti where r.D = v}|
    * percent(v) = count(v) / |R_Ti|

Capabilities:
- Compute marginal distribution P(D=v | T)
- Compare distributions across discrete time buckets
- Handle single, multiple, or binary time partitions

Limitations:
- Cannot show time continuity (use TREND operator)
- Cannot show relationships between two variables (use ASSOCIATION operator)
- Cannot show part-of-whole composition (use COMPOSITION operator)
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from .base import (
    DimensionType,
    TimeMode,
    TimeWindow,
    OperatorFilters
)


# ============================================================================
# REQUEST SCHEMA
# ============================================================================

class DistributionRequest(BaseModel):
    """
    Request schema for Distribution Operator.
    
    The operator supports three time modes:
    1. SINGLE: One time window → produces 1 bucket
    2. MULTI: Multiple time windows → produces N buckets (for comparison)
    3. BINARY_SPLIT: Before/After date → produces 2 buckets
    
    Time configuration is mutually exclusive based on mode.
    """
    
    # Core operator parameters
    dimension: DimensionType = Field(
        ...,
        description="The categorical dimension to analyze"
    )
    
    time_mode: TimeMode = Field(
        ...,
        description="Time partitioning strategy"
    )
    
    # Time configuration (mutually exclusive based on mode)
    time_window: Optional[TimeWindow] = Field(
        None,
        description="Single time window (required for SINGLE mode)"
    )
    
    time_windows: Optional[List[TimeWindow]] = Field(
        None,
        min_length=2,
        description="Multiple time windows (required for MULTI mode, min 2 windows)"
    )
    
    split_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Split date in YYYY-MM-DD format (required for BINARY_SPLIT mode)"
    )
    
    # Filters (orthogonal)
    filters: OperatorFilters = Field(
        default_factory=OperatorFilters,
        description="Optional filters applied before operator execution"
    )
    
    # ========================================================================
    # VALIDATION RULES
    # ========================================================================
    
    @model_validator(mode='after')
    def validate_time_configuration(self):
        """
        Enforce mutually exclusive time configuration based on mode.
        
        Rules:
        - SINGLE mode: time_window required, others must be None
        - MULTI mode: time_windows required (≥2), others must be None
        - BINARY_SPLIT mode: split_date required, others must be None
        """
        mode = self.time_mode
        
        # Count how many time configs are provided
        configs_provided = sum([
            self.time_window is not None,
            self.time_windows is not None,
            self.split_date is not None
        ])
        
        # Validate SINGLE mode
        if mode == TimeMode.SINGLE:
            if self.time_window is None:
                raise ValueError(
                    "SINGLE mode requires 'time_window' field. "
                    "Example: {\"type\": \"year\", \"value\": 2025}"
                )
            if configs_provided > 1:
                raise ValueError(
                    "SINGLE mode requires only 'time_window'. "
                    "Remove 'time_windows' and 'split_date'."
                )
        
        # Validate MULTI mode
        elif mode == TimeMode.MULTI:
            if self.time_windows is None:
                raise ValueError(
                    "MULTI mode requires 'time_windows' field with at least 2 windows."
                )
            if len(self.time_windows) < 2:
                raise ValueError(
                    "MULTI mode requires at least 2 time windows for comparison. "
                    f"Received {len(self.time_windows)} window(s)."
                )
            if configs_provided > 1:
                raise ValueError(
                    "MULTI mode requires only 'time_windows'. "
                    "Remove 'time_window' and 'split_date'."
                )
        
        # Validate BINARY_SPLIT mode
        elif mode == TimeMode.BINARY_SPLIT:
            if self.split_date is None:
                raise ValueError(
                    "BINARY_SPLIT mode requires 'split_date' field in YYYY-MM-DD format."
                )
            if configs_provided > 1:
                raise ValueError(
                    "BINARY_SPLIT mode requires only 'split_date'. "
                    "Remove 'time_window' and 'time_windows'."
                )
        
        return self
    
    @field_validator('time_windows')
    @classmethod
    def validate_time_windows_consistency(cls, v: Optional[List[TimeWindow]]) -> Optional[List[TimeWindow]]:
        """
        Ensure all time windows in MULTI mode have the same type.
        You cannot mix years with seasons or months.
        """
        if v is None:
            return v
        
        if len(v) == 0:
            return v
        
        # Get the type of the first window
        first_type = v[0].type
        
        # Check all windows have the same type
        for i, window in enumerate(v[1:], start=1):
            if window.type != first_type:
                raise ValueError(
                    f"All time windows must have the same type. "
                    f"Window 0 has type '{first_type}', but window {i} has type '{window.type}'. "
                    f"Cannot mix years, seasons, months, or ranges in MULTI mode."
                )
        
        return v
    
    model_config = {
        "extra": "forbid",  # Reject unexpected fields
        "json_schema_extra": {
            "examples": [
                {
                    "dimension": "severity",
                    "time_mode": "single",
                    "time_window": {"type": "year", "value": 2025},
                    "filters": {"department_id": 42}
                },
                {
                    "dimension": "domain",
                    "time_mode": "multi",
                    "time_windows": [
                        {"type": "season", "value": "2024-Q1"},
                        {"type": "season", "value": "2024-Q2"},
                        {"type": "season", "value": "2024-Q3"},
                        {"type": "season", "value": "2024-Q4"}
                    ],
                    "filters": {}
                },
                {
                    "dimension": "stage",
                    "time_mode": "binary_split",
                    "split_date": "2023-06-01",
                    "filters": {"severity": "High"}
                }
            ]
        }
    }


# ============================================================================
# RESPONSE SCHEMA
# ============================================================================

class DistributionValue(BaseModel):
    """
    Individual value in a distribution bucket.
    Represents count and percentage for one category value.
    """
    key: str = Field(..., description="Category value (e.g., 'Low', 'Medium', 'High')")
    count: int = Field(..., ge=0, description="Absolute count of incidents")
    percent: float = Field(..., ge=0.0, le=1.0, description="Percentage of total (0.0 to 1.0)")
    
    @field_validator('percent')
    @classmethod
    def validate_percent(cls, v: float) -> float:
        """Ensure percent is between 0 and 1"""
        if v < 0.0 or v > 1.0:
            raise ValueError("Percent must be between 0.0 and 1.0")
        return round(v, 6)  # Round to 6 decimal places for precision


class DistributionBucket(BaseModel):
    """
    A single time bucket containing the distribution for that period.
    
    States:
    - Normal: total > 0, values present, status = None
    - Zero: total = 0, values = [], status = None (valid measurement)
    - No Data: total = 0, values = [], status = "NO_DATA" (structural absence)
    """
    time_label: str = Field(
        ...,
        description="Human-readable label for this time bucket (e.g., '2025', '2024-Q1', 'Before')"
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total count of incidents in this time bucket"
    )
    values: List[DistributionValue] = Field(
        ...,
        description="Distribution of dimension values"
    )
    status: Optional[str] = Field(
        None,
        description="Status flag: 'NO_DATA' if no records exist, None otherwise"
    )
    
    @model_validator(mode='after')
    def validate_bucket_consistency(self):
        """
        Ensure internal consistency:
        - If total > 0, values must not be empty
        - If total = 0, values must be empty
        - Sum of counts must equal total (if values present)
        - Sum of percents must equal 1.0 (if values present)
        """
        if self.total > 0:
            if len(self.values) == 0:
                raise ValueError(
                    f"Bucket '{self.time_label}' has total={self.total} but no values. "
                    "Values list cannot be empty when total > 0."
                )
            
            # Verify sum of counts equals total
            sum_counts = sum(v.count for v in self.values)
            if sum_counts != self.total:
                raise ValueError(
                    f"Bucket '{self.time_label}': Sum of counts ({sum_counts}) "
                    f"does not equal total ({self.total})"
                )
            
            # Verify sum of percents equals 1.0 (with tolerance for floating point)
            sum_percents = sum(v.percent for v in self.values)
            if abs(sum_percents - 1.0) > 0.001:  # 0.1% tolerance
                raise ValueError(
                    f"Bucket '{self.time_label}': Sum of percents ({sum_percents:.6f}) "
                    "does not equal 1.0 (tolerance: ±0.001)"
                )
        
        else:  # total == 0
            if len(self.values) != 0:
                raise ValueError(
                    f"Bucket '{self.time_label}' has total=0 but contains values. "
                    "Values list must be empty when total=0."
                )
        
        return self
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        """Only allow 'NO_DATA' or None"""
        if v is not None and v != "NO_DATA":
            raise ValueError("status must be either 'NO_DATA' or None")
        return v


class DistributionResponse(BaseModel):
    """
    Response schema for Distribution Operator.
    
    Returns one or more buckets depending on time_mode:
    - SINGLE: 1 bucket
    - MULTI: N buckets (one per time window)
    - BINARY_SPLIT: 2 buckets (Before, After)
    """
    dimension: str = Field(
        ...,
        description="The dimension that was analyzed"
    )
    time_mode: str = Field(
        ...,
        description="Time partitioning mode used"
    )
    buckets: List[DistributionBucket] = Field(
        ...,
        min_length=1,
        description="Distribution buckets for each time partition"
    )
    
    @field_validator('buckets')
    @classmethod
    def validate_buckets_not_empty(cls, v: List[DistributionBucket]) -> List[DistributionBucket]:
        """Ensure at least one bucket is returned"""
        if len(v) == 0:
            raise ValueError("Response must contain at least one bucket")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dimension": "severity",
                    "time_mode": "single",
                    "buckets": [
                        {
                            "time_label": "2025",
                            "total": 1234,
                            "values": [
                                {"key": "Low", "count": 300, "percent": 0.243},
                                {"key": "Medium", "count": 700, "percent": 0.567},
                                {"key": "High", "count": 234, "percent": 0.190}
                            ],
                            "status": None
                        }
                    ]
                },
                {
                    "dimension": "stage",
                    "time_mode": "binary_split",
                    "buckets": [
                        {
                            "time_label": "Before",
                            "total": 0,
                            "values": [],
                            "status": "NO_DATA"
                        },
                        {
                            "time_label": "After",
                            "total": 856,
                            "values": [
                                {"key": "Stage 1", "count": 400, "percent": 0.467},
                                {"key": "Stage 2", "count": 456, "percent": 0.533}
                            ],
                            "status": None
                        }
                    ]
                }
            ]
        }
    }
