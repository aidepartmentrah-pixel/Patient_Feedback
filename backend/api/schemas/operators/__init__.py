"""
Operator Schemas Package

This package contains Pydantic schemas for the Generic Graph Generator Engine.
Each operator has strict input/output contracts that define analytical transformations.
"""

from .base import (
    DimensionType,
    TimeMode,
    TimeWindowType,
    TimeWindow,
    TimeWindowYear,
    TimeWindowSeason,
    TimeWindowMonth,
    TimeWindowRange,
    OperatorFilters
)

from .distribution import (
    DistributionRequest,
    DistributionResponse,
    DistributionValue,
    DistributionBucket
)

__all__ = [
    # Base types
    "DimensionType",
    "TimeMode",
    "TimeWindowType",
    "TimeWindow",
    "TimeWindowYear",
    "TimeWindowSeason",
    "TimeWindowMonth",
    "TimeWindowRange",
    "OperatorFilters",
    
    # Distribution operator
    "DistributionRequest",
    "DistributionResponse",
    "DistributionValue",
    "DistributionBucket",
]
