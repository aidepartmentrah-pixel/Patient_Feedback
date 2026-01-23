"""
API Router: Graph Operators

FastAPI endpoints for generic graph operators.
Each operator has one endpoint that accepts a typed request and returns a typed response.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Any
import traceback

from ..schemas.operators.distribution import DistributionRequest, DistributionResponse
from ..services.operators.distribution_service import DistributionService


# Create router
router = APIRouter(
    prefix="/api/operators",
    tags=["Operators"],
)


# ============================================================================
# DISTRIBUTION OPERATOR ENDPOINT
# ============================================================================

@router.post(
    "/distribution",
    response_model=DistributionResponse,
    status_code=status.HTTP_200_OK,
    summary="Distribution Operator",
    description="""
    **Univariate Categorical Distribution Operator with Time Partitioning**
    
    Computes P(D=v | T) for a selected dimension D across time partition(s) T.
    
    **Capabilities:**
    - Compute marginal distribution of a categorical dimension
    - Compare distributions across discrete time buckets
    - Handle single, multiple, or binary time partitions
    
    **Time Modes:**
    - `single`: One time window (year, season, month, range) → 1 bucket
    - `multi`: Multiple time windows for comparison → N buckets
    - `binary_split`: Before/After a date → 2 buckets
    
    **Dimensions:**
    - domain, category, subcategory, classification
    - stage, severity, harm
    
    **Filters:**
    - Organizational: org_unit_id, administration_id, department_id, section_id
    - Dimensional: domain, category, severity, etc.
    
    **Response:**
    Each bucket contains:
    - `time_label`: Human-readable label (e.g., "2025", "2024-Q1")
    - `total`: Total incident count in this time period
    - `values`: Array of {key, count, percent} for each dimension value
    - `status`: "NO_DATA" if no incidents exist, null otherwise
    
    **Mathematical Guarantees:**
    - Sum of counts = total
    - Sum of percentages = 1.0 (within floating-point tolerance)
    - Percentages are between 0.0 and 1.0
    """,
    responses={
        200: {
            "description": "Successful distribution computation",
            "content": {
                "application/json": {
                    "example": {
                        "dimension": "severity",
                        "time_mode": "single",
                        "buckets": [
                            {
                                "time_label": "2025",
                                "total": 1234,
                                "values": [
                                    {"key": "High", "count": 234, "percent": 0.190},
                                    {"key": "Medium", "count": 700, "percent": 0.567},
                                    {"key": "Low", "count": 300, "percent": 0.243}
                                ],
                                "status": None
                            }
                        ]
                    }
                }
            }
        },
        422: {
            "description": "Validation error - invalid request format",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "time_mode"],
                                "msg": "SINGLE mode requires 'time_window' field",
                                "type": "value_error"
                            }
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred"
                    }
                }
            }
        }
    }
)
def distribution_operator(request: DistributionRequest) -> DistributionResponse:
    """
    Execute Distribution Operator.
    
    Args:
        request: DistributionRequest (validated by Pydantic)
        
    Returns:
        DistributionResponse with computed distribution buckets
        
    Raises:
        HTTPException: If internal error occurs during processing
    """
    try:
        # Create service and execute
        service = DistributionService()
        response = service.execute(request)
        
        return response
        
    except ValueError as e:
        # Business logic error (e.g., invalid dimension)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected error - log and return generic error
        print("=" * 80)
        print("ERROR IN DISTRIBUTION OPERATOR:")
        print(traceback.format_exc())
        print("=" * 80)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred"
        )
