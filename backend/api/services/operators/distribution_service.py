"""
Service Layer: Distribution Operator

Business logic for the Distribution Operator (DIST_1D_TIME_PARTITIONED).

This service orchestrates:
1. Request validation (already done by Pydantic schemas)
2. Time window to date range conversion
3. Database queries via DistributionDB
4. Data transformation into response format
5. Statistical computations (percentages, totals)

The service ensures mathematical correctness:
- Sum of counts = total
- Sum of percentages = 1.0
- Proper NO_DATA status handling
"""

from typing import List
from ...schemas.operators.distribution import (
    DistributionRequest,
    DistributionResponse,
    DistributionBucket,
    DistributionValue
)
from ...schemas.operators.base import TimeMode
from ...db_layer.operators.distribution_db import DistributionDB, TimeWindowConverter


class DistributionService:
    """
    Service layer for Distribution Operator.
    
    Coordinates DB queries and transforms raw data into validated response format.
    """
    
    def __init__(self, db: DistributionDB = None):
        """
        Initialize DistributionService.
        
        Args:
            db: Optional DistributionDB instance. If None, creates new instance.
        """
        self.db = db if db else DistributionDB()
    
    def execute(self, request: DistributionRequest) -> DistributionResponse:
        """
        Execute distribution operator.
        
        Args:
            request: Validated DistributionRequest
            
        Returns:
            DistributionResponse with one or more buckets
            
        Raises:
            ValueError: If request validation fails (should not happen if schema validated)
        """
        # Route to appropriate handler based on time mode
        if request.time_mode == TimeMode.SINGLE:
            buckets = self._execute_single(request)
        elif request.time_mode == TimeMode.MULTI:
            buckets = self._execute_multi(request)
        elif request.time_mode == TimeMode.BINARY_SPLIT:
            buckets = self._execute_binary(request)
        else:
            raise ValueError(f"Unknown time mode: {request.time_mode}")
        
        # Build response
        return DistributionResponse(
            dimension=request.dimension.value,
            time_mode=request.time_mode.value,
            buckets=buckets
        )
    
    def _execute_single(self, request: DistributionRequest) -> List[DistributionBucket]:
        """
        Execute SINGLE mode: one time window → one bucket.
        
        Args:
            request: DistributionRequest with time_window set
            
        Returns:
            List with single DistributionBucket
        """
        # Query database for this time window
        raw_data = self.db.query_single_window(
            dimension=request.dimension.value,
            time_window=request.time_window,
            filters=request.filters
        )
        
        # Get label for this time window
        label = request.time_window.get_label()
        
        # Transform to bucket
        bucket = self._raw_data_to_bucket(label, raw_data)
        
        return [bucket]
    
    def _execute_multi(self, request: DistributionRequest) -> List[DistributionBucket]:
        """
        Execute MULTI mode: multiple time windows → multiple buckets.
        
        Args:
            request: DistributionRequest with time_windows set
            
        Returns:
            List of DistributionBuckets (one per time window)
        """
        buckets = []
        
        for window in request.time_windows:
            # Query database for this time window
            raw_data = self.db.query_single_window(
                dimension=request.dimension.value,
                time_window=window,
                filters=request.filters
            )
            
            # Get label for this time window
            label = window.get_label()
            
            # Transform to bucket
            bucket = self._raw_data_to_bucket(label, raw_data)
            buckets.append(bucket)
        
        return buckets
    
    def _execute_binary(self, request: DistributionRequest) -> List[DistributionBucket]:
        """
        Execute BINARY_SPLIT mode: before/after a date → two buckets.
        
        Args:
            request: DistributionRequest with split_date set
            
        Returns:
            List with two DistributionBuckets (Before, After)
        """
        from datetime import datetime
        
        # Parse split_date string to date object
        split_date = datetime.strptime(request.split_date, "%Y-%m-%d").date()
        
        # Query "Before" (everything before split_date)
        before_data = self.db.query_date_range(
            dimension=request.dimension.value,
            from_date=None,  # No lower bound
            to_date=split_date,
            filters=request.filters
        )
        
        # Query "After" (everything after split_date, inclusive)
        from datetime import timedelta
        after_date = split_date + timedelta(days=1)  # Day after split_date
        
        after_data = self.db.query_date_range(
            dimension=request.dimension.value,
            from_date=after_date,
            to_date=None,  # No upper bound
            filters=request.filters
        )
        
        # Transform to buckets
        before_bucket = self._raw_data_to_bucket("Before", before_data)
        after_bucket = self._raw_data_to_bucket("After", after_data)
        
        return [before_bucket, after_bucket]
    
    def _raw_data_to_bucket(
        self, 
        label: str, 
        raw_data: List[dict]
    ) -> DistributionBucket:
        """
        Transform raw database results into a DistributionBucket.
        
        Computes:
        - Total count
        - Percentage for each value
        - NO_DATA status if total = 0
        
        Args:
            label: Human-readable label for this bucket (e.g., "2025", "2024-Q1")
            raw_data: List of dicts from DB with keys: dimension_value, count
            
        Returns:
            DistributionBucket with properly computed percentages
            
        Example raw_data:
            [
                {"dimension_value": "High", "count": 150},
                {"dimension_value": "Medium", "count": 300},
                {"dimension_value": "Low", "count": 50}
            ]
        """
        # Compute total
        total = sum(row["count"] for row in raw_data)
        
        # Handle empty data
        if total == 0:
            return DistributionBucket(
                time_label=label,
                total=0,
                values=[],
                status="NO_DATA"
            )
        
        # Compute percentages and build values
        values = []
        for row in raw_data:
            count = row["count"]
            percent = count / total
            
            values.append(DistributionValue(
                key=str(row["dimension_value"]) if row["dimension_value"] is not None else "Unknown",
                count=count,
                percent=percent
            ))
        
        # Return bucket
        return DistributionBucket(
            time_label=label,
            total=total,
            values=values,
            status=None
        )
