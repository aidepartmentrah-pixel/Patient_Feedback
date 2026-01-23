"""
Quick test script for distribution endpoint
"""
import sys
sys.path.insert(0, 'backend')

from api.schemas.operators.distribution import DistributionRequest
from api.schemas.operators.base import TimeWindowYear, TimeMode, DimensionType
from api.services.operators.distribution_service import DistributionService

# Create test request
request = DistributionRequest(
    dimension=DimensionType.SEVERITY,
    time_mode=TimeMode.SINGLE,
    time_window=TimeWindowYear(type="year", value=2025)
)

print("Request created successfully:")
print(f"  Dimension: {request.dimension}")
print(f"  Time Mode: {request.time_mode}")
print(f"  Time Window: {request.time_window}")

# Try to execute
try:
    service = DistributionService()
    print("\nExecuting request...")
    response = service.execute(request)
    print("\nResponse:")
    print(f"  Dimension: {response.dimension}")
    print(f"  Time Mode: {response.time_mode}")
    print(f"  Buckets: {len(response.buckets)}")
    for bucket in response.buckets:
        print(f"\n  Bucket: {bucket.time_label}")
        print(f"    Total: {bucket.total}")
        print(f"    Values: {len(bucket.values)}")
        for value in bucket.values[:5]:  # Show first 5
            print(f"      {value.key}: {value.count} ({value.percent:.1%})")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
