"""
Comprehensive test of all distribution modes
"""
import sys
sys.path.insert(0, 'backend')

from api.schemas.operators.distribution import DistributionRequest
from api.schemas.operators.base import (
    TimeWindowYear, TimeWindowSeason, TimeMode, DimensionType, OperatorFilters
)
from api.services.operators.distribution_service import DistributionService

service = DistributionService()

print("=" * 80)
print("TEST 1: Single Mode - Year")
print("=" * 80)
request1 = DistributionRequest(
    dimension=DimensionType.DOMAIN,
    time_mode=TimeMode.SINGLE,
    time_window=TimeWindowYear(type="year", value=2024)
)
try:
    response1 = service.execute(request1)
    print(f"✓ Success! Buckets: {len(response1.buckets)}, Total: {response1.buckets[0].total}")
    for value in response1.buckets[0].values[:3]:
        print(f"  - {value.key}: {value.count}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("TEST 2: Multi Mode - Multiple Years")
print("=" * 80)
request2 = DistributionRequest(
    dimension=DimensionType.SEVERITY,
    time_mode=TimeMode.MULTI,
    time_windows=[
        TimeWindowYear(type="year", value=2023),
        TimeWindowYear(type="year", value=2024),
        TimeWindowYear(type="year", value=2025)
    ]
)
try:
    response2 = service.execute(request2)
    print(f"✓ Success! Buckets: {len(response2.buckets)}")
    for bucket in response2.buckets:
        print(f"  - {bucket.time_label}: Total={bucket.total}, Values={len(bucket.values)}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("TEST 3: Binary Split Mode")
print("=" * 80)
request3 = DistributionRequest(
    dimension=DimensionType.CATEGORY,
    time_mode=TimeMode.BINARY_SPLIT,
    split_date="2024-06-01"
)
try:
    response3 = service.execute(request3)
    print(f"✓ Success! Buckets: {len(response3.buckets)}")
    for bucket in response3.buckets:
        print(f"  - {bucket.time_label}: Total={bucket.total}, Values={len(bucket.values)}")
        for value in bucket.values[:2]:
            print(f"      {value.key}: {value.count}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("TEST 4: With Filters")
print("=" * 80)
request4 = DistributionRequest(
    dimension=DimensionType.SEVERITY,
    time_mode=TimeMode.SINGLE,
    time_window=TimeWindowYear(type="year", value=2024),
    filters=OperatorFilters(
        domain="Clinical"  # Text filter example
    )
)
try:
    response4 = service.execute(request4)
    print(f"✓ Success! Buckets: {len(response4.buckets)}, Total: {response4.buckets[0].total}")
    for value in response4.buckets[0].values:
        print(f"  - {value.key}: {value.count}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("TEST 5: Season (Quarter)")
print("=" * 80)
request5 = DistributionRequest(
    dimension=DimensionType.STAGE,
    time_mode=TimeMode.SINGLE,
    time_window=TimeWindowSeason(type="season", value="2024-Q4")
)
try:
    response5 = service.execute(request5)
    print(f"✓ Success! Buckets: {len(response5.buckets)}, Total: {response5.buckets[0].total}")
    for value in response5.buckets[0].values[:3]:
        print(f"  - {value.key}: {value.count}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE!")
print("=" * 80)
