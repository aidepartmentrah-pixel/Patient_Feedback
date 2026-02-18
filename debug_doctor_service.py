"""
Debug script to test DoctorSeasonalReportingService directly
"""

import sys
from pathlib import Path
from datetime import date

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from api.services.doctor_seasonal_reporting_service import DoctorSeasonalReportingService

# Test with doctor ID 1 and 2025 dates (same as successful single doctor test)
doctor_id = 1
season_start = date(2025, 1, 1)
season_end = date(2025, 12, 31)

print(f"\nTesting DoctorSeasonalReportingService for Doctor {doctor_id}")
print(f"  Date range: {season_start} to {season_end}")
print("="*70)

try:
    payload = DoctorSeasonalReportingService.build_doctor_seasonal_report_data(
        doctor_id=doctor_id,
        season_start=season_start,
        season_end=season_end
    )
    
    print("\n✓ Payload generated successfully!")
    print(f"\nMetrics:")
    metrics = payload.get("metrics", {})
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\nDoctor info:")
    doctor_info = payload.get("doctor_info", {})
    for key, value in doctor_info.items():
        print(f"  {key}: {value}")
        
    print(f"\nIncidents details count: {len(payload.get('incidents_details', []))}")
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
