"""
Test Script for 4-Quarter Seasonal Comparison (Full Year)
Tests the new 4-quarter comparison feature with data aggregation and report generation.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path so 'backend' and 'core' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from api.services.seasonal_comparison_service import seasonal_comparison_service
from api.services.seasonal_report_formatter import generate_4_quarter_comparison_report


def test_4quarter_comparison():
    """
    Test 4-quarter seasonal comparison generation (Full Year).
    
    Test Flow:
    1. Define 4 consecutive season IDs
    2. Fetch comparison data using seasonal_comparison_service
    3. Generate Word document
    4. Save and verify file
    """
    print("\n" + "="*80)
    print("TEST: 4-QUARTER SEASONAL COMPARISON (FULL YEAR)")
    print("="*80 + "\n")
    
    # ============================
    # CONFIGURATION
    # ============================
    # Use 4 consecutive quarters: Q1-2025, Q2-2025, Q3-2025, Q4-2025
    # Note: Adjust these season IDs based on your database
    season_ids = [2, 3, 4, 5]  # Example: Q1-2025, Q2-2025, Q3-2025, Q4-2025
    orgunit_id = 1  # Hospital level
    orgunit_type = 0  # 0=Hospital
    
    print(f"📋 Configuration:")
    print(f"   Season IDs: {season_ids}")
    print(f"   OrgUnit ID: {orgunit_id} (Type: {orgunit_type})")
    print()
    
    # ============================
    # STEP 1: FETCH COMPARISON DATA
    # ============================
    print("⏳ Step 1: Fetching 4-quarter comparison data...")
    start_time = datetime.now()
    
    try:
        comparison_data = seasonal_comparison_service.generate_4_quarter_comparison_data(
            season_ids=season_ids,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type,
            user_id=1
        )
        
        fetch_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Data fetched successfully ({fetch_time:.2f}s)")
        print()
        
        # Display summary
        print("📊 Quarterly Comparison Summary:")
        for i, report in enumerate(comparison_data['reports']):
            period = comparison_data['periods'][i]
            total_cases = report['header'].get('total_cases', 0)
            clinical = report['header'].get('clinical_domain_count', 0)
            management = report['header'].get('management_domain_count', 0)
            relational = report['header'].get('relational_domain_count', 0)
            
            print(f"   {period}:")
            print(f"      Total Cases: {total_cases}")
            print(f"      Clinical: {clinical}, Management: {management}, Relational: {relational}")
        
        print()
        print("📈 Yearly Totals:")
        yearly = comparison_data['yearly_totals']
        print(f"   Total Cases: {yearly['total_cases']}")
        print(f"   Clinical: {yearly['clinical']}, Management: {yearly['management']}, Relational: {yearly['relational']}")
        print(f"   Low: {yearly['low_severity']}, Medium: {yearly['medium_severity']}, High: {yearly['high_severity']}")
        
        print()
        print("📈 Trends (Q1 → Q4):")
        for metric, trend in comparison_data['trends'].items():
            print(f"   {metric}: {trend}")
        
        print()
        
    except Exception as e:
        print(f"❌ Failed to fetch comparison data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================
    # STEP 2: GENERATE WORD DOCUMENT
    # ============================
    print("⏳ Step 2: Generating Word document...")
    start_time = datetime.now()
    
    try:
        doc = generate_4_quarter_comparison_report(
            comparison_data=comparison_data,
            language='ar'
        )
        
        gen_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Document generated successfully ({gen_time:.2f}s)")
        print()
        
    except Exception as e:
        print(f"❌ Failed to generate document: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================
    # STEP 3: SAVE DOCUMENT
    # ============================
    print("⏳ Step 3: Saving document...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_4quarter_comparison_{timestamp}.docx"
    
    try:
        doc.save(filename)
        file_size = os.path.getsize(filename)
        print(f"✅ Document saved: {filename}")
        print(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print()
        
    except Exception as e:
        print(f"❌ Failed to save document: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================
    # STEP 4: VERIFICATION CHECKLIST
    # ============================
    print("="*80)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    print()
    print("📋 Manual Verification Checklist:")
    print("   [ ] Open the generated Word document")
    print("   [ ] Verify document structure:")
    print("       - Header with logo and title")
    print("       - Summary table (7 columns: Metric | Q1 | Q2 | Q3 | Q4 | Yearly | Trend)")
    print("       - Domain comparison table with yearly totals")
    print("       - Category comparison table with yearly totals")
    print("       - SubCategory comparison table with yearly totals")
    print("       - Page break before graphs")
    print("       - 3 Spider charts ONLY (Domain, Category, SubCategory)")
    print("       - 4 data series per chart (Q1, Q2, Q3, Q4)")
    print("       - NO bar charts or heatmaps")
    print("   [ ] Verify trend indicators (↑↑, ↑, →, ↓, ↓↓) are displayed")
    print("   [ ] Verify yearly totals are calculated correctly")
    print("   [ ] Verify all text is bilingual (Arabic + English)")
    print("   [ ] Verify tables are right-aligned (RTL layout)")
    print("   [ ] Verify graphs are centered with bilingual captions")
    print()
    print(f"📄 Generated file: {filename}")
    print()
    
    return True


if __name__ == "__main__":
    success = test_4quarter_comparison()
    sys.exit(0 if success else 1)
