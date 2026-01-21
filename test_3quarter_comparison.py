"""
Test Script for 3-Quarter Seasonal Comparison
Tests the new 3-quarter comparison feature with data aggregation and report generation.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path so 'backend' and 'core' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from api.services.seasonal_comparison_service import seasonal_comparison_service
from api.services.seasonal_report_formatter import generate_3_quarter_comparison_report


def test_3quarter_comparison():
    """
    Test 3-quarter seasonal comparison generation.
    
    Test Flow:
    1. Define 3 consecutive season IDs
    2. Fetch comparison data using seasonal_comparison_service
    3. Generate Word document
    4. Save and verify file
    """
    print("\n" + "="*80)
    print("TEST: 3-QUARTER SEASONAL COMPARISON")
    print("="*80 + "\n")
    
    # ============================
    # CONFIGURATION
    # ============================
    # Use 3 consecutive quarters: Q4-2025, Q1-2026, Q2-2026
    # Note: Adjust these season IDs based on your database
    season_ids = [4, 5, 6]  # Example: Q4-2025, Q1-2026, Q2-2026
    orgunit_id = 1  # Hospital level
    orgunit_type = 0  # 0=Hospital
    
    print(f"📋 Configuration:")
    print(f"   Season IDs: {season_ids}")
    print(f"   OrgUnit ID: {orgunit_id} (Type: {orgunit_type})")
    print()
    
    # ============================
    # STEP 1: FETCH COMPARISON DATA
    # ============================
    print("⏳ Step 1: Fetching 3-quarter comparison data...")
    start_time = datetime.now()
    
    try:
        comparison_data = seasonal_comparison_service.generate_3_quarter_comparison_data(
            season_ids=season_ids,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type,
            user_id=1
        )
        
        fetch_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Data fetched successfully ({fetch_time:.2f}s)")
        print()
        
        # Display summary
        print("📊 Comparison Summary:")
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
        print("📈 Trends:")
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
        doc = generate_3_quarter_comparison_report(
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
    filename = f"test_3quarter_comparison_{timestamp}.docx"
    
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
    print("       - Summary table (5 columns: Metric | Q1 | Q2 | Q3 | Trend)")
    print("       - Domain comparison table")
    print("       - Category comparison table")
    print("       - SubCategory comparison table")
    print("       - Page break before graphs")
    print("       - 3 Spider charts ONLY (Domain, Category, SubCategory)")
    print("       - NO bar charts or heatmaps")
    print("   [ ] Verify trend indicators (↑↑, ↑, →, ↓, ↓↓) are displayed")
    print("   [ ] Verify all text is bilingual (Arabic + English)")
    print("   [ ] Verify tables are right-aligned (RTL layout)")
    print("   [ ] Verify graphs are centered with bilingual captions")
    print()
    print(f"📄 Generated file: {filename}")
    print()
    
    return True


if __name__ == "__main__":
    success = test_3quarter_comparison()
    sys.exit(0 if success else 1)
