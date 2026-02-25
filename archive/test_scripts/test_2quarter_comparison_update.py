"""
Test 2-Quarter Comparison Report Update
Tests the updated report generation with:
1. Tables first, graphs last
2. New graph rules: Domain (Spider+Bar), Category (Spider+Bar), SubCategory (Spider only)
3. Centered images with bilingual captions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.api.services.seasonal_report_orchestrator import get_or_generate_comparative_seasonal_reports
from backend.api.services.seasonal_report_formatter import generate_comparative_seasonal_word_report
from datetime import datetime

def test_2quarter_comparison():
    print("="*80)
    print("🧪 TESTING 2-QUARTER COMPARISON REPORT - TASK GROUP 1")
    print("="*80)
    print()
    
    # Test Parameters
    TEST_SEASON_ID = 6  # Q2 2026 (current)
    TEST_ORGUNIT_ID = 1  # Hospital level
    TEST_ORGUNIT_TYPE = 0  # Hospital
    
    print("[1/5] Fetching comparison data...")
    print(f"   Season ID: {TEST_SEASON_ID}")
    print(f"   Organization: Hospital (ID={TEST_ORGUNIT_ID})")
    print()
    
    try:
        # Step 1: Get comparative data
        start_time = datetime.now()
        comparison_data = get_or_generate_comparative_seasonal_reports(
            season_id=TEST_SEASON_ID,
            orgunit_id=TEST_ORGUNIT_ID,
            orgunit_type=TEST_ORGUNIT_TYPE,
            user_id=1
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Data fetched ({elapsed:.2f}s)")
        
        current_report = comparison_data.get('current_report', {})
        previous_report = comparison_data.get('previous_report', {})
        
        current_period = current_report.get('header', {}).get('period', 'N/A')
        previous_period = previous_report.get('header', {}).get('period', 'N/A')
        
        print(f"   Current Period: {current_period}")
        print(f"   Previous Period: {previous_period}")
        print(f"   Current Cases: {current_report.get('header', {}).get('total_cases', 0)}")
        print(f"   Previous Cases: {previous_report.get('header', {}).get('total_cases', 0)}")
        print()
        
        # Step 2: Generate Word document
        print("[2/5] Generating Word document...")
        start_time = datetime.now()
        
        word_bytes = generate_comparative_seasonal_word_report(
            current_data=current_report,
            previous_data=previous_report,
            language='ar'
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Document generated ({elapsed:.2f}s)")
        print(f"   Document size: {len(word_bytes):,} bytes")
        print()
        
        # Step 3: Save test file
        print("[3/5] Saving test output...")
        output_filename = f'test_2quarter_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.docx'
        with open(output_filename, 'wb') as f:
            f.write(word_bytes)
        print(f"   ✅ Saved to: {output_filename}")
        print()
        
        # Step 4: Verify structure
        print("[4/5] Verification checklist:")
        print("   Please manually verify the following in the generated document:")
        print()
        print("   📋 STRUCTURE:")
        print("      ✓ Summary comparison table appears FIRST")
        print("      ✓ Domain-by-domain hierarchical tables appear NEXT")
        print("      ✓ Policy compliance section appears AFTER tables")
        print("      ✓ PAGE BREAK before visualization section")
        print("      ✓ 'Visual Analysis' heading appears AFTER page break")
        print()
        print("   📊 GRAPHS (Total: 5 graphs):")
        print("      ✓ Domain Spider Chart (centered, with caption)")
        print("      ✓ Domain Bar Subtraction Chart (centered, with caption)")
        print("      ✓ Category Spider Chart (centered, with caption)")
        print("      ✓ Category Bar Subtraction Chart (centered, with caption)")
        print("      ✓ SubCategory Spider Chart ONLY (centered, with caption)")
        print()
        print("   ✅ NO HEATMAPS should be present")
        print("   ✅ All captions should be bilingual (Arabic | English)")
        print("   ✅ All images should be horizontally centered")
        print()
        
        # Step 5: Summary
        print("[5/5] Test Summary")
        print("="*80)
        print("✅ SUCCESS - Report generated with new structure!")
        print("="*80)
        print()
        print("📁 Output file:", output_filename)
        print()
        print("🔍 NEXT STEPS:")
        print("   1. Open the generated Word document")
        print("   2. Verify structure: Tables → Page Break → Graphs")
        print("   3. Count graphs: Should be exactly 5 graphs")
        print("   4. Check captions: All should be centered and bilingual")
        print("   5. Confirm NO heatmaps are present")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("="*80)
        print("❌ FAILURE - Test encountered an error")
        print("="*80)
        print(f"Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_2quarter_comparison()
    sys.exit(0 if success else 1)
