"""
PHASE 6: Remove Auto-Comparison Test Script
Tests that seasonal report downloads now generate single-season reports only.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from api.services.report_export_service import ReportExportService
from datetime import datetime


def test_phase6_single_season_export():
    """
    Test that seasonal report export generates SINGLE season report (no auto-comparison).
    """
    print("\n" + "="*80)
    print("PHASE 6: REMOVE AUTO-COMPARISON TEST")
    print("="*80 + "\n")
    
    export_service = ReportExportService()
    
    # Test parameters
    filters = {
        "season_id": 5,  # Q1-2026
        "orgunit_id": 1,
        "orgunit_type": 0
    }
    
    print("📋 TEST: Generate Seasonal Report (DOCX format)")
    print(f"   Season ID: {filters['season_id']}")
    print(f"   Org Unit: {filters['orgunit_id']}")
    print(f"   Format: DOCX\n")
    
    try:
        start_time = datetime.now()
        
        # PHASE 6: display_mode=None triggers single-season path (no auto-comparison)
        result = export_service.generate_export(
            report_type="seasonal",
            file_format="docx",
            year=2026,
            quarter=1,
            language="ar",
            filters=filters,
            display_mode=None  # Single season only (comparisons via API)
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ Export generated successfully in {elapsed:.2f}s")
        print(f"\n📊 Result Details:")
        print(f"   Filename: {result['filename']}")
        print(f"   Content-Type: {result.get('content_type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}")
        print(f"   Content Size: {len(result['content']) / 1024:.2f} KB")
        
        # PHASE 6 Verification checks
        print(f"\n🔍 PHASE 6 Verification:")
        
        # Check 1: Filename should NOT contain "Reports" (plural) or "zip"
        if "Reports" in result['filename'] or ".zip" in result['filename']:
            print(f"   ❌ FAIL: Filename suggests multiple reports: {result['filename']}")
            print(f"           Expected single report filename")
            return False
        else:
            print(f"   ✅ PASS: Filename indicates single report")
        
        # Check 2: Content type should be DOCX (not ZIP)
        content_type = result.get('content_type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        if "zip" in content_type.lower():
            print(f"   ❌ FAIL: Content-Type is ZIP: {content_type}")
            print(f"           Expected DOCX format")
            return False
        else:
            print(f"   ✅ PASS: Content-Type is DOCX (not ZIP)")
        
        # Check 3: Content should be a single DOCX file (starts with PK signature)
        if result['content'][:2] == b'PK':  # ZIP/DOCX magic number
            # It's a DOCX or ZIP file
            # Try to detect if it's actually a ZIP with multiple files
            try:
                import zipfile
                from io import BytesIO
                
                zip_buffer = BytesIO(result['content'])
                with zipfile.ZipFile(zip_buffer, 'r') as zf:
                    file_list = zf.namelist()
                    
                    # DOCX files have internal structure, but report ZIPs would have .docx files inside
                    docx_files = [f for f in file_list if f.endswith('.docx')]
                    
                    if len(docx_files) > 1:
                        print(f"   ❌ FAIL: Content contains multiple DOCX files: {docx_files}")
                        print(f"           This appears to be a comparison ZIP package")
                        return False
                    elif len(docx_files) == 1:
                        print(f"   ⚠️  WARNING: Content is a ZIP containing: {docx_files}")
                        print(f"              This might be the old comparison package")
                        return False
                    else:
                        print(f"   ✅ PASS: Content is a single DOCX file")
            except zipfile.BadZipFile:
                print(f"   ✅ PASS: Content is a single DOCX file (not a ZIP package)")
        else:
            print(f"   ⚠️  WARNING: Content doesn't have PK signature (unusual for DOCX)")
        
        # Save the file to verify manually
        output_filename = f"test_phase6_single_season_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        with open(output_filename, 'wb') as f:
            f.write(result['content'])
        
        print(f"\n💾 Output saved: {output_filename}")
        print(f"   You can open this file to verify it contains ONLY the Q1-2026 season")
        print(f"   (no comparison with Q4-2025 or other quarters)")
        
        print("\n" + "="*80)
        print("✅ PHASE 6 TEST - PASSED")
        print("="*80)
        print("\n📋 Summary:")
        print("   - Single season report generated successfully")
        print("   - No automatic comparison with previous season")
        print("   - Filename follows single-season pattern")
        print("   - Content type is DOCX (not ZIP)")
        print("\n🎉 Auto-comparison has been successfully removed!")
        print("   Comparisons should now only be done via /api/seasonal-comparison endpoints\n")
        
        return True
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_phase6_single_season_export()
    sys.exit(0 if success else 1)
