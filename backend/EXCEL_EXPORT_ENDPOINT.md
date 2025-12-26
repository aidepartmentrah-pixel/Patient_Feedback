# Excel Export Endpoint - Implementation Complete

## ⚠️ CRITICAL: Route Order Fixed

**Issue:** The `/export` endpoint was originally placed AFTER `/{complaint_id}` in the router, causing FastAPI to match "export" as a complaint_id parameter.

**Fix Applied:** Moved `/export` endpoint BEFORE `/{complaint_id}` endpoint.

**Correct Route Order:**
1. `GET /api/complaints` - Main list
2. `GET /api/complaints/filter-options` - Filter dropdowns
3. `GET /api/complaints/count` - Count filtered results
4. `GET /api/complaints/export` ← Excel export (MUST come before /{complaint_id})
5. `GET /api/complaints/{complaint_id}` - Single record (last, catches everything else)
6. `POST /api/complaints/export` - Metadata export
7. `GET /api/complaints/views` - View configurations

## Endpoint Details

**URL:** `GET /api/complaints/export`

**Purpose:** Export filtered complaints data as Excel (.xlsx) file

## Query Parameters (All Optional)

Same as the main `/api/complaints` endpoint:

- `search` (string): Search across complaint number, patient name, complaint text
- `issuing_org_unit_id` (integer): Filter by organizational unit
- `domain_id` (integer): Filter by domain
- `category_id` (integer): Filter by category
- `severity_id` (integer): Filter by severity level
- `stage_id` (integer): Filter by stage
- `harm_level_id` (integer): Filter by harm level
- `case_status_id` (integer): Filter by case status
- `year` (integer): Filter by year (YYYY)
- `month` (integer): Filter by month (1-12)
- `start_date` (string): Filter by date >= (YYYY-MM-DD)
- `end_date` (string): Filter by date <= (YYYY-MM-DD)

## Response

**Content-Type:** `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`

**File Format:** Excel (.xlsx)

**Filename Pattern:** `Complaints_Export_YYYYMMDD_HHMMSS.xlsx`

## Excel File Contents

The exported Excel file includes the following columns (19 columns in exact order):

1. **تاريخ تلقي الملاحظة** (Received Date)
2. **الرقم** (Complaint Number)
3. **اسم المريض** (Patient Name)
4. **قسم الصادر** (Issuing Department)
5. **قسم المعني** (Concerned Department)
6. **المصدر 1** (Source 1) - From APP_LOOKUP_SOURCE table
7. **النوع (Feedback Type)** - Feedback Intent Type
8. **Domain** - HCAT Domain
9. **Category** - HCAT Category
10. **SubCategory** - HCAT SubCategory
11. **New-Classification in Arabic** - Classification name
12. **محتوى الشكوى (Raw Content)** - Full complaint text
13. **Immediate Action** - Immediate action taken
14. **الإجراءات المتخذة** (Taken Action)
15. **Severity** - Severity level
16. **Stage** - Case stage
17. **Harm** - Harm level
18. **Status** - Case status
19. **FeedbackRiskType** - Clinical/Non-clinical risk type

## Features

- ✅ Headers in Arabic with blue background
- ✅ Right-aligned text (RTL support)
- ✅ Auto-adjusted column widths
- ✅ Formatted dates (YYYY-MM-DD format)
- ✅ Word wrapping enabled for long text
- ✅ All filters from the main table view work the same way

## Usage Examples

### Example 1: Export All Complaints
```
GET http://127.0.0.1:8000/api/complaints/export
```

### Example 2: Export Filtered by Domain
```
GET http://127.0.0.1:8000/api/complaints/export?domain_id=1
```

### Example 3: Export with Date Range
```
GET http://127.0.0.1:8000/api/complaints/export?start_date=2024-01-01&end_date=2024-12-31
```

### Example 4: Export with Multiple Filters
```
GET http://127.0.0.1:8000/api/complaints/export?domain_id=1&severity_id=2&issuing_org_unit_id=12
```

### Example 5: Export with Search
```
GET http://127.0.0.1:8000/api/complaints/export?search=emergency
```

## Frontend Implementation (React/TypeScript)

```typescript
// Add to complaintsApi.ts
export const complaintsApi = {
  // ... existing methods
  
  async exportToExcel(params: ComplaintsQueryParams): Promise<void> {
    const response = await axios.get(`${API_BASE_URL}/api/complaints/export`, {
      params,
      responseType: 'blob' // Important!
    });
    
    // Create download link
    const blob = new Blob([response.data], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    });
    
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    
    // Extract filename from Content-Disposition header
    const contentDisposition = response.headers['content-disposition'];
    const filename = contentDisposition
      ? contentDisposition.split('filename=')[1].replace(/"/g, '')
      : `Complaints_Export_${new Date().getTime()}.xlsx`;
    
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }
};
```

## Usage in Component

```typescript
// In your TableView component
const handleExport = async () => {
  setExporting(true);
  try {
    // Pass current filters to export
    await complaintsApi.exportToExcel(filters);
    // Show success message
    toast.success('تم تصدير البيانات بنجاح');
  } catch (error) {
    console.error('Export failed:', error);
    toast.error('فشل تصدير البيانات');
  } finally {
    setExporting(false);
  }
};

// In your JSX
<button 
  onClick={handleExport} 
  disabled={exporting}
>
  {exporting ? 'جاري التصدير...' : 'تصدير إلى Excel'}
  📥
</button>
```

## Error Handling

**Status Code 500:** Server error during export generation

```json
{
  "error": "export_failed",
  "message": "An error occurred while generating export: ...",
  "message_ar": "حدث خطأ أثناء إنشاء التصدير: ..."
}
```

## Notes

1. **No Pagination:** Export returns ALL matching records (not paginated)
2. **Performance:** Large exports (10,000+ records) may take a few seconds
3. **File Size:** Typical file is ~50-100KB per 1000 records
4. **Date Format:** Dates are exported as text in YYYY-MM-DD format for consistency
5. **Character Encoding:** Full Unicode support for Arabic text
6. **CORS:** Make sure to expose `Content-Disposition` header for filename extraction
7. **New Lookup Tables:** 
   - **APP_LOOKUP_SOURCE**: Contains 8 feedback sources (جولات, حضور, خط ساخن, صندوق, مشرف, موظف, واتساب مكتب, وسائل التواصل)
   - **APP_LOOKUP_SUBCATEGORY**: SubCategory classifications
   - **APP_LOOKUP_CLASSIFICATION**: New classification system with Arabic names

## Database Schema Update Required

Before using this export, run the SQL script to create the APP_LOOKUP_SOURCE table:

```sql
-- File: CREATE_SOURCE_LOOKUP_TABLE.sql
-- This creates APP_LOOKUP_SOURCE with 8 source options and adds SourceID to APP_IncidentCase
```

The script will:
- Create APP_LOOKUP_SOURCE table with SourceID, SourceName, SourceNameAr
- Insert 8 source options (Tours, Attendance, Hotline, Box, Supervisor, Employee, Office WhatsApp, Social Media)
- Add SourceID column to APP_IncidentCase table
- Create foreign key constraint

## Testing

Test the endpoint in your browser:

```
http://127.0.0.1:8000/api/complaints/export
http://127.0.0.1:8000/api/complaints/export?domain_id=1
http://127.0.0.1:8000/api/complaints/export?severity_id=2&start_date=2024-01-01
```

The browser should automatically download an Excel file.

---

**Status:** ✅ Updated with 19 Columns
**Database Update:** ⚠️ Run CREATE_SOURCE_LOOKUP_TABLE.sql first
**Backend Version:** Running on http://127.0.0.1:8000
**Last Updated:** December 26, 2025
