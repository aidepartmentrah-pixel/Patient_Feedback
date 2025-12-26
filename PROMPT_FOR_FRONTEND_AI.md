# Prompt for Frontend AI Assistant

Copy and paste this entire prompt to the frontend AI coding assistant:

---

## Task: Implement Red Flags (Critical Issues) Page Frontend

I need you to create the frontend implementation for the **Red Flags page** in our Patient Feedback system. The backend API is already complete and running.

### API Base URL
```
http://127.0.0.1:8000
```

### Available Endpoints

#### 1. GET /api/red-flags - Get List of Red Flags
**Purpose:** Fetch filtered list of red flag incidents with pagination

**Query Parameters:**
- `search` (optional): Search by record ID or patient name
- `status` (optional): OPEN | UNDER_REVIEW | FINISHED | all
- `from_date` (optional): Filter from date (YYYY-MM-DD)
- `to_date` (optional): Filter to date (YYYY-MM-DD)
- `department` (optional): Filter by department name
- `category` (optional): Filter by category
- `severity` (optional): HIGH | CRITICAL
- `is_never_event` (optional): true | false
- `limit` (optional): Results per page (default: 100, max: 500)
- `offset` (optional): Pagination offset (default: 0)

**Example Request:**
```
GET http://127.0.0.1:8000/api/red-flags?status=OPEN&severity=CRITICAL&limit=50
```

**Response Format:**
```json
{
  "red_flags": [
    {
      "red_flag_id": 1,
      "recordID": "RF-2024-001",
      "patient_name": "محمد أحمد",
      "date_received": "2024-01-15",
      "department": "الطوارئ",
      "category": "Patient Safety",
      "severity": "CRITICAL",
      "status": "UNDER_REVIEW",
      "isNeverEvent": true,
      "complaint_summary": "حدث تأخير في تشخيص..."
    }
  ],
  "total": 156,
  "limit": 50,
  "offset": 0
}
```

#### 2. GET /api/red-flags/statistics - Get KPI Statistics
**Purpose:** Fetch summary statistics for dashboard cards

**Query Parameters:**
- `from_date` (optional): Statistics from date
- `to_date` (optional): Statistics to date

**Example Request:**
```
GET http://127.0.0.1:8000/api/red-flags/statistics
```

**Response Format:**
```json
{
  "total_red_flags": 245,
  "unfinished": 87,
  "finished": 158,
  "by_status": {
    "OPEN": 32,
    "UNDER_REVIEW": 55,
    "FINISHED": 158
  },
  "by_category": {
    "Patient Safety": 98,
    "Medical Errors": 67
  },
  "by_severity": {
    "CRITICAL": 89,
    "HIGH": 156
  },
  "current_month": {
    "count": 23,
    "month": "2024-12"
  },
  "previous_month": {
    "count": 19,
    "month": "2024-11"
  },
  "never_event_overlap": {
    "total_never_events": 45,
    "red_flags_also_never_events": 34,
    "never_events_only": 11,
    "red_flags_only": 211
  }
}
```

#### 3. GET /api/red-flags/trends - Get Trend Data for Charts
**Purpose:** Fetch time-series data for trend visualization

**Query Parameters:**
- `from_date` (optional): Trend from date
- `to_date` (optional): Trend to date
- `granularity` (optional): monthly | quarterly | weekly (default: monthly)
- `group_by` (optional): category | severity | department | none (default: none)

**Example Request:**
```
GET http://127.0.0.1:8000/api/red-flags/trends?granularity=monthly&group_by=category
```

**Response Format (No Grouping):**
```json
{
  "trends": [
    {
      "period": "يناير 2024",
      "count": 18
    },
    {
      "period": "فبراير 2024",
      "count": 22
    }
  ],
  "granularity": "monthly"
}
```

**Response Format (With Grouping):**
```json
{
  "trends": [
    {
      "period": "يناير 2024",
      "breakdown": {
        "Patient Safety": 8,
        "Medical Errors": 6,
        "Medication Issues": 4
      },
      "total": 18
    }
  ],
  "granularity": "monthly",
  "grouped_by": "category"
}
```

#### 4. GET /api/red-flags/{id} - Get Single Red Flag Details
**Purpose:** Fetch comprehensive details for a specific red flag (for modal/details view)

**Example Request:**
```
GET http://127.0.0.1:8000/api/red-flags/1
```

**Response Format:**
```json
{
  "red_flag_id": 1,
  "recordID": "RF-2024-001",
  "patient_name": "محمد أحمد",
  "date_received": "2024-01-15",
  "department": "الطوارئ",
  "category": "Patient Safety",
  "subcategory": "Delayed Diagnosis",
  "severity": "CRITICAL",
  "status": "UNDER_REVIEW",
  "isNeverEvent": true,
  "incident_details": {
    "complaint_text": "حدث تأخير في تشخيص...",
    "immediate_action": "تم نقل المريض للعناية المركزة",
    "actions_taken": "تم عقد اجتماع لفريق الطوارئ...",
    "root_cause": "نقص في عدد الأطباء المتخصصين",
    "harm_level": "Moderate Harm",
    "stage": "Occurrence Stage"
  },
  "timeline": [
    {
      "date": "2024-01-15",
      "event": "تلقي البلاغ",
      "details": "تم استلام البلاغ عن الحادث"
    }
  ],
  "related_actions": [
    {
      "action": "تحديث بروتوكول التشخيص",
      "responsible": "رئيس قسم الطوارئ",
      "deadline": "2024-02-15",
      "status": "In Progress"
    }
  ]
}
```

#### 5. POST /api/red-flags/{id}/export-pdf - Export PDF (NOT IMPLEMENTED)
**Status:** Returns 501 Not Implemented
**Note:** Show disabled button or "قريبًا" message in UI

#### 6. POST /api/red-flags/export-batch - Batch Export (NOT IMPLEMENTED)
**Status:** Returns 501 Not Implemented
**Note:** Show disabled button or "قريبًا" message in UI

### TypeScript Types Needed

```typescript
export interface RedFlag {
  red_flag_id: number;
  recordID: string;
  patient_name: string;
  date_received: string;
  department: string;
  category: string;
  severity: 'HIGH' | 'CRITICAL';
  status: 'OPEN' | 'UNDER_REVIEW' | 'FINISHED';
  isNeverEvent: boolean;
  complaint_summary: string;
}

export interface RedFlagsList {
  red_flags: RedFlag[];
  total: number;
  limit: number;
  offset: number;
}

export interface RedFlagStatistics {
  total_red_flags: number;
  unfinished: number;
  finished: number;
  by_status: {
    OPEN: number;
    UNDER_REVIEW: number;
    FINISHED: number;
  };
  by_category: Record<string, number>;
  by_severity: {
    CRITICAL: number;
    HIGH: number;
  };
  current_month: {
    count: number;
    month: string;
  };
  previous_month: {
    count: number;
    month: string;
  };
  never_event_overlap: {
    total_never_events: number;
    red_flags_also_never_events: number;
    never_events_only: number;
    red_flags_only: number;
  };
}

export interface TrendDataPoint {
  period: string;
  count?: number;
  breakdown?: Record<string, number>;
  total?: number;
}

export interface RedFlagTrends {
  trends: TrendDataPoint[];
  granularity: 'monthly' | 'quarterly' | 'weekly';
  grouped_by?: 'category' | 'severity' | 'department' | 'none';
}
```

### What to Implement

1. **Create API Service Layer** (e.g., `services/redFlagsApi.ts`)
   - Functions for all 4 working endpoints
   - Error handling with Arabic messages
   - axios or fetch implementation

2. **Red Flags List Page** with:
   - Data table showing all red flags
   - Filter controls (status, severity, date range, search, department, category)
   - Pagination (limit/offset based)
   - Click row to open details modal

3. **KPI Statistics Cards** at top of page:
   - Total red flags
   - Unfinished count (OPEN + UNDER_REVIEW)
   - Finished count
   - By severity breakdown (CRITICAL vs HIGH)
   - Current month count
   - Never Event overlap card

4. **Trend Chart Component**:
   - Line chart showing trends over time
   - Controls for granularity (monthly/quarterly/weekly)
   - Controls for grouping (category/severity/department/none)
   - Use the `/api/red-flags/trends` endpoint

5. **Details Modal**:
   - Opens when clicking a row in the table
   - Shows full red flag details from `/api/red-flags/{id}`
   - Display sections: Basic Info, Incident Details, Timeline, Related Actions
   - Close button

6. **UI/UX Requirements**:
   - Support RTL layout for Arabic text
   - Color-code severity badges (CRITICAL = red, HIGH = orange)
   - Color-code status badges (OPEN = blue, UNDER_REVIEW = yellow, FINISHED = green)
   - Show ✓ icon for Never Events (isNeverEvent = true)
   - Loading states while fetching data
   - Error messages in Arabic

7. **Export Buttons** (disabled for now):
   - "تصدير PDF" button (disabled with tooltip "قريبًا")
   - "تصدير جماعي" button (disabled with tooltip "قريبًا")

### Sample API Call (axios)

```typescript
import axios from 'axios';

const BASE_URL = 'http://127.0.0.1:8000';

// Get red flags list
const fetchRedFlags = async (filters) => {
  const response = await axios.get(`${BASE_URL}/api/red-flags`, {
    params: {
      status: filters.status,
      severity: filters.severity,
      from_date: filters.from_date,
      to_date: filters.to_date,
      limit: 50,
      offset: 0
    }
  });
  return response.data;
};

// Get statistics
const fetchStatistics = async () => {
  const response = await axios.get(`${BASE_URL}/api/red-flags/statistics`);
  return response.data;
};

// Get trends
const fetchTrends = async (granularity = 'monthly', groupBy = 'none') => {
  const response = await axios.get(`${BASE_URL}/api/red-flags/trends`, {
    params: { granularity, group_by: groupBy }
  });
  return response.data;
};

// Get single red flag
const fetchRedFlagDetails = async (id) => {
  const response = await axios.get(`${BASE_URL}/api/red-flags/${id}`);
  return response.data;
};
```

### Arabic Labels Reference

- **Status:**
  - OPEN → "مفتوح"
  - UNDER_REVIEW → "قيد المراجعة"
  - FINISHED → "منتهي"

- **Severity:**
  - CRITICAL → "حرج"
  - HIGH → "عالي"

- **UI Labels:**
  - Total Red Flags → "إجمالي الأعلام الحمراء"
  - Unfinished → "غير منتهي"
  - Finished → "منتهي"
  - Status → "الحالة"
  - Severity → "الخطورة"
  - Department → "القسم"
  - Category → "التصنيف"
  - Patient Name → "اسم المريض"
  - Date → "التاريخ"
  - Never Event → "حدث لا يجب أن يحدث"
  - Search → "بحث"
  - Filter → "تصفية"
  - Export PDF → "تصدير PDF"
  - Batch Export → "تصدير جماعي"
  - Coming Soon → "قريبًا"

### Testing

Test each endpoint in your browser or Postman first:
1. http://127.0.0.1:8000/api/red-flags?status=OPEN&limit=10
2. http://127.0.0.1:8000/api/red-flags/statistics
3. http://127.0.0.1:8000/api/red-flags/trends?granularity=monthly
4. http://127.0.0.1:8000/api/red-flags/1

Full API documentation available at: http://127.0.0.1:8000/docs

---

**Start by creating the API service layer, then build the components one by one. Test each endpoint as you integrate it.**
