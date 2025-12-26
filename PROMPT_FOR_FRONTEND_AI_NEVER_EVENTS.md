# Prompt for Frontend AI Assistant - Never Events Page

Copy and paste this entire prompt to the frontend AI coding assistant:

---

## Task: Implement Never Events Page Frontend

I need you to create the frontend implementation for the **Never Events page** in our Patient Feedback system. The backend API is already complete and running. Never events are zero-tolerance incidents that should never occur.

### API Base URL
```
http://127.0.0.1:8000
```

### Available Endpoints

#### 1. GET /api/never-events - Get List of Never Events
**Purpose:** Fetch filtered list of never event incidents with pagination

**Query Parameters:**
- `search` (optional): Search by record ID, patient name, or event type
- `status` (optional): OPEN | UNDER_REVIEW | FINISHED | all
- `from_date` (optional): Filter from date (YYYY-MM-DD)
- `to_date` (optional): Filter to date (YYYY-MM-DD)
- `department` (optional): Filter by department name
- `category` (optional): Filter by never event category
- `limit` (optional): Results per page (default: 100, max: 500)
- `offset` (optional): Pagination offset (default: 0)

**Example Request:**
```
GET http://127.0.0.1:8000/api/never-events?status=OPEN&limit=50
```

**Response Format:**
```json
{
  "never_events": [
    {
      "id": 1,
      "recordID": "NE-2024-001",
      "date": "2024-11-15",
      "patientName": "أحمد محمد",
      "patientID": "P-12345",
      "neverEventType": "Wrong Site Surgery",
      "neverEventTypeAr": "جراحة في موقع خاطئ",
      "neverEventCategory": "Surgical",
      "status": "FINISHED",
      "severity": "HIGH",
      "department": "Surgery",
      "qism": "قسم الجراحة العامة",
      "incidentID": "INC-2024-0234"
    }
  ],
  "total": 45,
  "limit": 100,
  "offset": 0
}
```

#### 2. GET /api/never-events/statistics - Get KPI Statistics
**Purpose:** Fetch summary statistics for dashboard cards

**Query Parameters:**
- `from_date` (optional): Statistics from date
- `to_date` (optional): Statistics to date

**Example Request:**
```
GET http://127.0.0.1:8000/api/never-events/statistics
```

**Response Format:**
```json
{
  "total_never_events": 45,
  "unfinished_count": 12,
  "finished_count": 33,
  "by_status": {
    "OPEN": 5,
    "UNDER_REVIEW": 7,
    "FINISHED": 33
  },
  "by_category": {
    "Surgical": 15,
    "Medication": 18,
    "Patient Safety": 8,
    "Device/Equipment": 4
  },
  "by_severity": {
    "HIGH": 42,
    "MEDIUM": 3,
    "LOW": 0
  },
  "current_month": {
    "count": 2,
    "month": "December 2024"
  },
  "previous_month": {
    "count": 3,
    "month": "November 2024"
  },
  "period": {
    "from": "2024-01-01",
    "to": "2024-12-31"
  }
}
```

#### 3. GET /api/never-events/trends - Get Trend Data for Charts
**Purpose:** Fetch time-series data for trend visualization

**Query Parameters:**
- `from_date` (optional): Trend from date (default: last 12 months)
- `to_date` (optional): Trend to date (default: today)
- `granularity` (optional): monthly | quarterly | weekly (default: monthly)
- `group_by` (optional): category | department | none (default: none)

**Example Request:**
```
GET http://127.0.0.1:8000/api/never-events/trends?granularity=monthly&group_by=category
```

**Response Format (No Grouping):**
```json
{
  "granularity": "monthly",
  "period": {
    "from": "2024-01-01",
    "to": "2024-12-31"
  },
  "data": [
    { "period": "Jan 2024", "count": 2 },
    { "period": "Feb 2024", "count": 1 },
    { "period": "Mar 2024", "count": 3 }
  ]
}
```

**Response Format (With Grouping):**
```json
{
  "granularity": "monthly",
  "group_by": "category",
  "data": [
    {
      "period": "Jan 2024",
      "total": 2,
      "breakdown": {
        "Surgical": 1,
        "Medication": 1,
        "Patient Safety": 0
      }
    }
  ]
}
```

#### 4. GET /api/never-events/{id} - Get Single Never Event Details
**Purpose:** Fetch comprehensive details for a specific never event (for modal view)

**Example Request:**
```
GET http://127.0.0.1:8000/api/never-events/1
```

**Response Format:**
```json
{
  "never_event": {
    "id": 1,
    "recordID": "NE-2024-001",
    "date": "2024-11-15",
    "patientName": "أحمد محمد",
    "patientID": "P-12345",
    "neverEventType": "Wrong Site Surgery",
    "neverEventTypeAr": "جراحة في موقع خاطئ",
    "neverEventCategory": "Surgical",
    "status": "FINISHED",
    "severity": "HIGH",
    "department": "Surgery",
    "qism": "قسم الجراحة العامة",
    "incidentID": "INC-2024-0234"
  },
  "incident_details": {
    "incidentID": "INC-2024-0234",
    "complaintText": "تم إجراء جراحة في الموقع الخطأ...",
    "immediateAction": "تم إيقاف العملية فوراً",
    "corrective_actions": "مراجعة بروتوكول تحديد الموقع الجراحي",
    "rootCause": "عدم اتباع بروتوكول التحقق المزدوج",
    "responsiblePerson": "د. خالد حسن",
    "targetDepartment": "Surgery",
    "feedbackReceivedDate": "2024-11-15",
    "classification": "Surgical > Site Error > Wrong Site Surgery"
  },
  "timeline": [
    {
      "date": "2024-11-15T08:30:00Z",
      "event": "تم الإبلاغ عن الحدث",
      "user": "نظام الإبلاغ"
    },
    {
      "date": "2024-11-15T09:00:00Z",
      "event": "بدء التحقيق",
      "user": "فريق الجودة"
    }
  ],
  "related_actions": [
    {
      "action_id": "ACT-2024-001",
      "description": "مراجعة البروتوكولات",
      "status": "in_progress",
      "due_date": "2024-12-01"
    }
  ]
}
```

### TypeScript Types Needed

```typescript
export interface NeverEvent {
  id: number;
  recordID: string;
  date: string;
  patientName: string;
  patientID: string;
  neverEventType: string;
  neverEventTypeAr: string;
  neverEventCategory: string;
  status: 'OPEN' | 'UNDER_REVIEW' | 'FINISHED';
  severity: string;
  department: string;
  qism: string;
  incidentID: string;
}

export interface NeverEventsList {
  never_events: NeverEvent[];
  total: number;
  limit: number;
  offset: number;
}

export interface NeverEventsStatistics {
  total_never_events: number;
  unfinished_count: number;
  finished_count: number;
  by_status: {
    OPEN: number;
    UNDER_REVIEW: number;
    FINISHED: number;
  };
  by_category: Record<string, number>;
  by_severity: Record<string, number>;
  current_month: {
    count: number;
    month: string;
  };
  previous_month: {
    count: number;
    month: string;
  };
  period: {
    from: string | null;
    to: string | null;
  };
}

export interface TrendDataPoint {
  period: string;
  count?: number;
  breakdown?: Record<string, number>;
  total?: number;
}

export interface NeverEventsTrends {
  granularity: 'monthly' | 'quarterly' | 'weekly';
  group_by?: 'category' | 'department' | 'none';
  data: TrendDataPoint[];
}

export interface NeverEventDetails {
  never_event: NeverEvent;
  incident_details: {
    incidentID: string;
    complaintText: string;
    immediateAction: string;
    corrective_actions: string;
    rootCause: string;
    responsiblePerson: string;
    targetDepartment: string;
    feedbackReceivedDate: string;
    classification: string;
  };
  timeline: Array<{
    date: string;
    event: string;
    user: string;
  }>;
  related_actions: Array<{
    action_id: string;
    description: string;
    status: string;
    due_date: string;
  }>;
}
```

### What to Implement

1. **Create API Service Layer** (e.g., `services/neverEventsApi.ts`)
   - Functions for all 4 endpoints
   - Error handling with Arabic messages
   - axios or fetch implementation

2. **Never Events List Page** with:
   - Data table showing all never events
   - Filter controls (status, date range, search, department, category)
   - Pagination (limit/offset based)
   - Click row to open details modal

3. **KPI Statistics Cards** at top of page:
   - Total never events (should be zero goal)
   - Unfinished count (OPEN + UNDER_REVIEW)
   - Finished count
   - By category breakdown
   - By severity breakdown
   - Current month count
   - Previous month count

4. **Trend Chart Component**:
   - Line chart showing trends over time
   - Controls for granularity (monthly/quarterly/weekly)
   - Controls for grouping (category/department/none)
   - Target line showing zero (goal)
   - Use the `/api/never-events/trends` endpoint

5. **Details Modal**:
   - Opens when clicking a row in the table
   - Shows full never event details from `/api/never-events/{id}`
   - Display sections:
     - Basic never event info
     - Linked incident details
     - Timeline with audit trail
     - Related corrective actions
   - Close button

6. **UI/UX Requirements**:
   - Support RTL layout for Arabic text
   - Color-code status badges (OPEN = blue, UNDER_REVIEW = yellow, FINISHED = green)
   - Severity is typically HIGH for all never events (red badge)
   - Show warning icons for never events (⚠️)
   - Loading states while fetching data
   - Error messages in Arabic
   - Zero-tolerance messaging (goal is zero never events)

7. **Special Considerations**:
   - Never events are more serious than regular incidents
   - Patient names may need anonymization
   - Emphasize that zero is the target
   - Timeline shows full audit trail
   - Related actions track corrective measures

### Sample API Call (axios)

```typescript
import axios from 'axios';

const BASE_URL = 'http://127.0.0.1:8000';

// Get never events list
const fetchNeverEvents = async (filters) => {
  const response = await axios.get(`${BASE_URL}/api/never-events`, {
    params: {
      status: filters.status,
      from_date: filters.from_date,
      to_date: filters.to_date,
      department: filters.department,
      category: filters.category,
      limit: 100,
      offset: 0
    }
  });
  return response.data;
};

// Get statistics
const fetchStatistics = async () => {
  const response = await axios.get(`${BASE_URL}/api/never-events/statistics`);
  return response.data;
};

// Get trends
const fetchTrends = async (granularity = 'monthly', groupBy = 'none') => {
  const response = await axios.get(`${BASE_URL}/api/never-events/trends`, {
    params: { granularity, group_by: groupBy }
  });
  return response.data;
};

// Get single never event
const fetchNeverEventDetails = async (id) => {
  const response = await axios.get(`${BASE_URL}/api/never-events/${id}`);
  return response.data;
};
```

### Arabic Labels Reference

- **Status:**
  - OPEN → "مفتوح"
  - UNDER_REVIEW → "قيد المراجعة"
  - FINISHED → "منتهي"

- **UI Labels:**
  - Never Events → "أحداث لا يجب أن تحدث"
  - Total Never Events → "إجمالي الأحداث"
  - Unfinished → "غير منتهي"
  - Finished → "منتهي"
  - Status → "الحالة"
  - Severity → "الخطورة"
  - Department → "القسم"
  - Category → "التصنيف"
  - Patient Name → "اسم المريض"
  - Date → "التاريخ"
  - Event Type → "نوع الحدث"
  - Incident ID → "معرف الحادث"
  - Search → "بحث"
  - Filter → "تصفية"
  - Details → "التفاصيل"
  - Timeline → "الجدول الزمني"
  - Related Actions → "الإجراءات ذات الصلة"
  - Root Cause → "السبب الجذري"
  - Corrective Actions → "الإجراءات التصحيحية"
  - Zero Tolerance → "عدم التسامح"
  - Goal: Zero → "الهدف: صفر"

### Key Differences from Red Flags

1. **Severity:** Never events are always HIGH severity (zero-tolerance)
2. **Goal:** Target is zero never events (not just reduction)
3. **Seriousness:** More critical than red flags
4. **Reporting:** May have regulatory/mandatory reporting requirements
5. **Timeline:** Full audit trail is critical
6. **Actions:** Corrective actions are mandatory, not optional

### Testing

Test each endpoint in your browser or Postman first:
1. http://127.0.0.1:8000/api/never-events?status=OPEN&limit=10
2. http://127.0.0.1:8000/api/never-events/statistics
3. http://127.0.0.1:8000/api/never-events/trends?granularity=monthly
4. http://127.0.0.1:8000/api/never-events/1

Full API documentation available at: http://127.0.0.1:8000/docs

---

**Start by creating the API service layer, then build the components one by one. Test each endpoint as you integrate it. Remember: Never events are zero-tolerance incidents—the goal is ZERO occurrences.**
