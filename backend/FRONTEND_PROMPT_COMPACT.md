# TableView Page - Frontend Implementation Prompt

## Task
Build a React/Next.js TableView page that displays patient complaints with filtering, searching, sorting, and pagination using the backend API.

## Backend API
**Base URL:** `http://127.0.0.1:8000`

### Endpoints

1. **GET `/api/complaints`** - Main data endpoint
   - Query params: `search`, `issuing_org_unit_id`, `domain_id`, `category_id`, `severity_id`, `stage_id`, `harm_level_id`, `case_status_id`, `year`, `month`, `start_date`, `end_date`, `sort_by`, `sort_order`, `page`, `page_size`, `view`
   - Returns: `{ complaints: [], pagination: {}, filters_applied: {}, view: "" }`

2. **GET `/api/complaints/filter-options`** - Dropdown options
   - Query params: `include_counts` (boolean)
   - Returns: `{ issuing_org_units: [], domains: [], categories: [], severities: [], stages: [], harm_levels: [], statuses: [] }`

3. **GET `/api/complaints/{id}`** - Single complaint details
   - Returns full complaint object

4. **GET `/api/complaints/count`** - Get filtered count
   - Same query params as endpoint 1
   - Returns: `{ total_count: number }`

5. **POST `/api/complaints/export`** - Export data
   - Body: `{ format: "csv"|"json", filters: {}, columns: [], include_patient_identifiers: bool, language: "en"|"ar" }`
   - Returns export metadata

6. **GET `/api/complaints/views`** - View configurations
   - Returns: `{ views: [{ id, name, name_ar, columns, default_sort, default_sort_order }] }`

## Requirements

### Page Structure
```
TableView Page (Arabic RTL)
├── Header with title "شكاوى المرضى" + Search Bar
├── Filter Panel
│   ├── Organizational Unit dropdown (issuing_org_unit_id)
│   ├── Domain dropdown (domain_id)
│   ├── Category dropdown (category_id)
│   ├── Severity dropdown (severity_id)
│   ├── Stage dropdown (stage_id)
│   ├── Status dropdown (case_status_id)
│   ├── Date range inputs (start_date, end_date)
│   └── Clear filters button
├── Data Table
│   ├── Sortable columns (click header to sort)
│   ├── Display 10 columns: complaint_number, received_date, patient_name, 
│   │   issuing_org_unit_name, domain_name, category_name, severity_name, 
│   │   stage_name, status_name, complaint_summary
│   ├── Colored badges for severity (Low=green, Medium=yellow, High=red)
│   ├── Colored badges for status (Open=blue, In Progress=yellow, Closed=green)
│   └── Clickable rows (navigate to /complaints/{id})
└── Pagination
    ├── Show "عرض X - Y من أصل Z سجل"
    └── First, Previous, Page numbers, Next, Last buttons
```

### Key Features

1. **State Management**
   - Store filters, pagination, loading state, error state
   - Default: `page: 1, page_size: 50, sort_by: 'FeedbackRecievedDate', sort_order: 'desc'`

2. **Filter Options**
   - Load on mount from `/api/complaints/filter-options?include_counts=true`
   - Show counts next to each option: "Emergency Department (45)"

3. **Search**
   - Debounce 500ms before calling API
   - Searches complaint_number, patient_name, complaint_text

4. **Sorting**
   - Click column header to toggle asc/desc
   - Show arrow icon (↑/↓) on active sort column

5. **Pagination**
   - Show max 5 page numbers centered around current page
   - Disable first/previous on page 1, disable next/last on last page

6. **Loading & Error States**
   - Show spinner while loading
   - Show Arabic error messages (use `message_ar` from error response)
   - Show "لا توجد شكاوى" if no results

7. **API Integration**
   - Use axios for HTTP requests
   - Reset to page 1 when filters change
   - Handle 404, 400, 500 errors gracefully

## Sample Data Structure

**Complaint object:**
```typescript
{
  id: number,
  complaint_number: number,
  complaint_summary: string,
  received_date: string, // "2024-12-15"
  patient_name: string,
  issuing_org_unit_name: string,
  domain_name: string,
  category_name: string,
  severity_id: number,
  severity_name: string,
  stage_name: string,
  case_status_id: number,
  status_name: string,
  // ... more fields
}
```

**Filter option:**
```typescript
{
  id: number,
  name: string,
  count?: number
}
```

## Implementation Notes

1. **Components to create:**
   - `TableView.tsx` (main page)
   - `FilterPanel.tsx` (all filter dropdowns)
   - `DataTable.tsx` (table with sorting)
   - `Pagination.tsx` (pagination controls)
   - `SearchBar.tsx` (search with debounce)
   - `complaintsApi.ts` (API service layer)

2. **Styling:**
   - RTL layout (`direction: rtl`)
   - Severity badges: severity-1 (green), severity-2 (yellow), severity-3 (red)
   - Status badges: status-1 (blue), status-2 (yellow), status-3 (green)
   - Hover effect on table rows
   - Responsive design

3. **API calls:**
   - Fetch filter options on mount
   - Fetch complaints whenever filters/pagination/sort changes
   - Use useEffect + useCallback for optimization
   - Handle loading states properly

4. **Arabic UI:**
   - All labels in Arabic
   - Error messages use `message_ar` from API
   - Date format can stay YYYY-MM-DD or convert to Arabic format

## Example API Call Flow

```typescript
// On mount
1. Load filter options: GET /api/complaints/filter-options?include_counts=true
2. Load initial complaints: GET /api/complaints?page=1&page_size=50&sort_by=FeedbackRecievedDate&sort_order=desc

// When user changes domain filter to ID 1
3. Load filtered complaints: GET /api/complaints?page=1&page_size=50&sort_by=FeedbackRecievedDate&sort_order=desc&domain_id=1

// When user clicks page 2
4. Load page 2: GET /api/complaints?page=2&page_size=50&sort_by=FeedbackRecievedDate&sort_order=desc&domain_id=1

// When user clicks row with ID 123
5. Navigate to: /complaints/123 (or open modal and fetch GET /api/complaints/123)
```

## TypeScript Interfaces

```typescript
interface ComplaintsQueryParams {
  search?: string;
  issuing_org_unit_id?: number;
  domain_id?: number;
  category_id?: number;
  severity_id?: number;
  stage_id?: number;
  harm_level_id?: number;
  case_status_id?: number;
  year?: number;
  month?: number;
  start_date?: string;
  end_date?: string;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  page?: number;
  page_size?: number;
  view?: string;
}

interface Complaint {
  id: number;
  complaint_number: number;
  complaint_summary: string;
  complaint_text: string;
  received_date: string;
  patient_name: string;
  doctor_name: string | null;
  issuing_org_unit_id: number;
  issuing_org_unit_name: string;
  domain_id: number;
  domain_name: string;
  category_id: number;
  category_name: string;
  severity_id: number | null;
  severity_name: string | null;
  stage_id: number;
  stage_name: string;
  harm_level_id: number;
  harm_level: string;
  case_status_id: number;
  status_name: string;
  created_at: string;
}

interface FilterOptions {
  issuing_org_units: Array<{ id: number; name: string; parent_id: number; count?: number }>;
  domains: Array<{ id: number; name: string; count?: number }>;
  categories: Array<{ id: number; name: string; domain_id: number; count?: number }>;
  severities: Array<{ id: number; name: string; count?: number }>;
  stages: Array<{ id: number; name: string; count?: number }>;
  harm_levels: Array<{ id: number; name: string; count?: number }>;
  statuses: Array<{ id: number; name: string; count?: number }>;
}

interface ComplaintsResponse {
  complaints: Complaint[];
  pagination: {
    page: number;
    page_size: number;
    total_records: number;
    total_pages: number;
  };
  filters_applied: Record<string, any>;
  view: string;
}
```

## Quick Start Code Snippet

```typescript
// services/complaintsApi.ts
import axios from 'axios';
const API_BASE_URL = 'http://127.0.0.1:8000';

export const complaintsApi = {
  async getComplaints(params: ComplaintsQueryParams) {
    const response = await axios.get(`${API_BASE_URL}/api/complaints`, { params });
    return response.data;
  },
  
  async getFilterOptions(includeCounts = false) {
    const response = await axios.get(`${API_BASE_URL}/api/complaints/filter-options`, {
      params: { include_counts: includeCounts }
    });
    return response.data;
  },
  
  async getComplaintById(id: number) {
    const response = await axios.get(`${API_BASE_URL}/api/complaints/${id}`);
    return response.data;
  }
};

// Main component structure
export function TableView() {
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filterOptions, setFilterOptions] = useState(null);
  const [filters, setFilters] = useState({
    page: 1,
    page_size: 50,
    sort_by: 'FeedbackRecievedDate',
    sort_order: 'desc'
  });

  useEffect(() => {
    loadFilterOptions();
  }, []);

  useEffect(() => {
    fetchComplaints();
  }, [filters]);

  const fetchComplaints = async () => {
    setLoading(true);
    try {
      const data = await complaintsApi.getComplaints(filters);
      setComplaints(data.complaints);
      setPagination(data.pagination);
    } catch (err) {
      setError(err.response?.data?.message_ar || 'خطأ في تحميل البيانات');
    } finally {
      setLoading(false);
    }
  };

  // ... rest of component
}
```

---

**Now implement this TableView page with all components following the structure above. Use React hooks, TypeScript, and ensure the UI is in Arabic with RTL support.**
