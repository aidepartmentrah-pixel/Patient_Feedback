# TableView Frontend Integration Guide

## 🎯 Overview

This document provides the complete API contract and integration guide for the **TableView** page, which is the main operational data table for viewing and managing feedback/incident records.

**Backend Status:** ✅ **COMPLETE** - All 6 endpoints implemented and tested  
**Backend Base URL:** `http://127.0.0.1:8000`

---

## 📋 Table of Contents

1. [API Endpoints Summary](#api-endpoints-summary)
2. [Endpoint Details & Integration Examples](#endpoint-details--integration-examples)
3. [Data Models & Field Descriptions](#data-models--field-descriptions)
4. [Frontend Implementation Checklist](#frontend-implementation-checklist)
5. [Error Handling](#error-handling)
6. [Testing Checklist](#testing-checklist)

---

## API Endpoints Summary

| Endpoint | Method | Purpose | Priority |
|----------|--------|---------|----------|
| `/api/complaints` | GET | Fetch paginated complaints list with filters | ⭐⭐⭐ HIGH |
| `/api/complaints/filter-options` | GET | Get dropdown options for filters | ⭐⭐⭐ HIGH |
| `/api/complaints/{id}` | GET | Get single complaint details (unmasked) | ⭐⭐ MEDIUM |
| `/api/complaints/count` | GET | Count complaints matching filters | ⭐⭐ MEDIUM |
| `/api/complaints/export` | POST | Export filtered complaints (CSV/JSON) | ⭐ LOW |
| `/api/complaints/views` | GET | Get table view configurations | ⭐ LOW |

---

## Endpoint Details & Integration Examples

### 1️⃣ GET /api/complaints (Primary Endpoint)

**Purpose:** Fetch paginated complaints list with extensive filtering and sorting.

**URL:** `GET http://127.0.0.1:8000/api/complaints`

**Query Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `search` | string | ❌ | Free-text search (complaint_number, patient_name, complaint_text) | `"Ahmed"` |
| `issuing_dept_id` | integer | ❌ | Filter by issuing department | `12` |
| `target_dept_id` | integer | ❌ | Filter by target department | `18` |
| `dayra_id` | integer | ❌ | Filter by primary department (Dayra) | `5` |
| `source` | string | ❌ | Filter by source (`"patient"`, `"staff"`, `"family"`, `"external"`) | `"patient"` |
| `status` | string | ❌ | Filter by status (`"open"`, `"in_progress"`, `"closed"`) | `"open"` |
| `severity_id` | integer | ❌ | Filter by severity level | `3` |
| `domain_id` | integer | ❌ | Filter by HCAT domain | `8` |
| `category_id` | integer | ❌ | Filter by category | `42` |
| `is_red_flag` | boolean | ❌ | Filter by red flag status | `true` |
| `is_never_event` | boolean | ❌ | Filter by Never Event status | `false` |
| `year` | integer | ❌ | Filter by year (YYYY) | `2024` |
| `month` | integer | ❌ | Filter by month (1-12) | `3` |
| `start_date` | string | ❌ | Filter by received_date >= (YYYY-MM-DD) | `"2024-03-01"` |
| `end_date` | string | ❌ | Filter by received_date <= (YYYY-MM-DD) | `"2024-03-31"` |
| `sort_by` | string | ❌ | Sort field | `"received_date"` |
| `sort_order` | string | ❌ | Sort order (`"asc"` or `"desc"`) | `"desc"` |
| `page` | integer | ❌ | Page number (1-indexed) | `1` |
| `page_size` | integer | ❌ | Results per page (1-500) | `50` |
| `view` | string | ❌ | View preset (`"complete"`, `"simplified"`, `"red_flags_only"`) | `"complete"` |

**Valid `sort_by` Values:**
- `"received_date"` (default)
- `"complaint_number"`
- `"severity_id"`
- `"status"`
- `"days_open"`
- `"updated_at"`
- `"patient_name"`

**Response Structure:**

```typescript
interface ComplaintsResponse {
  complaints: Complaint[];
  pagination: {
    page: number;
    page_size: number;
    total_records: number;
    total_pages: number;
  };
  filters_applied: {
    search: string | null;
    issuing_dept_id: number | null;
    target_dept_id: number | null;
    // ... all filter parameters
  };
  view: string;
}

interface Complaint {
  // Core fields
  id: number;
  complaint_number: string;
  complaint_summary: string;
  received_date: string; // YYYY-MM-DD
  incident_date: string | null;
  created_at: string; // ISO 8601
  updated_at: string; // ISO 8601
  
  // Patient info (MASKED)
  patient_mrn: string; // "MRN-***789"
  patient_name: string; // "Ahmed H."
  patient_age: number;
  patient_gender: "M" | "F";
  patient_gender_ar: string;
  
  // Issuing department
  issuing_dept_id: number;
  issuing_dept_name: string;
  issuing_dept_name_ar: string;
  issuing_dept_code: string;
  
  // Target department
  target_dept_id: number;
  target_dept_name: string;
  target_dept_name_ar: string;
  target_dept_code: string;
  
  // Primary department
  dayra_id: number;
  dayra_name: string;
  dayra_name_ar: string;
  
  // Classification
  domain_id: number;
  domain_name: string;
  domain_name_ar: string;
  category_id: number;
  category_name: string;
  category_name_ar: string;
  classification_id: number;
  classification_name_ar: string;
  
  // Severity
  severity_id: number;
  severity_name: string;
  severity_name_ar: string;
  severity_color: string; // Hex color code
  
  // Stage
  stage_id: number;
  stage_name: string;
  stage_name_ar: string;
  
  // Harm level
  harm_level: string;
  harm_level_ar: string;
  
  // Status and flags
  status: "open" | "in_progress" | "closed";
  status_ar: string;
  status_display: string; // "🟢 Open"
  is_closed: boolean;
  closure_date: string | null;
  is_red_flag: boolean;
  is_never_event: boolean;
  is_improvement_opportunity: boolean;
  priority: "urgent" | "high" | "normal" | "low";
  priority_ar: string;
  
  // Source
  source: "patient" | "staff" | "family" | "external";
  source_ar: string;
  source_detail: string;
  
  // Temporal metrics
  days_open: number | null;
  days_to_closure: number | null;
  
  // Follow-up indicators
  has_follow_up_actions: boolean;
  pending_actions_count: number;
  delayed_actions_count: number;
  
  // Permissions
  can_edit: boolean;
  can_delete: boolean;
}
```

**React Integration Example:**

```typescript
import { useState, useEffect } from 'react';
import axios from 'axios';

const TableView = () => {
  const [complaints, setComplaints] = useState([]);
  const [pagination, setPagination] = useState({ page: 1, page_size: 50, total_records: 0, total_pages: 0 });
  const [filters, setFilters] = useState({
    search: '',
    status: null,
    issuing_dept_id: null,
    start_date: null,
    end_date: null
  });
  const [loading, setLoading] = useState(false);

  const fetchComplaints = async () => {
    setLoading(true);
    try {
      const params = {
        page: pagination.page,
        page_size: pagination.page_size,
        ...filters
      };
      
      // Remove null/empty values
      Object.keys(params).forEach(key => {
        if (params[key] === null || params[key] === '') {
          delete params[key];
        }
      });

      const response = await axios.get('http://127.0.0.1:8000/api/complaints', { params });
      
      setComplaints(response.data.complaints);
      setPagination(response.data.pagination);
    } catch (error) {
      console.error('Error fetching complaints:', error);
      // Show error toast
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchComplaints();
  }, [pagination.page, pagination.page_size, filters]);

  const handleSearch = (searchTerm) => {
    setFilters({ ...filters, search: searchTerm });
    setPagination({ ...pagination, page: 1 }); // Reset to page 1 on new search
  };

  const handleFilterChange = (filterName, value) => {
    setFilters({ ...filters, [filterName]: value });
    setPagination({ ...pagination, page: 1 }); // Reset to page 1 on filter change
  };

  const handlePageChange = (newPage) => {
    setPagination({ ...pagination, page: newPage });
  };

  return (
    <div className="table-view">
      {/* Search bar */}
      <input
        type="text"
        placeholder="Search by ID or patient name..."
        value={filters.search}
        onChange={(e) => handleSearch(e.target.value)}
      />

      {/* Filters */}
      <select
        value={filters.status || ''}
        onChange={(e) => handleFilterChange('status', e.target.value || null)}
      >
        <option value="">All Statuses</option>
        <option value="open">Open</option>
        <option value="in_progress">In Progress</option>
        <option value="closed">Closed</option>
      </select>

      {/* Table */}
      {loading ? (
        <div>Loading...</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Complaint Number</th>
              <th>Received Date</th>
              <th>Patient</th>
              <th>Department</th>
              <th>Domain</th>
              <th>Severity</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {complaints.map((complaint) => (
              <tr key={complaint.id}>
                <td>{complaint.id}</td>
                <td>{complaint.complaint_number}</td>
                <td>{complaint.received_date}</td>
                <td>{complaint.patient_name}</td>
                <td>{complaint.issuing_dept_name}</td>
                <td>{complaint.domain_name}</td>
                <td>
                  <span style={{ color: complaint.severity_color }}>
                    {complaint.severity_name}
                  </span>
                </td>
                <td>{complaint.status_display}</td>
                <td>
                  <button onClick={() => handleView(complaint.id)}>View</button>
                  {complaint.can_edit && (
                    <button onClick={() => handleEdit(complaint.id)}>Edit</button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* Pagination */}
      <div className="pagination">
        <button
          disabled={pagination.page === 1}
          onClick={() => handlePageChange(pagination.page - 1)}
        >
          Previous
        </button>
        <span>
          Page {pagination.page} of {pagination.total_pages} ({pagination.total_records} records)
        </span>
        <button
          disabled={pagination.page === pagination.total_pages}
          onClick={() => handlePageChange(pagination.page + 1)}
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default TableView;
```

---

### 2️⃣ GET /api/complaints/filter-options

**Purpose:** Fetch available filter options for dropdown population.

**URL:** `GET http://127.0.0.1:8000/api/complaints/filter-options`

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `include_counts` | boolean | ❌ | Include record count for each option (default: false) |

**Response Structure:**

```typescript
interface FilterOptions {
  issuing_departments: Department[];
  target_departments: Department[];
  sources: Source[];
  statuses: Status[];
  severities: Severity[];
  domains: Domain[];
  categories: Category[];
}

interface Department {
  id: number;
  name: string;
  name_ar: string;
  code: string;
  count?: number; // Only if include_counts=true
}

interface Source {
  value: string; // "patient", "staff", "family", "external"
  label: string;
  label_ar: string;
  count?: number;
}

interface Status {
  value: string; // "open", "in_progress", "closed"
  label: string;
  label_ar: string;
  count?: number;
}

interface Severity {
  id: number;
  name: string;
  name_ar: string;
  color: string; // Hex color
  count?: number;
}

interface Domain {
  id: number;
  name: string;
  name_ar: string;
  count?: number;
}

interface Category {
  id: number;
  name: string;
  name_ar: string;
  domain_id: number;
  count?: number;
}
```

**React Integration Example:**

```typescript
import { useState, useEffect } from 'react';
import axios from 'axios';

const Filters = ({ onFilterChange }) => {
  const [filterOptions, setFilterOptions] = useState(null);

  useEffect(() => {
    const fetchFilterOptions = async () => {
      try {
        const response = await axios.get(
          'http://127.0.0.1:8000/api/complaints/filter-options',
          { params: { include_counts: true } }
        );
        setFilterOptions(response.data);
      } catch (error) {
        console.error('Error fetching filter options:', error);
      }
    };

    fetchFilterOptions();
  }, []);

  if (!filterOptions) return <div>Loading filters...</div>;

  return (
    <div className="filters">
      {/* Issuing Department Filter */}
      <select onChange={(e) => onFilterChange('issuing_dept_id', parseInt(e.target.value) || null)}>
        <option value="">All Issuing Departments</option>
        {filterOptions.issuing_departments.map((dept) => (
          <option key={dept.id} value={dept.id}>
            {dept.name} {dept.count && `(${dept.count})`}
          </option>
        ))}
      </select>

      {/* Source Filter */}
      <select onChange={(e) => onFilterChange('source', e.target.value || null)}>
        <option value="">All Sources</option>
        {filterOptions.sources.map((source) => (
          <option key={source.value} value={source.value}>
            {source.label} {source.count && `(${source.count})`}
          </option>
        ))}
      </select>

      {/* Status Filter */}
      <select onChange={(e) => onFilterChange('status', e.target.value || null)}>
        <option value="">All Statuses</option>
        {filterOptions.statuses.map((status) => (
          <option key={status.value} value={status.value}>
            {status.label} {status.count && `(${status.count})`}
          </option>
        ))}
      </select>

      {/* Severity Filter */}
      <select onChange={(e) => onFilterChange('severity_id', parseInt(e.target.value) || null)}>
        <option value="">All Severities</option>
        {filterOptions.severities.map((severity) => (
          <option key={severity.id} value={severity.id}>
            {severity.name} {severity.count && `(${severity.count})`}
          </option>
        ))}
      </select>

      {/* Domain Filter */}
      <select onChange={(e) => onFilterChange('domain_id', parseInt(e.target.value) || null)}>
        <option value="">All Domains</option>
        {filterOptions.domains.map((domain) => (
          <option key={domain.id} value={domain.id}>
            {domain.name} {domain.count && `(${domain.count})`}
          </option>
        ))}
      </select>
    </div>
  );
};

export default Filters;
```

---

### 3️⃣ GET /api/complaints/{id}

**Purpose:** Fetch full details of a single complaint (with UNMASKED patient data).

**URL:** `GET http://127.0.0.1:8000/api/complaints/{id}`

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | integer | ✅ | Complaint CaseID |

**Response Structure:**

Same as `Complaint` interface, but with **UNMASKED** patient data:
- `patient_mrn`: Full MRN (e.g., "MRN-123456789")
- `patient_name`: Full name (e.g., "Ahmed Hassan")

**React Integration Example:**

```typescript
import { useState, useEffect } from 'react';
import axios from 'axios';

const ComplaintDetails = ({ complaintId }) => {
  const [complaint, setComplaint] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchComplaint = async () => {
      try {
        const response = await axios.get(`http://127.0.0.1:8000/api/complaints/${complaintId}`);
        setComplaint(response.data);
      } catch (err) {
        if (err.response?.status === 404) {
          setError('Complaint not found');
        } else {
          setError('Error loading complaint details');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchComplaint();
  }, [complaintId]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div className="error">{error}</div>;
  if (!complaint) return null;

  return (
    <div className="complaint-details">
      <h2>{complaint.complaint_number}</h2>
      
      <section>
        <h3>Patient Information (Confidential)</h3>
        <p><strong>MRN:</strong> {complaint.patient_mrn}</p>
        <p><strong>Name:</strong> {complaint.patient_name}</p>
        <p><strong>Age:</strong> {complaint.patient_age}</p>
        <p><strong>Gender:</strong> {complaint.patient_gender}</p>
      </section>

      <section>
        <h3>Complaint Details</h3>
        <p><strong>Received:</strong> {complaint.received_date}</p>
        <p><strong>Incident Date:</strong> {complaint.incident_date}</p>
        <p><strong>Status:</strong> {complaint.status_display}</p>
        <p><strong>Severity:</strong> 
          <span style={{ color: complaint.severity_color }}>
            {complaint.severity_name}
          </span>
        </p>
        <p><strong>Domain:</strong> {complaint.domain_name}</p>
        <p><strong>Category:</strong> {complaint.category_name}</p>
      </section>

      <section>
        <h3>Description</h3>
        <p>{complaint.complaint_text}</p>
      </section>

      <div className="actions">
        {complaint.can_edit && (
          <button onClick={() => navigateToEdit(complaint.id)}>Edit</button>
        )}
      </div>
    </div>
  );
};
```

---

### 4️⃣ GET /api/complaints/count

**Purpose:** Get count of complaints matching current filters (for export preview).

**URL:** `GET http://127.0.0.1:8000/api/complaints/count`

**Query Parameters:** Same as `/api/complaints`, except pagination (`page`, `page_size`)

**Response Structure:**

```typescript
interface CountResponse {
  total_count: number;
  filters_applied: {
    search: string | null;
    status: string | null;
    // ... all filter parameters
  };
}
```

**React Integration Example:**

```typescript
const ExportPreview = ({ filters }) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const fetchCount = async () => {
      try {
        const params = { ...filters };
        // Remove null/empty values
        Object.keys(params).forEach(key => {
          if (params[key] === null || params[key] === '') delete params[key];
        });

        const response = await axios.get('http://127.0.0.1:8000/api/complaints/count', { params });
        setCount(response.data.total_count);
      } catch (error) {
        console.error('Error fetching count:', error);
      }
    };

    fetchCount();
  }, [filters]);

  return (
    <div className="export-preview">
      <p>{count} records found with current filters</p>
      <button onClick={() => handleExport()}>Export to CSV</button>
    </div>
  );
};
```

---

### 5️⃣ POST /api/complaints/export

**Purpose:** Export filtered complaints as CSV or JSON.

**URL:** `POST http://127.0.0.1:8000/api/complaints/export`

**Request Body:**

```typescript
interface ExportRequest {
  format: 'csv' | 'json';
  filters: {
    search?: string;
    status?: string;
    issuing_dept_id?: number;
    // ... any filter from GET /api/complaints
  };
  columns: string[]; // Column names to include
  include_patient_identifiers: boolean;
  language: 'en' | 'ar';
}
```

**Response Structure:**

```typescript
interface ExportResponse {
  export_id: string;
  file_name: string;
  file_size_bytes: number;
  download_url: string;
  record_count: number;
  generated_at: string; // ISO 8601
  expires_at: string; // ISO 8601
  audit_logged: boolean;
  status: 'pending' | 'processing' | 'completed' | 'failed';
}
```

**React Integration Example:**

```typescript
const ExportButton = ({ filters }) => {
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    setExporting(true);
    try {
      const response = await axios.post('http://127.0.0.1:8000/api/complaints/export', {
        format: 'csv',
        filters: filters,
        columns: [
          'complaint_number',
          'received_date',
          'patient_name',
          'issuing_dept_name',
          'domain_name',
          'severity_name',
          'status'
        ],
        include_patient_identifiers: false,
        language: 'en'
      });

      // Open download URL in new tab
      window.open(`http://127.0.0.1:8000${response.data.download_url}`, '_blank');
      
      // Show success toast
      alert(`Export created: ${response.data.file_name} (${response.data.record_count} records)`);
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed. Please try again.');
    } finally {
      setExporting(false);
    }
  };

  return (
    <button onClick={handleExport} disabled={exporting}>
      {exporting ? 'Exporting...' : 'Export to CSV'}
    </button>
  );
};
```

---

### 6️⃣ GET /api/complaints/views

**Purpose:** Get predefined table view configurations.

**URL:** `GET http://127.0.0.1:8000/api/complaints/views`

**Response Structure:**

```typescript
interface ViewsResponse {
  views: View[];
  default_view: string;
}

interface View {
  view_id: string;
  view_name: string;
  view_name_ar: string;
  columns: string[];
  default_sort: string;
  default_sort_order: 'asc' | 'desc';
  preset_filters?: {
    is_red_flag?: boolean;
    // ... other preset filters
  };
}
```

**React Integration Example:**

```typescript
const ViewSelector = ({ onViewChange }) => {
  const [views, setViews] = useState([]);
  const [selectedView, setSelectedView] = useState('complete');

  useEffect(() => {
    const fetchViews = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/api/complaints/views');
        setViews(response.data.views);
        setSelectedView(response.data.default_view);
      } catch (error) {
        console.error('Error fetching views:', error);
      }
    };

    fetchViews();
  }, []);

  const handleViewChange = (viewId) => {
    const view = views.find(v => v.view_id === viewId);
    setSelectedView(viewId);
    onViewChange(view);
  };

  return (
    <div className="view-selector">
      <label>View:</label>
      <select value={selectedView} onChange={(e) => handleViewChange(e.target.value)}>
        {views.map((view) => (
          <option key={view.view_id} value={view.view_id}>
            {view.view_name}
          </option>
        ))}
      </select>
    </div>
  );
};
```

---

## Data Models & Field Descriptions

### Patient Data Masking Rules

**In Table View (GET /api/complaints):**
- `patient_mrn`: `"MRN-***789"` (last 3 digits only)
- `patient_name`: `"Ahmed H."` (first name + last initial)

**In Single Record View (GET /api/complaints/{id}):**
- `patient_mrn`: `"MRN-123456789"` (full MRN)
- `patient_name`: `"Ahmed Hassan"` (full name)

**Privacy Warning:** Always show a warning banner when displaying unmasked patient data!

---

### Status Icons

Use these emoji icons for visual status display:

```typescript
const STATUS_ICONS = {
  open: '🟢',
  in_progress: '🟡',
  closed: '🔴'
};
```

---

### Severity Colors

Use `severity_color` field for visual highlighting:

```typescript
<span style={{ 
  color: complaint.severity_color,
  fontWeight: 'bold'
}}>
  {complaint.severity_name}
</span>
```

---

## Frontend Implementation Checklist

### Phase 1: Basic Table Display ⭐⭐⭐ HIGH PRIORITY

- [ ] 1. Create TableView page component
- [ ] 2. Fetch complaints from `GET /api/complaints`
- [ ] 3. Display complaints in table with key columns:
  - Complaint Number
  - Received Date
  - Patient Name (masked)
  - Issuing Department
  - Domain
  - Severity (with color)
  - Status (with icon)
  - Actions (View/Edit buttons)
- [ ] 4. Implement pagination (Previous/Next buttons)
- [ ] 5. Show total record count and page info

### Phase 2: Search & Filters ⭐⭐⭐ HIGH PRIORITY

- [ ] 6. Add search input (free-text search)
- [ ] 7. Fetch filter options from `GET /api/complaints/filter-options`
- [ ] 8. Add filter dropdowns:
  - Issuing Department
  - Target Department
  - Source
  - Status
  - Severity
  - Domain
- [ ] 9. Implement filter state management
- [ ] 10. Reset to page 1 on filter change
- [ ] 11. Add "Clear All Filters" button

### Phase 3: Sorting ⭐⭐ MEDIUM PRIORITY

- [ ] 12. Add sortable table headers (click to sort)
- [ ] 13. Show sort indicator (▲▼) on active column
- [ ] 14. Default sort: received_date DESC

### Phase 4: Single Record View ⭐⭐ MEDIUM PRIORITY

- [ ] 15. Create modal or detail page for single complaint
- [ ] 16. Fetch full details from `GET /api/complaints/{id}`
- [ ] 17. Display unmasked patient data with privacy warning
- [ ] 18. Add "Edit" button (if `can_edit === true`)
- [ ] 19. Handle 404 error (complaint not found)

### Phase 5: Export ⭐ LOW PRIORITY

- [ ] 20. Add export section with record count
- [ ] 21. Fetch count from `GET /api/complaints/count`
- [ ] 22. Add "Export to CSV" button
- [ ] 23. Call `POST /api/complaints/export` on click
- [ ] 24. Open download URL in new tab
- [ ] 25. Show success/error messages

### Phase 6: View Configurations ⭐ LOW PRIORITY

- [ ] 26. Fetch views from `GET /api/complaints/views`
- [ ] 27. Add view selector dropdown
- [ ] 28. Apply column configuration based on selected view
- [ ] 29. Apply preset filters (e.g., red_flags_only)

### Phase 7: Polish & UX ⭐ LOW PRIORITY

- [ ] 30. Add loading indicators (skeleton screens)
- [ ] 31. Add empty state ("No complaints found")
- [ ] 32. Add error handling with user-friendly messages
- [ ] 33. Add page size selector (10, 25, 50, 100)
- [ ] 34. Add "Go to page" input
- [ ] 35. Persist filters in URL query params (for shareable links)
- [ ] 36. Add tooltips for truncated text
- [ ] 37. Highlight search matches in table

---

## Error Handling

### Common Errors & Solutions

#### 400 Bad Request

**Cause:** Invalid parameter values

**Example Response:**
```json
{
  "error": "invalid_parameters",
  "message": "Page size must be between 1 and 500",
  "message_ar": "يجب أن يكون حجم الصفحة بين 1 و 500"
}
```

**Frontend Handling:**
```typescript
try {
  const response = await axios.get('/api/complaints', { params });
} catch (error) {
  if (error.response?.status === 400) {
    alert(error.response.data.message);
    // Reset invalid parameter
  }
}
```

---

#### 404 Not Found

**Cause:** Complaint with specified ID doesn't exist

**Example Response:**
```json
{
  "error": "complaint_not_found",
  "message": "Complaint with ID 1234 not found",
  "message_ar": "لم يتم العثور على الشكوى ذات المعرف 1234"
}
```

**Frontend Handling:**
```typescript
try {
  const response = await axios.get(`/api/complaints/${id}`);
} catch (error) {
  if (error.response?.status === 404) {
    // Show "Complaint not found" message
    // Redirect back to table view
  }
}
```

---

#### 500 Internal Server Error

**Cause:** Backend error (database connection, SQL error, etc.)

**Frontend Handling:**
```typescript
try {
  const response = await axios.get('/api/complaints');
} catch (error) {
  if (error.response?.status === 500) {
    alert('An error occurred on the server. Please try again later.');
    // Log to error tracking service (Sentry, etc.)
  }
}
```

---

## Testing Checklist

### Manual Testing Steps

#### ✅ Step 1: Test Basic Table Load
1. Navigate to TableView page
2. Verify complaints table loads with default settings (page 1, 50 records)
3. Verify pagination info shows correctly
4. Verify all columns display correctly

#### ✅ Step 2: Test Pagination
1. Click "Next" button → Verify page 2 loads
2. Click "Previous" button → Verify page 1 loads
3. Change page size to 10 → Verify 10 records display
4. Navigate to last page → Verify "Next" button is disabled

#### ✅ Step 3: Test Search
1. Enter complaint number in search → Verify matching results
2. Enter patient name in search → Verify matching results
3. Enter gibberish → Verify empty state or no results
4. Clear search → Verify full list returns

#### ✅ Step 4: Test Filters
1. Select status = "open" → Verify only open complaints show
2. Select issuing department → Verify filtering works
3. Apply multiple filters → Verify AND logic (all filters apply)
4. Clear filters → Verify full list returns

#### ✅ Step 5: Test Sorting
1. Click "Received Date" header → Verify sort order toggles
2. Click "Severity" header → Verify severity sorting works
3. Verify sort indicator (▲▼) shows on active column

#### ✅ Step 6: Test Single Complaint View
1. Click "View" on a complaint row
2. Verify full details modal/page opens
3. Verify unmasked patient data displays
4. Verify privacy warning is shown
5. Click "Edit" (if available) → Verify navigation to EditRecord

#### ✅ Step 7: Test Export
1. Apply filters
2. Verify export preview shows correct count
3. Click "Export to CSV"
4. Verify download link opens
5. Verify CSV file contains correct data

#### ✅ Step 8: Test View Selector
1. Select "Simplified" view
2. Verify only simplified columns show
3. Select "Red Flags Only" view
4. Verify red flag filter is auto-applied
5. Switch back to "Complete" view

#### ✅ Step 9: Test Error Handling
1. Enter invalid page number (e.g., page 999999)
2. Verify error message or redirect to valid page
3. Try to view non-existent complaint ID
4. Verify 404 error message displays

#### ✅ Step 10: Test Loading States
1. Throttle network in DevTools
2. Verify loading indicator shows during fetch
3. Verify skeleton screens or spinners display

---

## Integration Notes

### URL Query Parameter Persistence

Consider persisting filters in URL for shareable links:

```typescript
const TableView = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  useEffect(() => {
    // Load filters from URL on mount
    const filtersFromUrl = {
      status: searchParams.get('status'),
      issuing_dept_id: searchParams.get('issuing_dept_id'),
      page: parseInt(searchParams.get('page')) || 1
    };
    setFilters(filtersFromUrl);
  }, []);

  const handleFilterChange = (name, value) => {
    const newFilters = { ...filters, [name]: value };
    setFilters(newFilters);
    
    // Update URL
    const newParams = new URLSearchParams();
    Object.entries(newFilters).forEach(([key, val]) => {
      if (val) newParams.set(key, val);
    });
    setSearchParams(newParams);
  };
};
```

---

### Performance Optimization

1. **Debounce Search Input:**
   ```typescript
   import { debounce } from 'lodash';

   const debouncedSearch = useCallback(
     debounce((searchTerm) => {
       setFilters({ ...filters, search: searchTerm });
     }, 500),
     []
   );
   ```

2. **Memoize Table Rows:**
   ```typescript
   const TableRow = React.memo(({ complaint }) => {
     // ...
   });
   ```

3. **Virtual Scrolling (for large page sizes):**
   Consider using `react-window` or `react-virtualized` for page_size > 100

---

### Accessibility

1. Add ARIA labels to filter dropdowns
2. Add keyboard navigation for table rows
3. Add screen reader announcements for pagination changes
4. Ensure color contrast meets WCAG AA standards

---

## 🚀 Quick Start Summary

1. **Fetch complaints:** `GET /api/complaints?page=1&page_size=50`
2. **Fetch filter options:** `GET /api/complaints/filter-options?include_counts=true`
3. **Render table** with complaints data
4. **Implement filters** that update API parameters
5. **Implement pagination** that updates `page` parameter
6. **Handle row click** to navigate to detail view or EditRecord

---

## 📞 Backend Support

**All 6 endpoints are implemented and tested.**

Test URLs and examples available in:
`backend/TEST_URLS_TableView.md`

Interactive API docs:
`http://127.0.0.1:8000/docs`

---

**Good luck with frontend implementation! 🎉**
