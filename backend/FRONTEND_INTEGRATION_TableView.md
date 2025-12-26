# TableView Page - Frontend Integration Guide

## Overview
This document provides all necessary information for implementing the TableView page frontend that consumes the backend API endpoints.

**Backend Base URL:** `http://127.0.0.1:8000`

---

## API Endpoints

### 1. GET `/api/complaints` - Fetch Paginated Complaints

**Purpose:** Main endpoint for displaying the complaints table with filtering, searching, sorting, and pagination.

**Query Parameters:**
- `search` (string, optional): Free-text search across complaint number, patient name, and complaint text
- `issuing_org_unit_id` (integer, optional): Filter by organizational unit (Administration/Department/Section)
- `domain_id` (integer, optional): Filter by HCAT domain
- `category_id` (integer, optional): Filter by category
- `severity_id` (integer, optional): Filter by severity level
- `stage_id` (integer, optional): Filter by case stage
- `harm_level_id` (integer, optional): Filter by harm level
- `case_status_id` (integer, optional): Filter by case status
- `year` (integer, optional): Filter by year (YYYY)
- `month` (integer, optional): Filter by month (1-12)
- `start_date` (string, optional): Filter by received_date >= start_date (YYYY-MM-DD)
- `end_date` (string, optional): Filter by received_date <= end_date (YYYY-MM-DD)
- `sort_by` (string, default: "FeedbackRecievedDate"): Column to sort by
- `sort_order` (string, default: "desc"): Sort order ("asc" or "desc")
- `page` (integer, default: 1): Page number (1-indexed)
- `page_size` (integer, default: 50): Results per page (1-500)
- `view` (string, default: "complete"): View preset ("complete" or "simplified")

**Response Example:**
```json
{
  "complaints": [
    {
      "id": 123,
      "complaint_number": 123,
      "complaint_summary": "Patient complained about waiting time in emergency department...",
      "complaint_text": "Full complaint text here...",
      "received_date": "2024-12-15",
      "created_at": "2024-12-15T10:30:00",
      "patient_name": "أحمد محمد",
      "doctor_name": "د. سارة أحمد",
      "doctor_id": 45,
      "issuing_org_unit_id": 12,
      "issuing_org_unit_name": "Emergency Department",
      "domain_id": 1,
      "domain_name": "Clinical Care",
      "category_id": 5,
      "category_name": "Patient Safety",
      "subcategory_id": 15,
      "classification_id": 42,
      "severity_id": 2,
      "severity_name": "Medium",
      "stage_id": 1,
      "stage_name": "Initial Review",
      "harm_level_id": 1,
      "harm_level": "No Harm",
      "case_status_id": 1,
      "status_name": "Open",
      "immediate_action": "Patient was attended immediately",
      "taken_action": "Staff counseled about triage protocols",
      "in_out": "Inpatient",
      "created_by_user_id": 5
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 50,
    "total_records": 324,
    "total_pages": 7
  },
  "filters_applied": {
    "search": null,
    "issuing_org_unit_id": null,
    "domain_id": 1,
    "category_id": null,
    "severity_id": null,
    "stage_id": null,
    "harm_level_id": null,
    "case_status_id": null,
    "year": null,
    "month": null,
    "start_date": null,
    "end_date": null
  },
  "view": "complete"
}
```

---

### 2. GET `/api/complaints/filter-options` - Fetch Filter Dropdown Options

**Purpose:** Get all available options for filter dropdowns (organizational units, domains, categories, severities, etc.)

**Query Parameters:**
- `include_counts` (boolean, default: false): If true, includes record count for each option

**Response Example:**
```json
{
  "issuing_org_units": [
    {
      "id": 1,
      "name": "Emergency Department",
      "parent_id": 1,
      "count": 45
    },
    {
      "id": 2,
      "name": "Surgery Department",
      "parent_id": 1,
      "count": 32
    }
  ],
  "domains": [
    {
      "id": 1,
      "name": "Clinical Care",
      "count": 120
    },
    {
      "id": 2,
      "name": "Management & Administration",
      "count": 78
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "Patient Safety",
      "domain_id": 1,
      "count": 65
    },
    {
      "id": 2,
      "name": "Medical Errors",
      "domain_id": 1,
      "count": 42
    }
  ],
  "severities": [
    {
      "id": 1,
      "name": "Low",
      "count": 150
    },
    {
      "id": 2,
      "name": "Medium",
      "count": 80
    },
    {
      "id": 3,
      "name": "High",
      "count": 25
    }
  ],
  "stages": [
    {
      "id": 1,
      "name": "Initial Review",
      "count": 100
    },
    {
      "id": 2,
      "name": "Investigation",
      "count": 75
    }
  ],
  "harm_levels": [
    {
      "id": 1,
      "name": "No Harm",
      "count": 180
    },
    {
      "id": 2,
      "name": "Minor Harm",
      "count": 50
    }
  ],
  "statuses": [
    {
      "id": 1,
      "name": "Open",
      "count": 120
    },
    {
      "id": 2,
      "name": "In Progress",
      "count": 85
    },
    {
      "id": 3,
      "name": "Closed",
      "count": 50
    }
  ]
}
```

---

### 3. GET `/api/complaints/{id}` - Fetch Single Complaint

**Purpose:** Get full details of a specific complaint (for detail view/edit modal)

**Path Parameters:**
- `id` (integer, required): The complaint ID (IncidentRequestCaseID)

**Response Example:**
```json
{
  "id": 123,
  "complaint_text": "Full complaint description here...",
  "immediate_action": "Action taken immediately",
  "taken_action": "Follow-up actions",
  "received_date": "2024-12-15",
  "patient_name": "أحمد محمد عبدالله",
  "doctor_name": "د. سارة أحمد",
  "doctor_id": 45,
  "in_out": "Inpatient",
  "issuing_org_unit_id": 12,
  "issuing_org_unit_name": "Emergency Department",
  "domain_id": 1,
  "domain_name": "Clinical Care",
  "category_id": 5,
  "category_name": "Patient Safety",
  "subcategory_id": 15,
  "classification_id": 42,
  "severity_id": 2,
  "severity_name": "Medium",
  "stage_id": 1,
  "stage_name": "Initial Review",
  "harm_level_id": 1,
  "harm_level": "No Harm",
  "case_status_id": 1,
  "status_name": "Open",
  "building_id": 3,
  "clinical_risk_type_id": 2,
  "feedback_intent_type_id": 1,
  "created_at": "2024-12-15T10:30:00",
  "created_by_user_id": 5
}
```

**Error Response (404):**
```json
{
  "error": "complaint_not_found",
  "message": "Complaint with ID 999 not found",
  "message_ar": "لم يتم العثور على الشكوى ذات المعرف 999"
}
```

---

### 4. GET `/api/complaints/count` - Get Filtered Count

**Purpose:** Get count of complaints matching current filters (useful for export preview)

**Query Parameters:** Same as `/api/complaints` endpoint (all filter parameters)

**Response Example:**
```json
{
  "total_count": 324,
  "filters_applied": {
    "search": "emergency",
    "domain_id": 1,
    "severity_id": 2
  }
}
```

---

### 5. POST `/api/complaints/export` - Generate Export

**Purpose:** Generate export metadata for downloading complaints data

**Request Body:**
```json
{
  "format": "csv",
  "filters": {
    "domain_id": 1,
    "severity_id": 2,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  },
  "columns": [
    "complaint_number",
    "received_date",
    "patient_name",
    "domain_name",
    "severity_name",
    "status_name"
  ],
  "include_patient_identifiers": false,
  "language": "ar"
}
```

**Response Example:**
```json
{
  "export_id": "exp_20241215_103045_abc123",
  "format": "csv",
  "estimated_size_mb": 2.5,
  "estimated_rows": 324,
  "columns": ["complaint_number", "received_date", "patient_name"],
  "filters_applied": {
    "domain_id": 1,
    "severity_id": 2
  },
  "created_at": "2024-12-15T10:30:45",
  "status": "ready",
  "download_url": "/api/exports/exp_20241215_103045_abc123/download"
}
```

---

### 6. GET `/api/complaints/views` - Get Table View Configurations

**Purpose:** Get predefined view configurations (column visibility, default filters)

**Response Example:**
```json
{
  "views": [
    {
      "id": "complete",
      "name": "Complete View",
      "name_ar": "العرض الكامل",
      "columns": [
        "complaint_number",
        "received_date",
        "patient_name",
        "issuing_org_unit_name",
        "domain_name",
        "category_name",
        "severity_name",
        "stage_name",
        "harm_level",
        "status_name"
      ],
      "default_sort": "received_date",
      "default_sort_order": "desc"
    },
    {
      "id": "simplified",
      "name": "Simplified View",
      "name_ar": "العرض المبسط",
      "columns": [
        "complaint_number",
        "received_date",
        "complaint_summary",
        "severity_name",
        "status_name"
      ],
      "default_sort": "received_date",
      "default_sort_order": "desc"
    }
  ]
}
```

---

## TypeScript Interfaces

```typescript
// Core Data Types
interface Complaint {
  id: number;
  complaint_number: number;
  complaint_summary: string;
  complaint_text: string;
  received_date: string;
  created_at: string;
  patient_name: string;
  doctor_name: string | null;
  doctor_id: number | null;
  issuing_org_unit_id: number;
  issuing_org_unit_name: string;
  domain_id: number;
  domain_name: string;
  category_id: number;
  category_name: string;
  subcategory_id: number | null;
  classification_id: number | null;
  severity_id: number | null;
  severity_name: string | null;
  stage_id: number;
  stage_name: string;
  harm_level_id: number;
  harm_level: string;
  case_status_id: number;
  status_name: string;
  immediate_action: string | null;
  taken_action: string | null;
  in_out: string | null;
  created_by_user_id: number;
}

interface ComplaintDetail extends Complaint {
  building_id: number | null;
  clinical_risk_type_id: number;
  feedback_intent_type_id: number;
}

// Filter Options
interface FilterOption {
  id: number;
  name: string;
  count?: number;
}

interface CategoryFilterOption extends FilterOption {
  domain_id: number;
}

interface OrgUnitFilterOption extends FilterOption {
  parent_id: number;
}

interface FilterOptions {
  issuing_org_units: OrgUnitFilterOption[];
  domains: FilterOption[];
  categories: CategoryFilterOption[];
  severities: FilterOption[];
  stages: FilterOption[];
  harm_levels: FilterOption[];
  statuses: FilterOption[];
}

// API Request/Response Types
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
  view?: 'complete' | 'simplified';
}

interface PaginationMetadata {
  page: number;
  page_size: number;
  total_records: number;
  total_pages: number;
}

interface ComplaintsResponse {
  complaints: Complaint[];
  pagination: PaginationMetadata;
  filters_applied: Record<string, any>;
  view: string;
}

interface ExportRequest {
  format: 'csv' | 'json';
  filters: Record<string, any>;
  columns: string[];
  include_patient_identifiers: boolean;
  language: 'en' | 'ar';
}

interface ExportResponse {
  export_id: string;
  format: string;
  estimated_size_mb: number;
  estimated_rows: number;
  columns: string[];
  filters_applied: Record<string, any>;
  created_at: string;
  status: string;
  download_url: string;
}

interface TableView {
  id: string;
  name: string;
  name_ar: string;
  columns: string[];
  default_sort: string;
  default_sort_order: 'asc' | 'desc';
}

interface TableViewsResponse {
  views: TableView[];
}
```

---

## React/Next.js Implementation Examples

### 1. API Service Layer (services/complaintsApi.ts)

```typescript
import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

export const complaintsApi = {
  // Fetch paginated complaints
  async getComplaints(params: ComplaintsQueryParams): Promise<ComplaintsResponse> {
    const response = await axios.get(`${API_BASE_URL}/api/complaints`, { params });
    return response.data;
  },

  // Fetch filter options
  async getFilterOptions(includeCounts = false): Promise<FilterOptions> {
    const response = await axios.get(`${API_BASE_URL}/api/complaints/filter-options`, {
      params: { include_counts: includeCounts }
    });
    return response.data;
  },

  // Fetch single complaint
  async getComplaintById(id: number): Promise<ComplaintDetail> {
    const response = await axios.get(`${API_BASE_URL}/api/complaints/${id}`);
    return response.data;
  },

  // Get filtered count
  async getComplaintsCount(params: Omit<ComplaintsQueryParams, 'page' | 'page_size' | 'sort_by' | 'sort_order' | 'view'>): Promise<{ total_count: number }> {
    const response = await axios.get(`${API_BASE_URL}/api/complaints/count`, { params });
    return response.data;
  },

  // Export complaints
  async exportComplaints(request: ExportRequest): Promise<ExportResponse> {
    const response = await axios.post(`${API_BASE_URL}/api/complaints/export`, request);
    return response.data;
  },

  // Get table views
  async getTableViews(): Promise<TableViewsResponse> {
    const response = await axios.get(`${API_BASE_URL}/api/complaints/views`);
    return response.data;
  }
};
```

---

### 2. Main TableView Component (components/TableView.tsx)

```typescript
'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { complaintsApi } from '@/services/complaintsApi';
import { ComplaintsQueryParams, Complaint, FilterOptions } from '@/types';
import { DataTable } from './DataTable';
import { FilterPanel } from './FilterPanel';
import { SearchBar } from './SearchBar';
import { Pagination } from './Pagination';

export function TableView() {
  const [complaints, setComplaints] = useState<Complaint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [pagination, setPagination] = useState({
    page: 1,
    page_size: 50,
    total_records: 0,
    total_pages: 0
  });

  // Filter state
  const [filters, setFilters] = useState<ComplaintsQueryParams>({
    page: 1,
    page_size: 50,
    sort_by: 'FeedbackRecievedDate',
    sort_order: 'desc',
    view: 'complete'
  });

  // Fetch filter options on mount
  useEffect(() => {
    async function loadFilterOptions() {
      try {
        const options = await complaintsApi.getFilterOptions(true);
        setFilterOptions(options);
      } catch (err) {
        console.error('Failed to load filter options:', err);
      }
    }
    loadFilterOptions();
  }, []);

  // Fetch complaints whenever filters change
  const fetchComplaints = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await complaintsApi.getComplaints(filters);
      setComplaints(response.complaints);
      setPagination(response.pagination);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to load complaints');
      console.error('Error fetching complaints:', err);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    fetchComplaints();
  }, [fetchComplaints]);

  // Handle filter changes
  const handleFilterChange = (newFilters: Partial<ComplaintsQueryParams>) => {
    setFilters(prev => ({
      ...prev,
      ...newFilters,
      page: 1 // Reset to first page when filters change
    }));
  };

  // Handle search
  const handleSearch = (searchTerm: string) => {
    handleFilterChange({ search: searchTerm || undefined });
  };

  // Handle pagination
  const handlePageChange = (newPage: number) => {
    setFilters(prev => ({ ...prev, page: newPage }));
  };

  // Handle sort
  const handleSort = (column: string, order: 'asc' | 'desc') => {
    setFilters(prev => ({
      ...prev,
      sort_by: column,
      sort_order: order
    }));
  };

  // Handle row click (navigate to detail view)
  const handleRowClick = (complaint: Complaint) => {
    // Navigate to detail page or open modal
    window.location.href = `/complaints/${complaint.id}`;
  };

  return (
    <div className="table-view-container">
      <div className="table-view-header">
        <h1>شكاوى المرضى</h1>
        <SearchBar onSearch={handleSearch} />
      </div>

      <FilterPanel
        filters={filters}
        filterOptions={filterOptions}
        onFilterChange={handleFilterChange}
      />

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <DataTable
        complaints={complaints}
        loading={loading}
        sortBy={filters.sort_by}
        sortOrder={filters.sort_order}
        onSort={handleSort}
        onRowClick={handleRowClick}
      />

      <Pagination
        currentPage={pagination.page}
        pageSize={pagination.page_size}
        totalRecords={pagination.total_records}
        totalPages={pagination.total_pages}
        onPageChange={handlePageChange}
      />
    </div>
  );
}
```

---

### 3. Filter Panel Component (components/FilterPanel.tsx)

```typescript
import React from 'react';
import { ComplaintsQueryParams, FilterOptions } from '@/types';

interface FilterPanelProps {
  filters: ComplaintsQueryParams;
  filterOptions: FilterOptions | null;
  onFilterChange: (filters: Partial<ComplaintsQueryParams>) => void;
}

export function FilterPanel({ filters, filterOptions, onFilterChange }: FilterPanelProps) {
  if (!filterOptions) return <div>Loading filters...</div>;

  return (
    <div className="filter-panel">
      <div className="filter-row">
        {/* Organizational Unit Filter */}
        <div className="filter-group">
          <label>الوحدة التنظيمية</label>
          <select
            value={filters.issuing_org_unit_id || ''}
            onChange={(e) => onFilterChange({
              issuing_org_unit_id: e.target.value ? parseInt(e.target.value) : undefined
            })}
          >
            <option value="">الكل</option>
            {filterOptions.issuing_org_units.map(unit => (
              <option key={unit.id} value={unit.id}>
                {unit.name} {unit.count !== undefined && `(${unit.count})`}
              </option>
            ))}
          </select>
        </div>

        {/* Domain Filter */}
        <div className="filter-group">
          <label>المجال</label>
          <select
            value={filters.domain_id || ''}
            onChange={(e) => onFilterChange({
              domain_id: e.target.value ? parseInt(e.target.value) : undefined
            })}
          >
            <option value="">الكل</option>
            {filterOptions.domains.map(domain => (
              <option key={domain.id} value={domain.id}>
                {domain.name} {domain.count !== undefined && `(${domain.count})`}
              </option>
            ))}
          </select>
        </div>

        {/* Category Filter */}
        <div className="filter-group">
          <label>التصنيف</label>
          <select
            value={filters.category_id || ''}
            onChange={(e) => onFilterChange({
              category_id: e.target.value ? parseInt(e.target.value) : undefined
            })}
          >
            <option value="">الكل</option>
            {filterOptions.categories.map(category => (
              <option key={category.id} value={category.id}>
                {category.name} {category.count !== undefined && `(${category.count})`}
              </option>
            ))}
          </select>
        </div>

        {/* Severity Filter */}
        <div className="filter-group">
          <label>مستوى الخطورة</label>
          <select
            value={filters.severity_id || ''}
            onChange={(e) => onFilterChange({
              severity_id: e.target.value ? parseInt(e.target.value) : undefined
            })}
          >
            <option value="">الكل</option>
            {filterOptions.severities.map(severity => (
              <option key={severity.id} value={severity.id}>
                {severity.name} {severity.count !== undefined && `(${severity.count})`}
              </option>
            ))}
          </select>
        </div>

        {/* Stage Filter */}
        <div className="filter-group">
          <label>المرحلة</label>
          <select
            value={filters.stage_id || ''}
            onChange={(e) => onFilterChange({
              stage_id: e.target.value ? parseInt(e.target.value) : undefined
            })}
          >
            <option value="">الكل</option>
            {filterOptions.stages.map(stage => (
              <option key={stage.id} value={stage.id}>
                {stage.name} {stage.count !== undefined && `(${stage.count})`}
              </option>
            ))}
          </select>
        </div>

        {/* Status Filter */}
        <div className="filter-group">
          <label>الحالة</label>
          <select
            value={filters.case_status_id || ''}
            onChange={(e) => onFilterChange({
              case_status_id: e.target.value ? parseInt(e.target.value) : undefined
            })}
          >
            <option value="">الكل</option>
            {filterOptions.statuses.map(status => (
              <option key={status.id} value={status.id}>
                {status.name} {status.count !== undefined && `(${status.count})`}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="filter-row">
        {/* Date Range Filters */}
        <div className="filter-group">
          <label>من تاريخ</label>
          <input
            type="date"
            value={filters.start_date || ''}
            onChange={(e) => onFilterChange({ start_date: e.target.value || undefined })}
          />
        </div>

        <div className="filter-group">
          <label>إلى تاريخ</label>
          <input
            type="date"
            value={filters.end_date || ''}
            onChange={(e) => onFilterChange({ end_date: e.target.value || undefined })}
          />
        </div>

        {/* Clear Filters Button */}
        <button
          onClick={() => onFilterChange({
            issuing_org_unit_id: undefined,
            domain_id: undefined,
            category_id: undefined,
            severity_id: undefined,
            stage_id: undefined,
            harm_level_id: undefined,
            case_status_id: undefined,
            year: undefined,
            month: undefined,
            start_date: undefined,
            end_date: undefined,
            search: undefined
          })}
        >
          مسح الفلاتر
        </button>
      </div>
    </div>
  );
}
```

---

### 4. Data Table Component (components/DataTable.tsx)

```typescript
import React from 'react';
import { Complaint } from '@/types';

interface DataTableProps {
  complaints: Complaint[];
  loading: boolean;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  onSort: (column: string, order: 'asc' | 'desc') => void;
  onRowClick: (complaint: Complaint) => void;
}

export function DataTable({ complaints, loading, sortBy, sortOrder, onSort, onRowClick }: DataTableProps) {
  const handleHeaderClick = (column: string) => {
    const newOrder = sortBy === column && sortOrder === 'asc' ? 'desc' : 'asc';
    onSort(column, newOrder);
  };

  const getSortIcon = (column: string) => {
    if (sortBy !== column) return '⇅';
    return sortOrder === 'asc' ? '↑' : '↓';
  };

  if (loading) {
    return <div className="loading">جاري التحميل...</div>;
  }

  if (complaints.length === 0) {
    return <div className="no-data">لا توجد شكاوى</div>;
  }

  return (
    <div className="data-table-container">
      <table className="data-table">
        <thead>
          <tr>
            <th onClick={() => handleHeaderClick('IncidentRequestCaseID')}>
              رقم الشكوى {getSortIcon('IncidentRequestCaseID')}
            </th>
            <th onClick={() => handleHeaderClick('FeedbackRecievedDate')}>
              تاريخ الاستلام {getSortIcon('FeedbackRecievedDate')}
            </th>
            <th>اسم المريض</th>
            <th>الوحدة التنظيمية</th>
            <th>المجال</th>
            <th>التصنيف</th>
            <th>مستوى الخطورة</th>
            <th>المرحلة</th>
            <th>الحالة</th>
            <th>ملخص الشكوى</th>
          </tr>
        </thead>
        <tbody>
          {complaints.map((complaint) => (
            <tr
              key={complaint.id}
              onClick={() => onRowClick(complaint)}
              className="clickable-row"
            >
              <td>{complaint.complaint_number}</td>
              <td>{complaint.received_date}</td>
              <td>{complaint.patient_name}</td>
              <td>{complaint.issuing_org_unit_name}</td>
              <td>{complaint.domain_name}</td>
              <td>{complaint.category_name}</td>
              <td>
                <span className={`severity-badge severity-${complaint.severity_id}`}>
                  {complaint.severity_name}
                </span>
              </td>
              <td>{complaint.stage_name}</td>
              <td>
                <span className={`status-badge status-${complaint.case_status_id}`}>
                  {complaint.status_name}
                </span>
              </td>
              <td className="complaint-summary">{complaint.complaint_summary}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

---

### 5. Pagination Component (components/Pagination.tsx)

```typescript
import React from 'react';

interface PaginationProps {
  currentPage: number;
  pageSize: number;
  totalRecords: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

export function Pagination({
  currentPage,
  pageSize,
  totalRecords,
  totalPages,
  onPageChange
}: PaginationProps) {
  const startRecord = (currentPage - 1) * pageSize + 1;
  const endRecord = Math.min(currentPage * pageSize, totalRecords);

  return (
    <div className="pagination">
      <div className="pagination-info">
        عرض {startRecord} - {endRecord} من أصل {totalRecords} سجل
      </div>

      <div className="pagination-controls">
        <button
          onClick={() => onPageChange(1)}
          disabled={currentPage === 1}
        >
          الأولى
        </button>
        
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
        >
          السابق
        </button>

        <span className="page-numbers">
          {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
            let pageNum;
            if (totalPages <= 5) {
              pageNum = i + 1;
            } else if (currentPage <= 3) {
              pageNum = i + 1;
            } else if (currentPage >= totalPages - 2) {
              pageNum = totalPages - 4 + i;
            } else {
              pageNum = currentPage - 2 + i;
            }

            return (
              <button
                key={pageNum}
                onClick={() => onPageChange(pageNum)}
                className={currentPage === pageNum ? 'active' : ''}
              >
                {pageNum}
              </button>
            );
          })}
        </span>

        <button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
        >
          التالي
        </button>

        <button
          onClick={() => onPageChange(totalPages)}
          disabled={currentPage === totalPages}
        >
          الأخيرة
        </button>
      </div>
    </div>
  );
}
```

---

### 6. Search Bar Component (components/SearchBar.tsx)

```typescript
import React, { useState, useCallback } from 'react';
import { debounce } from 'lodash';

interface SearchBarProps {
  onSearch: (searchTerm: string) => void;
}

export function SearchBar({ onSearch }: SearchBarProps) {
  const [searchTerm, setSearchTerm] = useState('');

  // Debounce search to avoid excessive API calls
  const debouncedSearch = useCallback(
    debounce((term: string) => {
      onSearch(term);
    }, 500),
    [onSearch]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchTerm(value);
    debouncedSearch(value);
  };

  const handleClear = () => {
    setSearchTerm('');
    onSearch('');
  };

  return (
    <div className="search-bar">
      <input
        type="text"
        placeholder="ابحث في رقم الشكوى، اسم المريض، أو نص الشكوى..."
        value={searchTerm}
        onChange={handleChange}
      />
      {searchTerm && (
        <button onClick={handleClear} className="clear-button">
          ✕
        </button>
      )}
      <button className="search-button">
        🔍
      </button>
    </div>
  );
}
```

---

## Error Handling

All API endpoints return standardized error responses:

```typescript
interface ErrorResponse {
  error: string;
  message: string;
  message_ar: string;
}

// Example error handling in API calls
try {
  const response = await complaintsApi.getComplaints(params);
  // Handle success
} catch (error: any) {
  if (error.response) {
    // Server responded with error
    const errorData: ErrorResponse = error.response.data;
    console.error('API Error:', errorData.message);
    // Show user-friendly message (use message_ar for Arabic)
    alert(errorData.message_ar);
  } else if (error.request) {
    // Request made but no response
    console.error('Network Error:', error.request);
    alert('خطأ في الاتصال بالخادم');
  } else {
    // Something else happened
    console.error('Error:', error.message);
    alert('حدث خطأ غير متوقع');
  }
}
```

---

## Styling Recommendations

```css
/* Severity Badges */
.severity-badge {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
}

.severity-1 { background-color: #d4edda; color: #155724; } /* Low */
.severity-2 { background-color: #fff3cd; color: #856404; } /* Medium */
.severity-3 { background-color: #f8d7da; color: #721c24; } /* High */

/* Status Badges */
.status-badge {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
}

.status-1 { background-color: #cce5ff; color: #004085; } /* Open */
.status-2 { background-color: #fff3cd; color: #856404; } /* In Progress */
.status-3 { background-color: #d4edda; color: #155724; } /* Closed */

/* Table Styling */
.data-table {
  width: 100%;
  border-collapse: collapse;
  direction: rtl;
}

.data-table th {
  background-color: #f8f9fa;
  padding: 12px;
  text-align: right;
  font-weight: 600;
  cursor: pointer;
}

.data-table td {
  padding: 12px;
  border-bottom: 1px solid #dee2e6;
}

.clickable-row {
  cursor: pointer;
}

.clickable-row:hover {
  background-color: #f8f9fa;
}

.complaint-summary {
  max-width: 300px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
```

---

## Implementation Checklist

- [ ] Create API service layer with axios
- [ ] Define TypeScript interfaces for all data types
- [ ] Implement main TableView component with state management
- [ ] Create FilterPanel component with all filter dropdowns
- [ ] Implement DataTable component with sorting
- [ ] Create Pagination component
- [ ] Implement SearchBar with debouncing
- [ ] Add error handling for all API calls
- [ ] Style severity and status badges
- [ ] Implement row click navigation to detail view
- [ ] Add loading states and spinners
- [ ] Test with different filter combinations
- [ ] Test pagination navigation
- [ ] Test search functionality
- [ ] Test sorting on different columns
- [ ] Handle empty states (no data)
- [ ] Add responsive design for mobile
- [ ] Implement export functionality
- [ ] Add view switcher (complete/simplified)

---

## Notes

1. **RTL Support:** All Arabic text should be displayed right-to-left. Use `direction: rtl` in CSS.

2. **Date Formatting:** Backend returns dates as strings (YYYY-MM-DD). Format them as needed for display.

3. **Debouncing:** Implement search debouncing to avoid excessive API calls while user types.

4. **Error Messages:** Use `message_ar` field from error responses for Arabic error messages.

5. **Loading States:** Show loading indicators during API calls to improve UX.

6. **Patient Data Masking:** The main list endpoint returns masked patient data. Full details are only in the single complaint endpoint.

7. **Filter Persistence:** Consider saving filter state to localStorage or URL params for better UX.

8. **Export Flow:** The export endpoint returns metadata. Implement actual download separately.

9. **View Configurations:** Use the `/api/complaints/views` endpoint to get predefined column configurations.

10. **Performance:** Use React.memo and useCallback for optimization with large datasets.
