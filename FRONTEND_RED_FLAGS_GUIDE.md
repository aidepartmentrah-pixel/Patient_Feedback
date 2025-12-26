# Frontend Implementation Guide: Red Flags (Critical Issues) API

## Overview

This document provides comprehensive guidance for frontend developers to integrate the **Red Flags API** into the Critical Issues page. The Red Flags feature tracks high-risk incidents requiring immediate attention and governance follow-up.

---

## Table of Contents

1. [API Endpoints Summary](#api-endpoints-summary)
2. [TypeScript Interfaces](#typescript-interfaces)
3. [API Service Layer](#api-service-layer)
4. [React Components Examples](#react-components-examples)
5. [State Management](#state-management)
6. [UI Components Mapping](#ui-components-mapping)
7. [Error Handling](#error-handling)
8. [Testing URLs](#testing-urls)

---

## API Endpoints Summary

| Method | Endpoint | Purpose | Status |
|--------|----------|---------|--------|
| GET | `/api/red-flags` | Get filtered list of red flags | ✅ Ready |
| GET | `/api/red-flags/statistics` | Get KPI statistics | ✅ Ready |
| GET | `/api/red-flags/trends` | Get time-series trend data | ✅ Ready |
| GET | `/api/red-flags/{id}` | Get single red flag details | ✅ Ready |
| POST | `/api/red-flags/{id}/export-pdf` | Export single PDF | ⏳ Not Implemented |
| POST | `/api/red-flags/export-batch` | Batch export | ⏳ Not Implemented |
| GET | `/api/red-flags/test` | Health check | ✅ Ready |

**Base URL:** `http://127.0.0.1:8000`

---

## TypeScript Interfaces

### Core Types

```typescript
// RedFlag.types.ts

export interface RedFlag {
  red_flag_id: number;
  recordID: string; // Format: "RF-YYYY-NNN"
  patient_name: string;
  date_received: string; // ISO date string
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
  date_range: {
    from: string | null;
    to: string | null;
  };
}

export interface TrendDataPoint {
  period: string;
  count?: number; // When no grouping
  breakdown?: Record<string, number>; // When grouped
  total?: number; // When grouped
}

export interface RedFlagTrends {
  trends: TrendDataPoint[];
  date_range: {
    from: string | null;
    to: string | null;
  };
  granularity: 'monthly' | 'quarterly' | 'weekly';
  grouped_by?: 'category' | 'severity' | 'department' | 'none';
}

export interface RedFlagDetails {
  red_flag_id: number;
  recordID: string;
  patient_name: string;
  date_received: string;
  department: string;
  category: string;
  subcategory: string;
  severity: 'HIGH' | 'CRITICAL';
  status: 'OPEN' | 'UNDER_REVIEW' | 'FINISHED';
  isNeverEvent: boolean;
  incident_details: {
    complaint_text: string;
    immediate_action: string;
    actions_taken: string;
    root_cause: string;
    harm_level: string;
    stage: string;
  };
  timeline: Array<{
    date: string;
    event: string;
    details: string;
  }>;
  related_actions: Array<{
    action: string;
    responsible: string;
    deadline: string;
    status: string;
  }>;
}

export interface RedFlagFilters {
  search?: string;
  status?: 'OPEN' | 'UNDER_REVIEW' | 'FINISHED' | 'all';
  from_date?: string;
  to_date?: string;
  department?: string;
  category?: string;
  severity?: 'HIGH' | 'CRITICAL';
  is_never_event?: boolean;
  limit?: number;
  offset?: number;
}

export interface RedFlagTrendParams {
  from_date?: string;
  to_date?: string;
  granularity?: 'monthly' | 'quarterly' | 'weekly';
  group_by?: 'category' | 'severity' | 'department' | 'none';
}
```

---

## API Service Layer

### Create `redFlagsApi.ts`

```typescript
// services/api/redFlagsApi.ts

import axios, { AxiosError } from 'axios';
import {
  RedFlag,
  RedFlagsList,
  RedFlagStatistics,
  RedFlagTrends,
  RedFlagDetails,
  RedFlagFilters,
  RedFlagTrendParams
} from '../types/RedFlag.types';

const BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:8000';

// ==================== API Client ====================

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const errorData = error.response?.data as any;
    
    // Log error with Arabic message if available
    console.error('API Error:', {
      status: error.response?.status,
      message: errorData?.message || error.message,
      message_ar: errorData?.message_ar,
      error_code: errorData?.error
    });
    
    return Promise.reject(error);
  }
);

// ==================== API Functions ====================

/**
 * Fetch list of red flags with optional filters
 */
export async function fetchRedFlags(
  filters: RedFlagFilters = {}
): Promise<RedFlagsList> {
  const response = await api.get<RedFlagsList>('/api/red-flags', {
    params: filters
  });
  return response.data;
}

/**
 * Fetch red flags statistics/KPIs
 */
export async function fetchRedFlagsStatistics(
  from_date?: string,
  to_date?: string
): Promise<RedFlagStatistics> {
  const response = await api.get<RedFlagStatistics>('/api/red-flags/statistics', {
    params: { from_date, to_date }
  });
  return response.data;
}

/**
 * Fetch trend data for charts
 */
export async function fetchRedFlagsTrends(
  params: RedFlagTrendParams = {}
): Promise<RedFlagTrends> {
  const response = await api.get<RedFlagTrends>('/api/red-flags/trends', {
    params: {
      granularity: params.granularity || 'monthly',
      group_by: params.group_by || 'none',
      from_date: params.from_date,
      to_date: params.to_date
    }
  });
  return response.data;
}

/**
 * Fetch detailed information for a single red flag
 */
export async function fetchRedFlagDetails(
  id: number
): Promise<RedFlagDetails> {
  const response = await api.get<RedFlagDetails>(`/api/red-flags/${id}`);
  return response.data;
}

/**
 * Test API connectivity
 */
export async function testRedFlagsApi(): Promise<any> {
  const response = await api.get('/api/red-flags/test');
  return response.data;
}

// ==================== Error Handler ====================

export function handleRedFlagsApiError(error: any): string {
  if (axios.isAxiosError(error)) {
    const errorData = error.response?.data as any;
    
    // Return Arabic message if available
    if (errorData?.message_ar) {
      return errorData.message_ar;
    }
    
    // Return English message
    if (errorData?.message) {
      return errorData.message;
    }
    
    // Default messages based on status code
    switch (error.response?.status) {
      case 404:
        return 'لم يتم العثور على العلم الأحمر';
      case 500:
        return 'حدث خطأ في الخادم. يرجى المحاولة مرة أخرى';
      default:
        return 'حدث خطأ غير متوقع';
    }
  }
  
  return 'خطأ في الاتصال بالخادم';
}
```

---

## React Components Examples

### 1. Red Flags List Component

```typescript
// components/RedFlagsList.tsx

import React, { useState, useEffect } from 'react';
import { fetchRedFlags, handleRedFlagsApiError } from '../services/api/redFlagsApi';
import { RedFlag, RedFlagFilters } from '../types/RedFlag.types';

export const RedFlagsList: React.FC = () => {
  const [redFlags, setRedFlags] = useState<RedFlag[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Filters state
  const [filters, setFilters] = useState<RedFlagFilters>({
    status: 'all',
    limit: 50,
    offset: 0
  });

  useEffect(() => {
    loadRedFlags();
  }, [filters]);

  const loadRedFlags = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await fetchRedFlags(filters);
      
      setRedFlags(data.red_flags);
      setTotal(data.total);
    } catch (err) {
      setError(handleRedFlagsApiError(err));
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (newFilters: Partial<RedFlagFilters>) => {
    setFilters(prev => ({ ...prev, ...newFilters, offset: 0 }));
  };

  const handlePageChange = (newOffset: number) => {
    setFilters(prev => ({ ...prev, offset: newOffset }));
  };

  if (loading) return <div>جاري التحميل...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="red-flags-list">
      {/* Filters */}
      <div className="filters">
        <select
          value={filters.status}
          onChange={(e) => handleFilterChange({ status: e.target.value as any })}
        >
          <option value="all">جميع الحالات</option>
          <option value="OPEN">مفتوح</option>
          <option value="UNDER_REVIEW">قيد المراجعة</option>
          <option value="FINISHED">منتهي</option>
        </select>

        <select
          value={filters.severity}
          onChange={(e) => handleFilterChange({ severity: e.target.value as any })}
        >
          <option value="">جميع المستويات</option>
          <option value="CRITICAL">حرج</option>
          <option value="HIGH">عالي</option>
        </select>

        <input
          type="text"
          placeholder="بحث..."
          onChange={(e) => handleFilterChange({ search: e.target.value })}
        />
      </div>

      {/* Table */}
      <table>
        <thead>
          <tr>
            <th>رقم السجل</th>
            <th>اسم المريض</th>
            <th>التاريخ</th>
            <th>القسم</th>
            <th>التصنيف</th>
            <th>الخطورة</th>
            <th>الحالة</th>
            <th>حدث لا يجب أن يحدث</th>
          </tr>
        </thead>
        <tbody>
          {redFlags.map((flag) => (
            <tr key={flag.red_flag_id}>
              <td>{flag.recordID}</td>
              <td>{flag.patient_name}</td>
              <td>{new Date(flag.date_received).toLocaleDateString('ar-SA')}</td>
              <td>{flag.department}</td>
              <td>{flag.category}</td>
              <td>
                <span className={`severity ${flag.severity.toLowerCase()}`}>
                  {flag.severity === 'CRITICAL' ? 'حرج' : 'عالي'}
                </span>
              </td>
              <td>
                <span className={`status ${flag.status.toLowerCase()}`}>
                  {flag.status === 'OPEN' ? 'مفتوح' : 
                   flag.status === 'UNDER_REVIEW' ? 'قيد المراجعة' : 'منتهي'}
                </span>
              </td>
              <td>{flag.isNeverEvent ? '✓' : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Pagination */}
      <div className="pagination">
        <span>إجمالي: {total}</span>
        <button
          disabled={filters.offset === 0}
          onClick={() => handlePageChange(Math.max(0, filters.offset! - filters.limit!))}
        >
          السابق
        </button>
        <button
          disabled={filters.offset! + filters.limit! >= total}
          onClick={() => handlePageChange(filters.offset! + filters.limit!)}
        >
          التالي
        </button>
      </div>
    </div>
  );
};
```

### 2. Statistics/KPI Cards Component

```typescript
// components/RedFlagsStatistics.tsx

import React, { useState, useEffect } from 'react';
import { fetchRedFlagsStatistics, handleRedFlagsApiError } from '../services/api/redFlagsApi';
import { RedFlagStatistics } from '../types/RedFlag.types';

export const RedFlagsStatistics: React.FC = () => {
  const [stats, setStats] = useState<RedFlagStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadStatistics();
  }, []);

  const loadStatistics = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await fetchRedFlagsStatistics();
      setStats(data);
    } catch (err) {
      setError(handleRedFlagsApiError(err));
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div>جاري تحميل الإحصائيات...</div>;
  if (error) return <div className="error">{error}</div>;
  if (!stats) return null;

  return (
    <div className="statistics-cards">
      {/* Total Red Flags */}
      <div className="stat-card">
        <h3>إجمالي الأعلام الحمراء</h3>
        <div className="stat-value">{stats.total_red_flags}</div>
      </div>

      {/* Unfinished */}
      <div className="stat-card warning">
        <h3>غير منتهي</h3>
        <div className="stat-value">{stats.unfinished}</div>
        <div className="stat-breakdown">
          <div>مفتوح: {stats.by_status.OPEN}</div>
          <div>قيد المراجعة: {stats.by_status.UNDER_REVIEW}</div>
        </div>
      </div>

      {/* Finished */}
      <div className="stat-card success">
        <h3>منتهي</h3>
        <div className="stat-value">{stats.finished}</div>
      </div>

      {/* By Severity */}
      <div className="stat-card">
        <h3>حسب الخطورة</h3>
        <div className="stat-breakdown">
          <div className="critical">حرج: {stats.by_severity.CRITICAL}</div>
          <div className="high">عالي: {stats.by_severity.HIGH}</div>
        </div>
      </div>

      {/* Current Month */}
      <div className="stat-card">
        <h3>الشهر الحالي</h3>
        <div className="stat-value">{stats.current_month.count}</div>
        <div className="stat-meta">{stats.current_month.month}</div>
      </div>

      {/* Never Event Overlap */}
      <div className="stat-card info">
        <h3>تداخل الأحداث الحرجة</h3>
        <div className="stat-breakdown">
          <div>أعلام حمراء + أحداث لا تحدث: {stats.never_event_overlap.red_flags_also_never_events}</div>
          <div>أعلام حمراء فقط: {stats.never_event_overlap.red_flags_only}</div>
          <div>أحداث لا تحدث فقط: {stats.never_event_overlap.never_events_only}</div>
        </div>
      </div>
    </div>
  );
};
```

### 3. Trend Chart Component

```typescript
// components/RedFlagsTrendChart.tsx

import React, { useState, useEffect } from 'react';
import { fetchRedFlagsTrends, handleRedFlagsApiError } from '../services/api/redFlagsApi';
import { RedFlagTrends, RedFlagTrendParams } from '../types/RedFlag.types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

export const RedFlagsTrendChart: React.FC = () => {
  const [trends, setTrends] = useState<RedFlagTrends | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const [params, setParams] = useState<RedFlagTrendParams>({
    granularity: 'monthly',
    group_by: 'none'
  });

  useEffect(() => {
    loadTrends();
  }, [params]);

  const loadTrends = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await fetchRedFlagsTrends(params);
      setTrends(data);
    } catch (err) {
      setError(handleRedFlagsApiError(err));
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div>جاري تحميل الاتجاهات...</div>;
  if (error) return <div className="error">{error}</div>;
  if (!trends) return null;

  // Format data for chart
  const chartData = trends.trends.map(trend => ({
    period: trend.period,
    count: trend.count || trend.total || 0,
    ...trend.breakdown
  }));

  return (
    <div className="trend-chart">
      <div className="chart-controls">
        <select
          value={params.granularity}
          onChange={(e) => setParams({ ...params, granularity: e.target.value as any })}
        >
          <option value="monthly">شهري</option>
          <option value="quarterly">ربع سنوي</option>
          <option value="weekly">أسبوعي</option>
        </select>

        <select
          value={params.group_by}
          onChange={(e) => setParams({ ...params, group_by: e.target.value as any })}
        >
          <option value="none">بدون تجميع</option>
          <option value="category">حسب التصنيف</option>
          <option value="severity">حسب الخطورة</option>
          <option value="department">حسب القسم</option>
        </select>
      </div>

      <LineChart width={800} height={400} data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="period" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="count" stroke="#8884d8" name="العدد" />
      </LineChart>
    </div>
  );
};
```

### 4. Red Flag Details Modal

```typescript
// components/RedFlagDetailsModal.tsx

import React, { useState, useEffect } from 'react';
import { fetchRedFlagDetails, handleRedFlagsApiError } from '../services/api/redFlagsApi';
import { RedFlagDetails } from '../types/RedFlag.types';

interface Props {
  redFlagId: number;
  onClose: () => void;
}

export const RedFlagDetailsModal: React.FC<Props> = ({ redFlagId, onClose }) => {
  const [details, setDetails] = useState<RedFlagDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDetails();
  }, [redFlagId]);

  const loadDetails = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await fetchRedFlagDetails(redFlagId);
      setDetails(data);
    } catch (err) {
      setError(handleRedFlagsApiError(err));
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="modal">جاري التحميل...</div>;
  if (error) return <div className="modal error">{error}</div>;
  if (!details) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="close-btn" onClick={onClose}>×</button>
        
        <h2>{details.recordID}</h2>
        
        {/* Basic Info */}
        <section>
          <h3>المعلومات الأساسية</h3>
          <div className="info-grid">
            <div><strong>اسم المريض:</strong> {details.patient_name}</div>
            <div><strong>التاريخ:</strong> {details.date_received}</div>
            <div><strong>القسم:</strong> {details.department}</div>
            <div><strong>التصنيف:</strong> {details.category}</div>
            <div><strong>التصنيف الفرعي:</strong> {details.subcategory}</div>
            <div><strong>الخطورة:</strong> {details.severity}</div>
            <div><strong>الحالة:</strong> {details.status}</div>
            <div><strong>حدث لا يجب أن يحدث:</strong> {details.isNeverEvent ? 'نعم' : 'لا'}</div>
          </div>
        </section>

        {/* Incident Details */}
        <section>
          <h3>تفاصيل الحادث</h3>
          <div><strong>نص الشكوى:</strong> {details.incident_details.complaint_text}</div>
          <div><strong>الإجراء الفوري:</strong> {details.incident_details.immediate_action}</div>
          <div><strong>الإجراءات المتخذة:</strong> {details.incident_details.actions_taken}</div>
          <div><strong>السبب الجذري:</strong> {details.incident_details.root_cause}</div>
          <div><strong>مستوى الضرر:</strong> {details.incident_details.harm_level}</div>
          <div><strong>المرحلة:</strong> {details.incident_details.stage}</div>
        </section>

        {/* Timeline */}
        <section>
          <h3>الجدول الزمني</h3>
          <div className="timeline">
            {details.timeline.map((event, index) => (
              <div key={index} className="timeline-item">
                <div className="timeline-date">{event.date}</div>
                <div className="timeline-event">{event.event}</div>
                <div className="timeline-details">{event.details}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Related Actions */}
        <section>
          <h3>الإجراءات ذات الصلة</h3>
          <table>
            <thead>
              <tr>
                <th>الإجراء</th>
                <th>المسؤول</th>
                <th>الموعد النهائي</th>
                <th>الحالة</th>
              </tr>
            </thead>
            <tbody>
              {details.related_actions.map((action, index) => (
                <tr key={index}>
                  <td>{action.action}</td>
                  <td>{action.responsible}</td>
                  <td>{action.deadline}</td>
                  <td>{action.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
};
```

---

## State Management

### Using React Context

```typescript
// context/RedFlagsContext.tsx

import React, { createContext, useContext, useState, useCallback } from 'react';
import { RedFlagFilters } from '../types/RedFlag.types';

interface RedFlagsContextType {
  filters: RedFlagFilters;
  updateFilters: (newFilters: Partial<RedFlagFilters>) => void;
  clearFilters: () => void;
}

const RedFlagsContext = createContext<RedFlagsContextType | undefined>(undefined);

export const RedFlagsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [filters, setFilters] = useState<RedFlagFilters>({
    status: 'all',
    limit: 50,
    offset: 0
  });

  const updateFilters = useCallback((newFilters: Partial<RedFlagFilters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({ status: 'all', limit: 50, offset: 0 });
  }, []);

  return (
    <RedFlagsContext.Provider value={{ filters, updateFilters, clearFilters }}>
      {children}
    </RedFlagsContext.Provider>
  );
};

export const useRedFlags = () => {
  const context = useContext(RedFlagsContext);
  if (!context) {
    throw new Error('useRedFlags must be used within RedFlagsProvider');
  }
  return context;
};
```

---

## UI Components Mapping

| Backend Field | UI Display | Notes |
|---------------|-----------|-------|
| `recordID` | Record ID column | Format: RF-YYYY-NNN |
| `patient_name` | Patient Name column | Arabic text |
| `date_received` | Date column | Format as Arabic date |
| `department` | Department column | Arabic text |
| `category` | Category column | Arabic text |
| `severity` | Severity badge | CRITICAL=حرج, HIGH=عالي |
| `status` | Status badge | OPEN=مفتوح, UNDER_REVIEW=قيد المراجعة, FINISHED=منتهي |
| `isNeverEvent` | Never Event indicator | ✓ or - |
| `complaint_summary` | Summary text | Show in tooltip or modal |

---

## Error Handling

```typescript
// Example error handling pattern

try {
  const data = await fetchRedFlags({ status: 'OPEN' });
  // Handle success
} catch (error) {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status;
    const errorData = error.response?.data;
    
    switch (status) {
      case 404:
        showNotification('لم يتم العثور على البيانات', 'warning');
        break;
      case 500:
        showNotification('خطأ في الخادم. يرجى المحاولة مرة أخرى', 'error');
        break;
      default:
        showNotification(errorData?.message_ar || 'حدث خطأ غير متوقع', 'error');
    }
  } else {
    showNotification('خطأ في الاتصال بالخادم', 'error');
  }
}
```

---

## Testing URLs

### Test each endpoint in browser or Postman:

1. **Health Check:**
   ```
   http://127.0.0.1:8000/api/red-flags/test
   ```

2. **List (Open cases):**
   ```
   http://127.0.0.1:8000/api/red-flags?status=OPEN&limit=50
   ```

3. **Statistics:**
   ```
   http://127.0.0.1:8000/api/red-flags/statistics
   ```

4. **Trends (Monthly):**
   ```
   http://127.0.0.1:8000/api/red-flags/trends?granularity=monthly
   ```

5. **Single Details:**
   ```
   http://127.0.0.1:8000/api/red-flags/1
   ```

---

## Implementation Checklist

- [ ] Create TypeScript interfaces from provided types
- [ ] Set up axios API service layer
- [ ] Implement Red Flags list component with filters
- [ ] Implement KPI statistics cards
- [ ] Implement trend chart component
- [ ] Implement details modal
- [ ] Add error handling with Arabic messages
- [ ] Add loading states
- [ ] Test all endpoints
- [ ] Handle pagination
- [ ] Add date range filters
- [ ] Test search functionality
- [ ] Verify Arabic text rendering
- [ ] Add export buttons (disabled for now)

---

## Notes

1. **PDF Export:** Not yet implemented - display "قريبًا" message
2. **Batch Export:** Not yet implemented - display "قريبًا" message
3. **Date Format:** Backend returns ISO format, convert to Arabic format in UI
4. **Pagination:** Implement using limit/offset parameters
5. **Error Messages:** Backend provides both English and Arabic messages
6. **Never Events:** Red flags can overlap with Never Events (ClinicalRiskTypeID = 3)

---

## Support

For questions or issues, contact the backend development team.

**API Documentation:** http://127.0.0.1:8000/docs

---

**Last Updated:** 2024-12-XX  
**Version:** 1.0
