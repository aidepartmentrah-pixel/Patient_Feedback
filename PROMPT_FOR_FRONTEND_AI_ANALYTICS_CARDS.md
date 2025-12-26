# Frontend Implementation: Analytics Dashboard Cards

## Overview

Implement **4 analytics dashboard cards** using the new analytics endpoints for Red Flags and Never Events. These cards provide visual insights into incident distribution and trends.

**Priority Order:**
1. Red Flags - Category Breakdown Card
2. Never Events - Category Breakdown Card
3. Never Events - Timeline Comparison Card
4. Red Flags - Department Breakdown Card

---

## Card 1: Red Flags Category Breakdown

### Purpose
Show distribution of red flags across categories with severity breakdown (CRITICAL vs HIGH).

### API Endpoint
```
GET /api/red-flags/category-breakdown
```

### Query Parameters
- `from_date` (optional): Filter from date (YYYY-MM-DD)
- `to_date` (optional): Filter to date (YYYY-MM-DD)

### Response Structure
```typescript
interface CategoryBreakdownResponse {
  total: number;
  period: string;
  categories: Array<{
    category_name: string;        // English name
    category_name_ar: string;     // Arabic name
    count: number;
    percentage: number;           // 0-100, one decimal
    severity_breakdown: {
      CRITICAL: number;
      HIGH: number;
    };
  }>;
}
```

### Card Design Requirements

**Layout:**
- Title: "Red Flags by Category" / "الأعلام الحمراء حسب الفئة"
- Date range filter (optional)
- Pie/Donut chart showing category distribution
- Legend with category names (Arabic preferred)
- Hover tooltip showing count, percentage, and severity breakdown

**Visual Elements:**
- Use distinct colors for each category
- Show top 5 categories, group rest as "Other"
- Display total count as KPI number
- Empty state: "No red flags in selected period"

**Interactivity:**
- Click category to filter/drill-down (future enhancement)
- Hover to see severity breakdown: "CRITICAL: X, HIGH: Y"

### Implementation Example
```typescript
// Fetch data
const fetchCategoryBreakdown = async (fromDate?: string, toDate?: string) => {
  const params = new URLSearchParams();
  if (fromDate) params.append('from_date', fromDate);
  if (toDate) params.append('to_date', toDate);
  
  const response = await fetch(
    `http://127.0.0.1:8000/api/red-flags/category-breakdown?${params}`
  );
  return await response.json();
};

// Component usage
const CategoryBreakdownCard = () => {
  const { data, isLoading } = useQuery(['red-flags-category'], 
    () => fetchCategoryBreakdown()
  );
  
  const chartData = data?.categories.slice(0, 5).map(cat => ({
    name: cat.category_name_ar,
    value: cat.count,
    percentage: cat.percentage,
    severity: cat.severity_breakdown
  }));
  
  return (
    <Card>
      <CardHeader>
        <h3>الأعلام الحمراء حسب الفئة</h3>
        <Badge variant="critical">{data?.total} Total</Badge>
      </CardHeader>
      <CardContent>
        <PieChart data={chartData} />
      </CardContent>
    </Card>
  );
};
```

---

## Card 2: Never Events Category Breakdown

### Purpose
Show distribution of never events across categories with drill-down to specific event types. Emphasize zero-tolerance goal.

### API Endpoint
```
GET /api/never-events/category-breakdown
```

### Query Parameters
- `from_date` (optional): Filter from date (YYYY-MM-DD)
- `to_date` (optional): Filter to date (YYYY-MM-DD)

### Response Structure
```typescript
interface NeverEventCategoryResponse {
  total: number;
  goal: number;              // Always 0
  period: string;
  categories: Array<{
    category_name: string;
    category_name_ar: string;
    count: number;
    percentage: number;
    types: Array<{           // Specific event types
      type: string;
      type_ar: string;
      count: number;
    }>;
  }>;
}
```

### Card Design Requirements

**Layout:**
- Title: "Never Events by Category" / "الأحداث التي لا يجب أن تحدث حسب الفئة"
- Goal indicator: **"Goal: 0"** prominently displayed (red/warning color)
- Actual count vs goal comparison
- Pie/Donut chart with category distribution
- Expandable sections to show specific event types

**Visual Elements:**
- Red/warning color scheme (since any count > 0 is bad)
- Show total with alert styling: "12 events (Goal: 0)"
- Hover tooltip shows specific event types
- Empty state: "✓ Zero never events - Goal achieved!" (green)

**Interactivity:**
- Click category to expand and see specific event types
- Example: "Surgical Events (5)" → Shows "Wrong Site Surgery (3), Wrong Patient (2)"
- Color-code: Green if total = 0, Red if total > 0

### Implementation Example
```typescript
const NeverEventsCategoryCard = () => {
  const { data } = useQuery(['never-events-category'], 
    () => fetch('http://127.0.0.1:8000/api/never-events/category-breakdown').then(r => r.json())
  );
  
  const isGoalMet = data?.total === 0;
  
  return (
    <Card className={isGoalMet ? 'border-green' : 'border-red'}>
      <CardHeader>
        <h3>الأحداث التي لا يجب أن تحدث</h3>
        <div className="flex items-center gap-2">
          <Badge variant={isGoalMet ? 'success' : 'danger'}>
            {data?.total} / Goal: {data?.goal}
          </Badge>
          {isGoalMet && <CheckCircle className="text-green-500" />}
        </div>
      </CardHeader>
      <CardContent>
        {isGoalMet ? (
          <EmptyState 
            icon="✓" 
            title="Zero Never Events"
            subtitle="Goal achieved!"
          />
        ) : (
          <>
            <PieChart data={data?.categories} />
            <Accordion>
              {data?.categories.map(cat => (
                <AccordionItem key={cat.category_name}>
                  <AccordionTrigger>
                    {cat.category_name_ar} ({cat.count})
                  </AccordionTrigger>
                  <AccordionContent>
                    <ul>
                      {cat.types.map(type => (
                        <li key={type.type}>
                          {type.type_ar}: {type.count}
                        </li>
                      ))}
                    </ul>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </>
        )}
      </CardContent>
    </Card>
  );
};
```

---

## Card 3: Never Events Timeline Comparison

### Purpose
Compare current period vs previous period to track progress toward zero. Show trend direction (improving/worsening).

### API Endpoint
```
GET /api/never-events/timeline-comparison
```

### Query Parameters
- `period` (optional): Time period - "month", "quarter", or "year" (default: month)

### Response Structure
```typescript
interface TimelineComparisonResponse {
  goal: number;                  // Always 0
  current: {
    period: string;              // "December 2024"
    period_ar: string;           // "ديسمبر 2024"
    count: number;
    start_date: string;          // "2024-12-01"
    end_date: string;            // "2024-12-31"
  };
  previous: {
    period: string;
    period_ar: string;
    count: number;
  };
  change: {
    absolute: number;            // Can be negative (good)
    percentage: number;          // Can be negative (good)
    trend: "improving" | "worsening" | "stable";
  };
  year_to_date: {
    count: number;
    average_per_month: number;
  };
}
```

### Card Design Requirements

**Layout:**
- Title: "Never Events Trend" / "اتجاه الأحداث"
- Period selector: Month / Quarter / Year tabs
- Comparison display: Current vs Previous
- Trend indicator with arrow (↑ worsening, ↓ improving)
- YTD summary section

**Visual Elements:**
- Large numbers showing current and previous counts
- Color-coded trend:
  - **Green** + ↓ arrow: Improving (count decreased)
  - **Red** + ↑ arrow: Worsening (count increased)
  - **Yellow** + → arrow: Stable (no change)
- Percentage change with +/- sign
- Goal reminder: "Goal: 0"
- YTD stats at bottom

**Interactivity:**
- Period selector (Month/Quarter/Year) updates comparison
- Hover on numbers to see exact dates

### Implementation Example
```typescript
const TimelineComparisonCard = () => {
  const [period, setPeriod] = useState<'month' | 'quarter' | 'year'>('month');
  
  const { data } = useQuery(
    ['never-events-timeline', period],
    () => fetch(`http://127.0.0.1:8000/api/never-events/timeline-comparison?period=${period}`).then(r => r.json())
  );
  
  const trendConfig = {
    improving: { color: 'green', icon: '↓', text: 'Improving' },
    worsening: { color: 'red', icon: '↑', text: 'Worsening' },
    stable: { color: 'yellow', icon: '→', text: 'Stable' }
  };
  
  const trend = trendConfig[data?.change.trend || 'stable'];
  
  return (
    <Card>
      <CardHeader>
        <h3>اتجاه الأحداث</h3>
        <Tabs value={period} onValueChange={setPeriod}>
          <TabsList>
            <TabsTrigger value="month">Month</TabsTrigger>
            <TabsTrigger value="quarter">Quarter</TabsTrigger>
            <TabsTrigger value="year">Year</TabsTrigger>
          </TabsList>
        </Tabs>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="text-center">
            <p className="text-sm text-gray-500">{data?.current.period_ar}</p>
            <p className="text-4xl font-bold">{data?.current.count}</p>
            <p className="text-xs">Current</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-500">{data?.previous.period_ar}</p>
            <p className="text-4xl font-bold text-gray-400">{data?.previous.count}</p>
            <p className="text-xs">Previous</p>
          </div>
        </div>
        
        <div className={`flex items-center justify-center gap-2 p-3 rounded bg-${trend.color}-50`}>
          <span className={`text-2xl text-${trend.color}-600`}>{trend.icon}</span>
          <div>
            <p className={`font-semibold text-${trend.color}-600`}>
              {trend.text} ({data?.change.percentage > 0 ? '+' : ''}{data?.change.percentage.toFixed(1)}%)
            </p>
            <p className="text-sm">
              {data?.change.absolute > 0 ? '+' : ''}{data?.change.absolute} vs previous period
            </p>
          </div>
        </div>
        
        <Separator className="my-4" />
        
        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-sm">Goal:</span>
            <span className="font-semibold">{data?.goal}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm">YTD Total:</span>
            <span className="font-semibold">{data?.year_to_date.count}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm">Avg per Month:</span>
            <span className="font-semibold">{data?.year_to_date.average_per_month.toFixed(1)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
```

---

## Card 4: Red Flags Department Breakdown

### Purpose
Show which departments have the most red flags with status breakdown (OPEN, UNDER_REVIEW, FINISHED).

### API Endpoint
```
GET /api/red-flags/department-breakdown
```

### Query Parameters
- `from_date` (optional): Filter from date (YYYY-MM-DD)
- `to_date` (optional): Filter to date (YYYY-MM-DD)
- `limit` (optional): Max departments (default: 10, max: 50)

### Response Structure
```typescript
interface DepartmentBreakdownResponse {
  total: number;
  period: string;
  departments: Array<{
    department: string;          // Arabic name
    department_en: string;       // English name
    count: number;
    percentage: number;
    status_breakdown: {
      OPEN: number;
      UNDER_REVIEW: number;
      FINISHED: number;
    };
  }>;
}
```

### Card Design Requirements

**Layout:**
- Title: "Red Flags by Department" / "الأعلام الحمراء حسب القسم"
- Horizontal bar chart showing top departments
- Status breakdown as stacked bars (3 colors)
- Limit selector: Top 5 / Top 10 / Top 20
- Date range filter

**Visual Elements:**
- Stacked horizontal bar chart
  - Red segment: OPEN
  - Yellow segment: UNDER_REVIEW
  - Green segment: FINISHED
- Department names in Arabic
- Show count and percentage on hover
- Legend explaining status colors

**Interactivity:**
- Limit dropdown: "Show top: 5 / 10 / 20 departments"
- Click department to navigate to filtered view
- Hover to see exact status counts

### Implementation Example
```typescript
const DepartmentBreakdownCard = () => {
  const [limit, setLimit] = useState(10);
  
  const { data } = useQuery(
    ['red-flags-department', limit],
    () => fetch(`http://127.0.0.1:8000/api/red-flags/department-breakdown?limit=${limit}`).then(r => r.json())
  );
  
  return (
    <Card>
      <CardHeader>
        <h3>الأعلام الحمراء حسب القسم</h3>
        <Select value={limit.toString()} onValueChange={(v) => setLimit(Number(v))}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="5">Top 5</SelectItem>
            <SelectItem value="10">Top 10</SelectItem>
            <SelectItem value="20">Top 20</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {data?.departments.map(dept => (
            <div key={dept.department} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="font-medium">{dept.department}</span>
                <span className="text-gray-500">
                  {dept.count} ({dept.percentage.toFixed(1)}%)
                </span>
              </div>
              <div className="flex h-6 rounded overflow-hidden">
                <Tooltip content={`Open: ${dept.status_breakdown.OPEN}`}>
                  <div 
                    className="bg-red-500 hover:bg-red-600 transition-colors"
                    style={{ width: `${(dept.status_breakdown.OPEN / dept.count) * 100}%` }}
                  />
                </Tooltip>
                <Tooltip content={`Under Review: ${dept.status_breakdown.UNDER_REVIEW}`}>
                  <div 
                    className="bg-yellow-500 hover:bg-yellow-600 transition-colors"
                    style={{ width: `${(dept.status_breakdown.UNDER_REVIEW / dept.count) * 100}%` }}
                  />
                </Tooltip>
                <Tooltip content={`Finished: ${dept.status_breakdown.FINISHED}`}>
                  <div 
                    className="bg-green-500 hover:bg-green-600 transition-colors"
                    style={{ width: `${(dept.status_breakdown.FINISHED / dept.count) * 100}%` }}
                  />
                </Tooltip>
              </div>
            </div>
          ))}
        </div>
        
        <div className="flex gap-4 mt-4 pt-4 border-t text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-red-500 rounded" />
            <span>Open</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-yellow-500 rounded" />
            <span>Under Review</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-green-500 rounded" />
            <span>Finished</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
```

---

## Common Implementation Guidelines

### Error Handling
```typescript
const { data, isLoading, error } = useQuery(
  ['analytics-key'],
  fetchFunction,
  {
    retry: 2,
    staleTime: 5 * 60 * 1000,  // 5 minutes
    onError: (err) => {
      toast.error('Failed to load analytics data');
      console.error(err);
    }
  }
);

if (error) {
  return <ErrorCard message="Unable to load analytics" retry={refetch} />;
}

if (isLoading) {
  return <CardSkeleton />;
}
```

### Date Range Filtering
```typescript
const DateRangeFilter = ({ onDateChange }) => {
  const [fromDate, setFromDate] = useState('');
  const [toDate, setToDate] = useState('');
  
  const handleApply = () => {
    onDateChange(fromDate, toDate);
  };
  
  return (
    <div className="flex gap-2">
      <Input 
        type="date" 
        value={fromDate}
        onChange={(e) => setFromDate(e.target.value)}
        max={toDate || undefined}
      />
      <Input 
        type="date" 
        value={toDate}
        onChange={(e) => setToDate(e.target.value)}
        min={fromDate || undefined}
      />
      <Button onClick={handleApply}>Apply</Button>
    </div>
  );
};
```

### Empty States
```typescript
// When no data
if (data?.categories.length === 0) {
  return (
    <EmptyState
      icon={<InboxIcon />}
      title="No data available"
      subtitle="Try adjusting your date range"
    />
  );
}

// When goal is met (Never Events)
if (data?.total === 0) {
  return (
    <SuccessState
      icon={<CheckCircleIcon />}
      title="Zero Never Events"
      subtitle="Goal achieved for this period!"
    />
  );
}
```

### Loading States
```typescript
const CardSkeleton = () => (
  <Card>
    <CardHeader>
      <Skeleton className="h-6 w-40" />
    </CardHeader>
    <CardContent>
      <Skeleton className="h-64 w-full" />
    </CardContent>
  </Card>
);
```

---

## Dashboard Layout

### Recommended Grid Layout
```tsx
<div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6">
  {/* Top Row - Category Breakdowns */}
  <CategoryBreakdownCard endpoint="red-flags" />
  <CategoryBreakdownCard endpoint="never-events" />
  
  {/* Bottom Row - Timeline & Departments */}
  <TimelineComparisonCard />
  <DepartmentBreakdownCard />
</div>
```

### Responsive Behavior
- **Mobile**: Single column, cards stack vertically
- **Tablet**: 2 columns
- **Desktop**: 2x2 grid or 4 columns

---

## Testing Requirements

### Functional Tests
- [ ] All 4 endpoints fetch data successfully
- [ ] Date range filters update card data
- [ ] Period selector (Month/Quarter/Year) works
- [ ] Limit selector updates department count
- [ ] Empty states display correctly
- [ ] Error states handle API failures gracefully
- [ ] Loading states show during data fetch

### Visual Tests
- [ ] Charts render correctly with real data
- [ ] Arabic text displays properly (RTL support)
- [ ] Colors match design system
- [ ] Hover tooltips show full information
- [ ] Cards are responsive across screen sizes

### Data Validation
- [ ] Percentages sum to ~100%
- [ ] Sorting is correct (DESC by count)
- [ ] Severity/Status breakdowns add up to total
- [ ] Trend calculations are accurate
- [ ] Goal (0) is always displayed for Never Events

---

## Performance Optimization

### Data Caching
```typescript
// Cache analytics data for 5 minutes
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      cacheTime: 10 * 60 * 1000,
    },
  },
});
```

### Lazy Loading
```typescript
// Load cards as they enter viewport
const CategoryCard = lazy(() => import('./CategoryCard'));

<Suspense fallback={<CardSkeleton />}>
  <CategoryCard />
</Suspense>
```

### Debounce Filters
```typescript
const debouncedDateChange = useMemo(
  () => debounce((from, to) => {
    refetch({ from_date: from, to_date: to });
  }, 500),
  [refetch]
);
```

---

## Accessibility

- Use semantic HTML: `<section>`, `<article>`, `<figure>`
- Add `aria-label` to charts
- Ensure sufficient color contrast (WCAG AA)
- Support keyboard navigation
- Add screen reader descriptions for visual data

```tsx
<div 
  role="img" 
  aria-label={`Pie chart showing ${data.total} red flags across ${data.categories.length} categories`}
>
  <PieChart data={data.categories} />
</div>
```

---

## API Error Responses

All endpoints return HTTP 200 even with errors. Handle these cases:

```typescript
// Empty data
{
  "total": 0,
  "period": "all time",
  "categories": []
}

// Invalid date format returns all-time data
// No 400 errors for bad dates - just ignores them
```

---

## Priority Implementation Order

1. **Start with Red Flags Category Breakdown** (simplest)
   - Basic pie chart
   - No complex interactions
   - Test API integration

2. **Never Events Category Breakdown** (medium complexity)
   - Add drill-down functionality
   - Goal indicator logic
   - Expandable sections

3. **Never Events Timeline Comparison** (complex)
   - Period selector
   - Trend calculations
   - Color-coded indicators

4. **Red Flags Department Breakdown** (most complex)
   - Stacked bar charts
   - Multiple status colors
   - Limit selector

---

## Visual Design References

**Color Palette:**
- **Red Flags**: Blue/purple theme (professional)
- **Never Events**: Red/warning theme (zero tolerance)
- **Success**: Green (#10B981)
- **Warning**: Yellow (#F59E0B)
- **Danger**: Red (#EF4444)

**Typography:**
- Titles: 18px bold
- Counts: 32px bold (KPI numbers)
- Labels: 14px regular
- Arabic text: Ensure proper font support (e.g., Cairo, Tajawal)

---

## Complete Example: Dashboard Page

```tsx
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { PieChart, BarChart } from '@/components/charts';

const AnalyticsDashboard = () => {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Analytics Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <RedFlagsCategoryCard />
        <NeverEventsCategoryCard />
        <NeverEventsTimelineCard />
        <RedFlagsDepartmentCard />
      </div>
    </div>
  );
};

export default AnalyticsDashboard;
```

---

**Last Updated:** December 26, 2025  
**Status:** ✅ Ready for frontend implementation  
**Base URL:** `http://127.0.0.1:8000`  
**Documentation:** `http://127.0.0.1:8000/docs`
