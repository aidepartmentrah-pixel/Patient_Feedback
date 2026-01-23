/**
 * Base Types for Graph Operators
 * 
 * TypeScript types matching the backend Pydantic schemas.
 * These types ensure type safety when interacting with the operator API.
 */

/**
 * Available dimension types for distributing data.
 * Maps to backend DimensionType enum.
 */
export enum DimensionType {
  DOMAIN = "domain",
  CATEGORY = "category",
  SUBCATEGORY = "subcategory",
  CLASSIFICATION = "classification",
  STAGE = "stage",
  SEVERITY = "severity",
  HARM = "harm",
}

/**
 * Time modes for data distribution.
 * Maps to backend TimeMode enum.
 */
export enum TimeMode {
  /** Single time window - produces 1 bucket */
  SINGLE = "single",
  /** Multiple discrete time windows - produces N buckets for comparison */
  MULTI = "multi",
  /** Before/After a specific date - produces 2 buckets */
  BINARY_SPLIT = "binary_split",
}

/**
 * Time window granularity types.
 * Maps to backend TimeWindowType enum.
 */
export enum TimeWindowType {
  YEAR = "year",
  SEASON = "season",
  MONTH = "month",
  RANGE = "range",
}

/**
 * Time window for a full year.
 */
export interface TimeWindowYear {
  type: "year";
  value: number; // 2000-2100
}

/**
 * Time window for a quarter or trimester.
 * Quarters: 2025-Q1, 2025-Q2, 2025-Q3, 2025-Q4
 * Trimesters: 2025-T1, 2025-T2, 2025-T3
 */
export interface TimeWindowSeason {
  type: "season";
  value: string; // Format: YYYY-Q[1-4] or YYYY-T[1-3]
}

/**
 * Time window for a specific month.
 */
export interface TimeWindowMonth {
  type: "month";
  value: string; // Format: YYYY-MM
}

/**
 * Time window for a custom date range.
 */
export interface TimeWindowRange {
  type: "range";
  from_date: string; // ISO date: YYYY-MM-DD
  to_date: string; // ISO date: YYYY-MM-DD
}

/**
 * Union type for all time window variants.
 * TypeScript discriminated union matching backend Pydantic discriminated union.
 */
export type TimeWindow =
  | TimeWindowYear
  | TimeWindowSeason
  | TimeWindowMonth
  | TimeWindowRange;

/**
 * Optional filters for organizational and dimensional constraints.
 */
export interface OperatorFilters {
  // Organizational filters
  hospital_id?: number | null;
  department_id?: number | null;
  unit_id?: number | null;

  // Dimensional filters (for filtering within a dimension)
  domain?: string | null;
  category?: string | null;
  subcategory?: string | null;
  classification?: string | null;
  stage?: string | null;
  severity?: string | null;
  harm?: string | null;
}
