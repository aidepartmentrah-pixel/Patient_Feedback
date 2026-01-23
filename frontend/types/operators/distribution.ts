/**
 * Distribution Operator Types
 * 
 * TypeScript types for the Distribution Operator (DIST_1D_TIME_PARTITIONED).
 * Matches backend DistributionRequest and DistributionResponse schemas.
 */

import {
  DimensionType,
  TimeMode,
  TimeWindow,
  OperatorFilters,
} from "./base";

/**
 * Distribution Operator Request
 * 
 * Request format for the Distribution Operator endpoint.
 * Supports three time modes: single, multi, and binary_split.
 */
export interface DistributionRequest {
  dimension: DimensionType;
  time_mode: TimeMode;

  // Time configuration (mutually exclusive based on time_mode)
  time_window?: TimeWindow; // For SINGLE mode
  time_windows?: TimeWindow[]; // For MULTI mode (min 2 windows)
  split_date?: string; // For BINARY_SPLIT mode (ISO date: YYYY-MM-DD)

  // Optional filters
  filters?: OperatorFilters;
}

/**
 * Single distribution value (one bar/slice in the chart).
 */
export interface DistributionValue {
  key: string; // Dimension value (e.g., "High", "Clinical")
  count: number; // Absolute count
  percent: number; // Percentage (0.0 to 1.0)
}

/**
 * One time bucket with distribution values.
 * Represents one period in the comparison (e.g., "2025", "Q1", "Before").
 */
export interface DistributionBucket {
  time_label: string; // Display label for the time period
  total: number; // Sum of all counts
  values: DistributionValue[]; // Distribution breakdown
  status?: "NO_DATA" | null; // Special status if no data exists
}

/**
 * Distribution Operator Response
 * 
 * Response from the Distribution Operator endpoint.
 * Contains one or more time buckets with distribution data.
 */
export interface DistributionResponse {
  dimension: string; // Echo of requested dimension
  time_mode: string; // Echo of requested time mode
  buckets: DistributionBucket[]; // Distribution data for each time period
}

/**
 * Type guard to check if a request is for single mode.
 */
export function isSingleModeRequest(
  request: DistributionRequest
): request is DistributionRequest & { time_window: TimeWindow } {
  return request.time_mode === TimeMode.SINGLE && request.time_window !== undefined;
}

/**
 * Type guard to check if a request is for multi mode.
 */
export function isMultiModeRequest(
  request: DistributionRequest
): request is DistributionRequest & { time_windows: TimeWindow[] } {
  return request.time_mode === TimeMode.MULTI && request.time_windows !== undefined;
}

/**
 * Type guard to check if a request is for binary split mode.
 */
export function isBinarySplitRequest(
  request: DistributionRequest
): request is DistributionRequest & { split_date: string } {
  return request.time_mode === TimeMode.BINARY_SPLIT && request.split_date !== undefined;
}
