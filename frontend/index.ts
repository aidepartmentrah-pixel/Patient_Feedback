/**
 * Graph Operators - Frontend Client Library
 * 
 * Main entry point for the operators client library.
 * Exports all types, enums, and API client functions.
 */

// Export base types and enums
export {
  DimensionType,
  TimeMode,
  TimeWindowType,
  type TimeWindow,
  type TimeWindowYear,
  type TimeWindowSeason,
  type TimeWindowMonth,
  type TimeWindowRange,
  type OperatorFilters,
} from "./types/operators/base";

// Export distribution types
export {
  type DistributionRequest,
  type DistributionResponse,
  type DistributionBucket,
  type DistributionValue,
  isSingleModeRequest,
  isMultiModeRequest,
  isBinarySplitRequest,
} from "./types/operators/distribution";

// Export API client
export {
  OperatorsClient,
  OperatorsAPIError,
  defaultClient,
  type OperatorsClientConfig,
} from "./api/operators-client";
