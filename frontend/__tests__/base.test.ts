/**
 * Tests for Base Operator Types
 * 
 * Comprehensive tests for enums and type definitions.
 */

import {
  DimensionType,
  TimeMode,
  TimeWindowType,
  type TimeWindow,
  type TimeWindowYear,
  type TimeWindowSeason,
  type TimeWindowMonth,
  type TimeWindowRange,
  type OperatorFilters,
} from "../types/operators/base";

describe("DimensionType Enum", () => {
  test("should have all dimension values", () => {
    expect(DimensionType.DOMAIN).toBe("domain");
    expect(DimensionType.CATEGORY).toBe("category");
    expect(DimensionType.SUBCATEGORY).toBe("subcategory");
    expect(DimensionType.CLASSIFICATION).toBe("classification");
    expect(DimensionType.STAGE).toBe("stage");
    expect(DimensionType.SEVERITY).toBe("severity");
    expect(DimensionType.HARM).toBe("harm");
  });

  test("should have exactly 7 dimension types", () => {
    const values = Object.values(DimensionType);
    expect(values).toHaveLength(7);
  });
});

describe("TimeMode Enum", () => {
  test("should have all time mode values", () => {
    expect(TimeMode.SINGLE).toBe("single");
    expect(TimeMode.MULTI).toBe("multi");
    expect(TimeMode.BINARY_SPLIT).toBe("binary_split");
  });

  test("should have exactly 3 time modes", () => {
    const values = Object.values(TimeMode);
    expect(values).toHaveLength(3);
  });
});

describe("TimeWindowType Enum", () => {
  test("should have all window type values", () => {
    expect(TimeWindowType.YEAR).toBe("year");
    expect(TimeWindowType.SEASON).toBe("season");
    expect(TimeWindowType.MONTH).toBe("month");
    expect(TimeWindowType.RANGE).toBe("range");
  });

  test("should have exactly 4 window types", () => {
    const values = Object.values(TimeWindowType);
    expect(values).toHaveLength(4);
  });
});

describe("TimeWindow Types", () => {
  test("TimeWindowYear should have correct structure", () => {
    const window: TimeWindowYear = {
      type: "year",
      value: 2025,
    };

    expect(window.type).toBe("year");
    expect(window.value).toBe(2025);
  });

  test("TimeWindowSeason should have correct structure for quarter", () => {
    const window: TimeWindowSeason = {
      type: "season",
      value: "2025-Q1",
    };

    expect(window.type).toBe("season");
    expect(window.value).toBe("2025-Q1");
  });

  test("TimeWindowSeason should have correct structure for trimester", () => {
    const window: TimeWindowSeason = {
      type: "season",
      value: "2025-T2",
    };

    expect(window.type).toBe("season");
    expect(window.value).toBe("2025-T2");
  });

  test("TimeWindowMonth should have correct structure", () => {
    const window: TimeWindowMonth = {
      type: "month",
      value: "2025-06",
    };

    expect(window.type).toBe("month");
    expect(window.value).toBe("2025-06");
  });

  test("TimeWindowRange should have correct structure", () => {
    const window: TimeWindowRange = {
      type: "range",
      from_date: "2025-01-01",
      to_date: "2025-12-31",
    };

    expect(window.type).toBe("range");
    expect(window.from_date).toBe("2025-01-01");
    expect(window.to_date).toBe("2025-12-31");
  });

  test("TimeWindow discriminated union should accept all variants", () => {
    const yearWindow: TimeWindow = { type: "year", value: 2025 };
    const seasonWindow: TimeWindow = { type: "season", value: "2025-Q1" };
    const monthWindow: TimeWindow = { type: "month", value: "2025-06" };
    const rangeWindow: TimeWindow = {
      type: "range",
      from_date: "2025-01-01",
      to_date: "2025-12-31",
    };

    expect(yearWindow.type).toBe("year");
    expect(seasonWindow.type).toBe("season");
    expect(monthWindow.type).toBe("month");
    expect(rangeWindow.type).toBe("range");
  });
});

describe("OperatorFilters Type", () => {
  test("should allow empty filters", () => {
    const filters: OperatorFilters = {};
    expect(filters).toEqual({});
  });

  test("should allow organizational filters", () => {
    const filters: OperatorFilters = {
      hospital_id: 1,
      department_id: 2,
      unit_id: 3,
    };

    expect(filters.hospital_id).toBe(1);
    expect(filters.department_id).toBe(2);
    expect(filters.unit_id).toBe(3);
  });

  test("should allow dimensional filters", () => {
    const filters: OperatorFilters = {
      domain: "Clinical",
      category: "Patient Safety",
      severity: "High",
    };

    expect(filters.domain).toBe("Clinical");
    expect(filters.category).toBe("Patient Safety");
    expect(filters.severity).toBe("High");
  });

  test("should allow mixed filters", () => {
    const filters: OperatorFilters = {
      department_id: 42,
      severity: "High",
      domain: "Clinical",
    };

    expect(filters.department_id).toBe(42);
    expect(filters.severity).toBe("High");
    expect(filters.domain).toBe("Clinical");
  });

  test("should allow null values", () => {
    const filters: OperatorFilters = {
      hospital_id: null,
      severity: null,
    };

    expect(filters.hospital_id).toBeNull();
    expect(filters.severity).toBeNull();
  });
});
