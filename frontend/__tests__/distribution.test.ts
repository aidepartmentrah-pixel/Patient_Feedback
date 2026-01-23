/**
 * Tests for Distribution Operator Types
 * 
 * Tests for distribution-specific types and type guards.
 */

import {
  DimensionType,
  TimeMode,
} from "../types/operators/base";

import {
  type DistributionRequest,
  type DistributionResponse,
  type DistributionBucket,
  type DistributionValue,
  isSingleModeRequest,
  isMultiModeRequest,
  isBinarySplitRequest,
} from "../types/operators/distribution";

describe("DistributionRequest Type", () => {
  test("should support single mode request", () => {
    const request: DistributionRequest = {
      dimension: DimensionType.SEVERITY,
      time_mode: TimeMode.SINGLE,
      time_window: {
        type: "year",
        value: 2025,
      },
    };

    expect(request.dimension).toBe(DimensionType.SEVERITY);
    expect(request.time_mode).toBe(TimeMode.SINGLE);
    expect(request.time_window).toBeDefined();
  });

  test("should support multi mode request", () => {
    const request: DistributionRequest = {
      dimension: DimensionType.DOMAIN,
      time_mode: TimeMode.MULTI,
      time_windows: [
        { type: "season", value: "2024-Q4" },
        { type: "season", value: "2025-Q1" },
      ],
    };

    expect(request.dimension).toBe(DimensionType.DOMAIN);
    expect(request.time_mode).toBe(TimeMode.MULTI);
    expect(request.time_windows).toHaveLength(2);
  });

  test("should support binary split mode request", () => {
    const request: DistributionRequest = {
      dimension: DimensionType.STAGE,
      time_mode: TimeMode.BINARY_SPLIT,
      split_date: "2024-06-01",
    };

    expect(request.dimension).toBe(DimensionType.STAGE);
    expect(request.time_mode).toBe(TimeMode.BINARY_SPLIT);
    expect(request.split_date).toBe("2024-06-01");
  });

  test("should support request with filters", () => {
    const request: DistributionRequest = {
      dimension: DimensionType.SEVERITY,
      time_mode: TimeMode.SINGLE,
      time_window: { type: "year", value: 2025 },
      filters: {
        department_id: 42,
        severity: "High",
      },
    };

    expect(request.filters).toBeDefined();
    expect(request.filters?.department_id).toBe(42);
    expect(request.filters?.severity).toBe("High");
  });
});

describe("DistributionValue Type", () => {
  test("should have correct structure", () => {
    const value: DistributionValue = {
      key: "High",
      count: 150,
      percent: 0.3,
    };

    expect(value.key).toBe("High");
    expect(value.count).toBe(150);
    expect(value.percent).toBe(0.3);
  });
});

describe("DistributionBucket Type", () => {
  test("should have correct structure with data", () => {
    const bucket: DistributionBucket = {
      time_label: "2025",
      total: 500,
      values: [
        { key: "High", count: 150, percent: 0.3 },
        { key: "Medium", count: 300, percent: 0.6 },
        { key: "Low", count: 50, percent: 0.1 },
      ],
    };

    expect(bucket.time_label).toBe("2025");
    expect(bucket.total).toBe(500);
    expect(bucket.values).toHaveLength(3);
  });

  test("should support NO_DATA status", () => {
    const bucket: DistributionBucket = {
      time_label: "2025",
      total: 0,
      values: [],
      status: "NO_DATA",
    };

    expect(bucket.status).toBe("NO_DATA");
    expect(bucket.total).toBe(0);
    expect(bucket.values).toHaveLength(0);
  });

  test("should allow null status", () => {
    const bucket: DistributionBucket = {
      time_label: "2025",
      total: 100,
      values: [{ key: "Test", count: 100, percent: 1.0 }],
      status: null,
    };

    expect(bucket.status).toBeNull();
  });

  test("should allow undefined status", () => {
    const bucket: DistributionBucket = {
      time_label: "2025",
      total: 100,
      values: [{ key: "Test", count: 100, percent: 1.0 }],
    };

    expect(bucket.status).toBeUndefined();
  });
});

describe("DistributionResponse Type", () => {
  test("should have correct structure", () => {
    const response: DistributionResponse = {
      dimension: "severity",
      time_mode: "single",
      buckets: [
        {
          time_label: "2025",
          total: 500,
          values: [
            { key: "High", count: 150, percent: 0.3 },
            { key: "Medium", count: 300, percent: 0.6 },
            { key: "Low", count: 50, percent: 0.1 },
          ],
        },
      ],
    };

    expect(response.dimension).toBe("severity");
    expect(response.time_mode).toBe("single");
    expect(response.buckets).toHaveLength(1);
  });

  test("should support multiple buckets", () => {
    const response: DistributionResponse = {
      dimension: "severity",
      time_mode: "multi",
      buckets: [
        {
          time_label: "2023",
          total: 100,
          values: [{ key: "High", count: 100, percent: 1.0 }],
        },
        {
          time_label: "2024",
          total: 150,
          values: [{ key: "High", count: 150, percent: 1.0 }],
        },
      ],
    };

    expect(response.buckets).toHaveLength(2);
  });
});

describe("Type Guard Functions", () => {
  test("isSingleModeRequest should correctly identify single mode", () => {
    const singleRequest: DistributionRequest = {
      dimension: DimensionType.SEVERITY,
      time_mode: TimeMode.SINGLE,
      time_window: { type: "year", value: 2025 },
    };

    expect(isSingleModeRequest(singleRequest)).toBe(true);

    if (isSingleModeRequest(singleRequest)) {
      // Type narrowing should work
      expect(singleRequest.time_window.type).toBe("year");
    }
  });

  test("isSingleModeRequest should reject non-single modes", () => {
    const multiRequest: DistributionRequest = {
      dimension: DimensionType.SEVERITY,
      time_mode: TimeMode.MULTI,
      time_windows: [{ type: "year", value: 2025 }],
    };

    expect(isSingleModeRequest(multiRequest)).toBe(false);
  });

  test("isMultiModeRequest should correctly identify multi mode", () => {
    const multiRequest: DistributionRequest = {
      dimension: DimensionType.SEVERITY,
      time_mode: TimeMode.MULTI,
      time_windows: [
        { type: "year", value: 2024 },
        { type: "year", value: 2025 },
      ],
    };

    expect(isMultiModeRequest(multiRequest)).toBe(true);

    if (isMultiModeRequest(multiRequest)) {
      // Type narrowing should work
      expect(multiRequest.time_windows.length).toBe(2);
    }
  });

  test("isMultiModeRequest should reject non-multi modes", () => {
    const singleRequest: DistributionRequest = {
      dimension: DimensionType.SEVERITY,
      time_mode: TimeMode.SINGLE,
      time_window: { type: "year", value: 2025 },
    };

    expect(isMultiModeRequest(singleRequest)).toBe(false);
  });

  test("isBinarySplitRequest should correctly identify binary split mode", () => {
    const binaryRequest: DistributionRequest = {
      dimension: DimensionType.STAGE,
      time_mode: TimeMode.BINARY_SPLIT,
      split_date: "2024-06-01",
    };

    expect(isBinarySplitRequest(binaryRequest)).toBe(true);

    if (isBinarySplitRequest(binaryRequest)) {
      // Type narrowing should work
      expect(binaryRequest.split_date).toBe("2024-06-01");
    }
  });

  test("isBinarySplitRequest should reject non-binary-split modes", () => {
    const singleRequest: DistributionRequest = {
      dimension: DimensionType.SEVERITY,
      time_mode: TimeMode.SINGLE,
      time_window: { type: "year", value: 2025 },
    };

    expect(isBinarySplitRequest(singleRequest)).toBe(false);
  });
});
