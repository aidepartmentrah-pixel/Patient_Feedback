/**
 * Tests for Operators API Client
 * 
 * Comprehensive tests for the API client with mocked fetch.
 */

import {
  OperatorsClient,
  OperatorsAPIError,
} from "../api/operators-client";

import {
  DimensionType,
  TimeMode,
  type DistributionRequest,
  type DistributionResponse,
} from "../index";

// Mock fetch globally
global.fetch = jest.fn();

describe("OperatorsClient", () => {
  let client: OperatorsClient;

  beforeEach(() => {
    client = new OperatorsClient({ baseUrl: "http://localhost:8000" });
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe("Constructor and Configuration", () => {
    test("should use default baseUrl if not provided", () => {
      const defaultClient = new OperatorsClient();
      expect(defaultClient).toBeDefined();
    });

    test("should use custom baseUrl if provided", () => {
      const customClient = new OperatorsClient({
        baseUrl: "http://api.example.com",
      });
      expect(customClient).toBeDefined();
    });

    test("should use custom headers if provided", () => {
      const customClient = new OperatorsClient({
        headers: {
          Authorization: "Bearer token123",
          "X-Custom-Header": "value",
        },
      });
      expect(customClient).toBeDefined();
    });

    test("should use custom timeout if provided", () => {
      const customClient = new OperatorsClient({
        timeout: 60000,
      });
      expect(customClient).toBeDefined();
    });
  });

  describe("distribution() - Successful Requests", () => {
    test("should successfully call distribution endpoint with single mode", async () => {
      const mockResponse: DistributionResponse = {
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

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const request: DistributionRequest = {
        dimension: DimensionType.SEVERITY,
        time_mode: TimeMode.SINGLE,
        time_window: { type: "year", value: 2025 },
      };

      const response = await client.distribution(request);

      expect(response).toEqual(mockResponse);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/operators/distribution",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            "Content-Type": "application/json",
          }),
          body: JSON.stringify(request),
        })
      );
    });

    test("should successfully call distribution endpoint with multi mode", async () => {
      const mockResponse: DistributionResponse = {
        dimension: "domain",
        time_mode: "multi",
        buckets: [
          {
            time_label: "2024-Q4",
            total: 100,
            values: [{ key: "Clinical", count: 100, percent: 1.0 }],
          },
          {
            time_label: "2025-Q1",
            total: 150,
            values: [{ key: "Clinical", count: 150, percent: 1.0 }],
          },
        ],
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const request: DistributionRequest = {
        dimension: DimensionType.DOMAIN,
        time_mode: TimeMode.MULTI,
        time_windows: [
          { type: "season", value: "2024-Q4" },
          { type: "season", value: "2025-Q1" },
        ],
      };

      const response = await client.distribution(request);

      expect(response).toEqual(mockResponse);
      expect(response.buckets).toHaveLength(2);
    });

    test("should successfully call distribution endpoint with binary split mode", async () => {
      const mockResponse: DistributionResponse = {
        dimension: "stage",
        time_mode: "binary_split",
        buckets: [
          {
            time_label: "Before",
            total: 50,
            values: [{ key: "Stage 1", count: 50, percent: 1.0 }],
          },
          {
            time_label: "After",
            total: 150,
            values: [{ key: "Stage 1", count: 150, percent: 1.0 }],
          },
        ],
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const request: DistributionRequest = {
        dimension: DimensionType.STAGE,
        time_mode: TimeMode.BINARY_SPLIT,
        split_date: "2024-06-01",
      };

      const response = await client.distribution(request);

      expect(response).toEqual(mockResponse);
      expect(response.buckets).toHaveLength(2);
    });

    test("should include filters in request", async () => {
      const mockResponse: DistributionResponse = {
        dimension: "severity",
        time_mode: "single",
        buckets: [
          {
            time_label: "2025",
            total: 100,
            values: [{ key: "High", count: 100, percent: 1.0 }],
          },
        ],
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const request: DistributionRequest = {
        dimension: DimensionType.SEVERITY,
        time_mode: TimeMode.SINGLE,
        time_window: { type: "year", value: 2025 },
        filters: {
          department_id: 42,
          severity: "High",
        },
      };

      await client.distribution(request);

      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/operators/distribution",
        expect.objectContaining({
          body: JSON.stringify(request),
        })
      );
    });
  });

  describe("distribution() - Error Handling", () => {
    test("should throw OperatorsAPIError on HTTP 400 error", async () => {
      const errorResponse = {
        detail: "Invalid request parameters",
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: async () => errorResponse,
      });

      const request: DistributionRequest = {
        dimension: DimensionType.SEVERITY,
        time_mode: TimeMode.SINGLE,
        time_window: { type: "year", value: 2025 },
      };

      try {
        await client.distribution(request);
        fail("Expected OperatorsAPIError to be thrown");
      } catch (error) {
        expect(error).toBeInstanceOf(OperatorsAPIError);
        expect((error as OperatorsAPIError).message).toBe(
          "API request failed: Bad Request"
        );
      }
    });

    test("should throw OperatorsAPIError on HTTP 422 validation error", async () => {
      const errorResponse = {
        detail: [
          {
            loc: ["body", "dimension"],
            msg: "field required",
            type: "value_error.missing",
          },
        ],
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 422,
        statusText: "Unprocessable Entity",
        json: async () => errorResponse,
      });

      const request: DistributionRequest = {
        dimension: DimensionType.SEVERITY,
        time_mode: TimeMode.SINGLE,
        time_window: { type: "year", value: 2025 },
      };

      await expect(client.distribution(request)).rejects.toThrow(
        OperatorsAPIError
      );
    });

    test("should throw OperatorsAPIError on HTTP 500 server error", async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: async () => ({ detail: "Internal server error" }),
      });

      const request: DistributionRequest = {
        dimension: DimensionType.SEVERITY,
        time_mode: TimeMode.SINGLE,
        time_window: { type: "year", value: 2025 },
      };

      await expect(client.distribution(request)).rejects.toThrow(
        OperatorsAPIError
      );
    });

    test("should handle non-JSON error responses", async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: async () => {
          throw new Error("Not JSON");
        },
        text: async () => "Plain text error",
      });

      const request: DistributionRequest = {
        dimension: DimensionType.SEVERITY,
        time_mode: TimeMode.SINGLE,
        time_window: { type: "year", value: 2025 },
      };

      await expect(client.distribution(request)).rejects.toThrow(
        OperatorsAPIError
      );
    });

    test("should throw OperatorsAPIError on network error", async () => {
      (global.fetch as jest.Mock).mockRejectedValueOnce(
        new Error("Network error")
      );

      const request: DistributionRequest = {
        dimension: DimensionType.SEVERITY,
        time_mode: TimeMode.SINGLE,
        time_window: { type: "year", value: 2025 },
      };

      try {
        await client.distribution(request);
        fail("Expected OperatorsAPIError to be thrown");
      } catch (error) {
        expect(error).toBeInstanceOf(OperatorsAPIError);
        expect((error as OperatorsAPIError).message).toBe(
          "Network error: Network error"
        );
      }
    });

    test("should throw OperatorsAPIError on timeout", async () => {
      const shortTimeoutClient = new OperatorsClient({
        baseUrl: "http://localhost:8000",
        timeout: 100,
      });

      // Mock fetch to simulate an aborted request
      (global.fetch as jest.Mock).mockImplementationOnce(() => {
        return new Promise((_, reject) => {
          setTimeout(() => {
            const error = new Error("The operation was aborted");
            error.name = "AbortError";
            reject(error);
          }, 50);
        });
      });

      const request: DistributionRequest = {
        dimension: DimensionType.SEVERITY,
        time_mode: TimeMode.SINGLE,
        time_window: { type: "year", value: 2025 },
      };

      try {
        await shortTimeoutClient.distribution(request);
        fail("Expected OperatorsAPIError to be thrown");
      } catch (error) {
        expect(error).toBeInstanceOf(OperatorsAPIError);
        expect((error as OperatorsAPIError).message).toBe(
          "Request timeout after 100ms"
        );
      }
    });
  });

  describe("OperatorsAPIError Class", () => {
    test("should create error with message only", () => {
      const error = new OperatorsAPIError("Test error");

      expect(error.message).toBe("Test error");
      expect(error.name).toBe("OperatorsAPIError");
      expect(error.status).toBeUndefined();
      expect(error.response).toBeUndefined();
    });

    test("should create error with status and response", () => {
      const error = new OperatorsAPIError("Test error", 400, {
        detail: "Bad request",
      });

      expect(error.message).toBe("Test error");
      expect(error.status).toBe(400);
      expect(error.response).toEqual({ detail: "Bad request" });
    });

    test("should be instance of Error", () => {
      const error = new OperatorsAPIError("Test error");

      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(OperatorsAPIError);
    });
  });
});
