/**
 * Operators API Client
 * 
 * TypeScript client for interacting with the Graph Operators API.
 * Provides type-safe methods for calling operator endpoints.
 */

import {
  DistributionRequest,
  DistributionResponse,
} from "../types/operators/distribution";

/**
 * API Client configuration options.
 */
export interface OperatorsClientConfig {
  baseUrl?: string; // Default: http://localhost:8000
  headers?: Record<string, string>; // Additional headers
  timeout?: number; // Request timeout in milliseconds
}

/**
 * API Error with additional context.
 */
export class OperatorsAPIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: any
  ) {
    super(message);
    this.name = "OperatorsAPIError";
  }
}

/**
 * Operators API Client
 * 
 * Main client class for interacting with the Graph Operators API.
 * 
 * @example
 * ```typescript
 * const client = new OperatorsClient({ baseUrl: 'http://localhost:8000' });
 * 
 * const response = await client.distribution({
 *   dimension: DimensionType.SEVERITY,
 *   time_mode: TimeMode.SINGLE,
 *   time_window: { type: 'year', value: 2025 }
 * });
 * ```
 */
export class OperatorsClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeout: number;

  constructor(config: OperatorsClientConfig = {}) {
    this.baseUrl = config.baseUrl || "http://localhost:8000";
    this.headers = {
      "Content-Type": "application/json",
      ...config.headers,
    };
    this.timeout = config.timeout || 30000; // 30 seconds default
  }

  /**
   * Call the Distribution Operator endpoint.
   * 
   * @param request - Distribution operator request
   * @returns Distribution operator response
   * @throws OperatorsAPIError if the request fails
   * 
   * @example
   * ```typescript
   * // Single mode - one year
   * const response = await client.distribution({
   *   dimension: DimensionType.SEVERITY,
   *   time_mode: TimeMode.SINGLE,
   *   time_window: { type: 'year', value: 2025 }
   * });
   * 
   * // Multi mode - compare quarters
   * const comparison = await client.distribution({
   *   dimension: DimensionType.DOMAIN,
   *   time_mode: TimeMode.MULTI,
   *   time_windows: [
   *     { type: 'season', value: '2024-Q4' },
   *     { type: 'season', value: '2025-Q1' }
   *   ]
   * });
   * 
   * // Binary split - before/after a date
   * const split = await client.distribution({
   *   dimension: DimensionType.STAGE,
   *   time_mode: TimeMode.BINARY_SPLIT,
   *   split_date: '2024-06-01'
   * });
   * ```
   */
  async distribution(
    request: DistributionRequest
  ): Promise<DistributionResponse> {
    return this.post<DistributionResponse>(
      "/api/operators/distribution",
      request
    );
  }

  /**
   * Generic POST request method.
   * 
   * @private
   */
  private async post<T>(endpoint: string, body: any): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const url = `${this.baseUrl}${endpoint}`;
      const response = await fetch(url, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorData: any;
        try {
          errorData = await response.json();
        } catch {
          errorData = await response.text();
        }

        throw new OperatorsAPIError(
          `API request failed: ${response.statusText}`,
          response.status,
          errorData
        );
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof OperatorsAPIError) {
        throw error;
      }

      if (error instanceof Error) {
        if (error.name === "AbortError") {
          throw new OperatorsAPIError(
            `Request timeout after ${this.timeout}ms`
          );
        }
        throw new OperatorsAPIError(`Network error: ${error.message}`);
      }

      throw new OperatorsAPIError("Unknown error occurred");
    }
  }
}

/**
 * Default client instance for convenience.
 * Uses default configuration (localhost:8000).
 * 
 * @example
 * ```typescript
 * import { defaultClient } from './api/operators-client';
 * 
 * const response = await defaultClient.distribution({
 *   dimension: DimensionType.SEVERITY,
 *   time_mode: TimeMode.SINGLE,
 *   time_window: { type: 'year', value: 2025 }
 * });
 * ```
 */
export const defaultClient = new OperatorsClient();
