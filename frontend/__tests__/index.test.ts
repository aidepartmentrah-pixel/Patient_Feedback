/**
 * Tests for Main Export File
 * 
 * Ensures all exports are properly exposed.
 */

import * as indexExports from "../index";

describe("Index Exports", () => {
  test("should export all enums", () => {
    expect(indexExports.DimensionType).toBeDefined();
    expect(indexExports.TimeMode).toBeDefined();
    expect(indexExports.TimeWindowType).toBeDefined();
  });

  test("should export type guards", () => {
    expect(indexExports.isSingleModeRequest).toBeDefined();
    expect(indexExports.isMultiModeRequest).toBeDefined();
    expect(indexExports.isBinarySplitRequest).toBeDefined();
    expect(typeof indexExports.isSingleModeRequest).toBe("function");
    expect(typeof indexExports.isMultiModeRequest).toBe("function");
    expect(typeof indexExports.isBinarySplitRequest).toBe("function");
  });

  test("should export API client classes", () => {
    expect(indexExports.OperatorsClient).toBeDefined();
    expect(indexExports.OperatorsAPIError).toBeDefined();
  });

  test("should be able to create a client from export", () => {
    const client = new indexExports.OperatorsClient({ baseUrl: "http://test.com" });
    expect(client).toBeInstanceOf(indexExports.OperatorsClient);
  });

  test("should be able to create an error from export", () => {
    const error = new indexExports.OperatorsAPIError("Test error");
    expect(error).toBeInstanceOf(indexExports.OperatorsAPIError);
    expect(error).toBeInstanceOf(Error);
  });
});
