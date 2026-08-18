-- Confirms installation configuration tables were seeded.
-- Expect: row_count matches what was present in the source install at package-build time
-- (see docs/DATABASE_STRUCTURE_REPORT.md for the baseline counts).

SELECT 'dbo.APP_OrgUnitPolicy' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_OrgUnitPolicy]
UNION ALL
SELECT 'dbo.APP_DepartmentPolicy' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_DepartmentPolicy]
UNION ALL
SELECT 'dbo.APP_DepartmentEvaluationRule' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_DepartmentEvaluationRule]
UNION ALL
SELECT 'dbo.APP_Roles' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_Roles];