-- Confirms every universal lookup table was actually seeded (non-empty).
-- Expect: every row_count > 0. A zero means seeding silently failed for that table.

SELECT 'dbo.APP_LOOKUP_BUILDING' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_BUILDING]
UNION ALL
SELECT 'dbo.APP_LOOKUP_CASE_STAGE' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_CASE_STAGE]
UNION ALL
SELECT 'dbo.APP_LOOKUP_CASE_STATUS' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_CASE_STATUS]
UNION ALL
SELECT 'dbo.APP_LOOKUP_CATEGORY' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_CATEGORY]
UNION ALL
SELECT 'dbo.APP_LOOKUP_CLASSIFICATION' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_CLASSIFICATION]
UNION ALL
SELECT 'dbo.APP_LOOKUP_CLINICAL_RISK_TYPE' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE]
UNION ALL
SELECT 'dbo.APP_LOOKUP_DOMAIN' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_DOMAIN]
UNION ALL
SELECT 'dbo.APP_LOOKUP_EXPLANATION_STATUS' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_EXPLANATION_STATUS]
UNION ALL
SELECT 'dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_FEEDBACK_INTENT_TYPE]
UNION ALL
SELECT 'dbo.APP_LOOKUP_HARM_LEVEL' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_HARM_LEVEL]
UNION ALL
SELECT 'dbo.APP_LOOKUP_RECORD_TYPE' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_RECORD_TYPE]
UNION ALL
SELECT 'dbo.APP_Lookup_SatisfactionStatus' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_Lookup_SatisfactionStatus]
UNION ALL
SELECT 'dbo.APP_LOOKUP_SEVERITY' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_SEVERITY]
UNION ALL
SELECT 'dbo.APP_LOOKUP_SOURCE' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_SOURCE]
UNION ALL
SELECT 'dbo.APP_Lookup_SubcaseActionItemStatus' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_Lookup_SubcaseActionItemStatus]
UNION ALL
SELECT 'dbo.APP_Lookup_SubcaseStatus' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_Lookup_SubcaseStatus]
UNION ALL
SELECT 'dbo.APP_Lookup_SubcaseType' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_Lookup_SubcaseType]
UNION ALL
SELECT 'dbo.APP_LOOKUP_SUBCATEGORY' AS table_name, COUNT(*) AS row_count FROM [dbo].[APP_LOOKUP_SUBCATEGORY]
UNION ALL
SELECT 'ml.EmbeddingModelVersion' AS table_name, COUNT(*) AS row_count FROM [ml].[EmbeddingModelVersion];