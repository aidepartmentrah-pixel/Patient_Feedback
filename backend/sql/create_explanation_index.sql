-- ============================================================================
-- Performance Index for Explanation Query
-- ============================================================================
-- This index optimizes the get_pending_explanations query by covering the
-- WHERE clause conditions:
-- - ClinicalRiskTypeID IN (2, 3) OR RequiresExplanation = 1
-- - ExplanationStatusID IS NULL OR ExplanationStatusID = 1
-- - CaseStatusID IN (1, 2)
--
-- Impact: Reduces query time from potential table scan to index seek
-- ============================================================================

USE IncidentManager;
GO

-- Check if index already exists
IF NOT EXISTS (
    SELECT 1 
    FROM sys.indexes 
    WHERE name = 'IX_APP_IncidentCase_ExplanationLookup' 
    AND object_id = OBJECT_ID('dbo.APP_IncidentCase')
)
BEGIN
    PRINT 'Creating index IX_APP_IncidentCase_ExplanationLookup...';
    
    CREATE NONCLUSTERED INDEX IX_APP_IncidentCase_ExplanationLookup
    ON dbo.APP_IncidentCase (
        CaseStatusID,
        ExplanationStatusID,
        ClinicalRiskTypeID,
        RequiresExplanation
    )
    INCLUDE (
        IncidentRequestCaseID,
        ComplaintText,
        PatientName,
        FeedbackRecievedDate,
        CreatedAt
    );
    
    PRINT 'Index created successfully.';
END
ELSE
BEGIN
    PRINT 'Index IX_APP_IncidentCase_ExplanationLookup already exists.';
END
GO

-- Optional: Display index info
SELECT 
    i.name AS IndexName,
    i.type_desc AS IndexType,
    ds.name AS FileGroup,
    p.rows AS RowCount
FROM sys.indexes i
INNER JOIN sys.data_spaces ds ON i.data_space_id = ds.data_space_id
INNER JOIN sys.partitions p ON i.object_id = p.object_id AND i.index_id = p.index_id
WHERE i.object_id = OBJECT_ID('dbo.APP_IncidentCase')
AND i.name = 'IX_APP_IncidentCase_ExplanationLookup';
GO
