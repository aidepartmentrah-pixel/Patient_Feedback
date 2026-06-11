/*
================================================================================
PHASE RCA — STEP 3: CAUSE/ACTION PAIRING
================================================================================
Purpose: Link each RCA "CAUSE" suggestion to its corresponding "ACTION_ITEM"
         suggestion via a self-referencing PairedSuggestionID column, so the
         Settings UI can manage and display them as a single Cause->Action pair
         instead of two independently-ordered lists.

Date: 2026-06-11

SAFE TO RUN MULTIPLE TIMES (idempotent checks included)

Changes:
- Add nullable PairedSuggestionID column to APP_RCASuggestion
- Add self-referencing FK to APP_RCASuggestion(SuggestionID)
- Add filtered unique index on PairedSuggestionID (NULLs allowed/ignored)

No existing columns, rows, or constraints are modified or removed.
================================================================================
*/

BEGIN TRANSACTION;

BEGIN TRY

    PRINT '========================================';
    PRINT 'Phase RCA - Step 3: Cause/Action Pairing';
    PRINT '========================================';
    PRINT '';

    -- ========================================================================
    -- Add PairedSuggestionID column
    -- ========================================================================

    IF NOT EXISTS (
        SELECT * FROM sys.columns
        WHERE object_id = OBJECT_ID('dbo.APP_RCASuggestion')
        AND name = 'PairedSuggestionID'
    )
    BEGIN
        PRINT 'Adding column: PairedSuggestionID...';

        ALTER TABLE dbo.APP_RCASuggestion
        ADD PairedSuggestionID INT NULL;

        PRINT 'PairedSuggestionID column added successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT 'PairedSuggestionID column already exists';
        PRINT '';
    END

    -- ========================================================================
    -- Add self-referencing FK constraint
    -- ========================================================================

    IF NOT EXISTS (
        SELECT * FROM sys.foreign_keys
        WHERE name = 'FK_APP_RCASuggestion_PairedSuggestion'
    )
    BEGIN
        PRINT 'Adding constraint: FK_APP_RCASuggestion_PairedSuggestion...';

        ALTER TABLE dbo.APP_RCASuggestion
        ADD CONSTRAINT FK_APP_RCASuggestion_PairedSuggestion
            FOREIGN KEY (PairedSuggestionID)
            REFERENCES dbo.APP_RCASuggestion(SuggestionID);

        PRINT 'FK_APP_RCASuggestion_PairedSuggestion added successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT 'FK_APP_RCASuggestion_PairedSuggestion already exists';
        PRINT '';
    END

    -- ========================================================================
    -- Add filtered unique index on PairedSuggestionID
    -- ========================================================================

    IF NOT EXISTS (
        SELECT * FROM sys.indexes
        WHERE object_id = OBJECT_ID('dbo.APP_RCASuggestion')
        AND name = 'UQ_APP_RCASuggestion_PairedSuggestionID'
    )
    BEGIN
        PRINT 'Adding index: UQ_APP_RCASuggestion_PairedSuggestionID...';

        CREATE UNIQUE INDEX UQ_APP_RCASuggestion_PairedSuggestionID
            ON dbo.APP_RCASuggestion(PairedSuggestionID)
            WHERE PairedSuggestionID IS NOT NULL;

        PRINT 'UQ_APP_RCASuggestion_PairedSuggestionID added successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT 'UQ_APP_RCASuggestion_PairedSuggestionID already exists';
        PRINT '';
    END

    -- ========================================================================
    -- Commit Transaction
    -- ========================================================================

    COMMIT TRANSACTION;

    PRINT '';
    PRINT '========================================';
    PRINT 'Phase RCA Step 3 completed successfully';
    PRINT '========================================';
    PRINT '';

END TRY
BEGIN CATCH

    IF @@TRANCOUNT > 0
        ROLLBACK TRANSACTION;

    PRINT '';
    PRINT '========================================';
    PRINT 'ERROR: Phase RCA Step 3 failed';
    PRINT '========================================';
    PRINT 'Error: ' + ERROR_MESSAGE();
    PRINT 'Line: ' + CAST(ERROR_LINE() AS NVARCHAR(10));
    PRINT '';

    THROW;

END CATCH;

GO
