/*
================================================================================
Phase G-B1: Drawer Notes - Create Drawer_Note Table
================================================================================
Purpose: Create the APP_DrawerNote table for drawer notes feature
Author: Phase G Implementation
Date: 2026-02-07
Version: 1.0

IMPORTANT: Safe to run multiple times (uses IF NOT EXISTS checks)

Table Created:
- APP_DrawerNote - Stores drawer-style notes (not part of workflow/reporting)

Design Decisions:
- Notes are editable (text overwrite allowed)
- NO edit history columns
- NO optional metadata
- Soft delete only (is_deleted flag)
- No org scope columns
- No workflow linkage

Access Control:
- SOFTWARE_ADMIN
- COMPLAINT_SUPERVISOR
- WORKER

================================================================================
*/

BEGIN TRANSACTION;

BEGIN TRY

    PRINT '========================================';
    PRINT 'Phase G-B1: Creating Drawer Note Table';
    PRINT '========================================';
    PRINT '';

    -- ========================================================================
    -- 1) APP_DrawerNote Table
    -- ========================================================================
    -- Stores drawer notes (side notes, not part of main workflow)
    
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'APP_DrawerNote' AND schema_id = SCHEMA_ID('dbo'))
    BEGIN
        PRINT 'Creating table: APP_DrawerNote...';
        
        CREATE TABLE dbo.APP_DrawerNote (
            -- Primary Key
            NoteID INT IDENTITY(1,1) PRIMARY KEY,
            
            -- Note Content
            NoteText NVARCHAR(MAX) NOT NULL,
            
            -- Audit Fields
            CreatedAt DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
            CreatedByUserID INT NOT NULL,
            CreatedByName NVARCHAR(200) NOT NULL,
            
            -- Soft Delete Flag
            IsDeleted BIT NOT NULL DEFAULT 0,
            
            -- Indexes for performance
            INDEX IX_DrawerNote_CreatedAt NONCLUSTERED (CreatedAt DESC),
            INDEX IX_DrawerNote_CreatedByUserID NONCLUSTERED (CreatedByUserID),
            INDEX IX_DrawerNote_IsDeleted NONCLUSTERED (IsDeleted)
        );
        
        PRINT '✓ APP_DrawerNote table created successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT '✓ APP_DrawerNote table already exists';
        PRINT '';
    END

    -- ========================================================================
    -- Verification Query
    -- ========================================================================
    PRINT 'Verifying table structure...';
    PRINT '';
    
    SELECT 
        COLUMN_NAME,
        DATA_TYPE,
        CHARACTER_MAXIMUM_LENGTH,
        IS_NULLABLE,
        COLUMN_DEFAULT
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'APP_DrawerNote'
    ORDER BY ORDINAL_POSITION;
    
    PRINT '';
    PRINT '========================================';
    PRINT 'Phase G-B1: Completed Successfully';
    PRINT '========================================';
    
    COMMIT TRANSACTION;
    PRINT 'Transaction committed.';

END TRY
BEGIN CATCH
    
    ROLLBACK TRANSACTION;
    
    PRINT '';
    PRINT '========================================';
    PRINT 'ERROR: Transaction rolled back';
    PRINT '========================================';
    PRINT 'Error Message: ' + ERROR_MESSAGE();
    PRINT 'Error Line: ' + CAST(ERROR_LINE() AS NVARCHAR(10));
    PRINT '';
    
    THROW;
    
END CATCH;

GO

/*
================================================================================
ROLLBACK SCRIPT (if needed)
================================================================================
-- Uncomment and run if you need to remove the table

-- DROP TABLE IF EXISTS dbo.APP_DrawerNote;
-- PRINT 'APP_DrawerNote table dropped';

================================================================================
*/
