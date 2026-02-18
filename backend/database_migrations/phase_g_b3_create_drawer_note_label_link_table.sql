/*
================================================================================
Phase G-B3: Drawer Note-Label Link - Create Bridge Table
================================================================================
Purpose: Create the APP_DrawerNoteLabelLink junction table for many-to-many
         relationships between drawer notes and labels
Author: Phase G Implementation
Date: 2026-02-07
Version: 1.0

IMPORTANT: Safe to run multiple times (uses IF NOT EXISTS checks)

Table Created:
- APP_DrawerNoteLabelLink - Bridge table linking notes to labels

Design Decisions:
- Pure bridge table (no extra columns)
- No metadata fields
- Composite primary key prevents duplicate links
- Foreign keys with CASCADE DELETE for data integrity
- Index on label_id for efficient label-based filtering

Relationships:
- Many-to-Many between APP_DrawerNote and APP_DrawerLabel
- A note can have multiple labels
- A label can be assigned to multiple notes

================================================================================
*/

BEGIN TRANSACTION;

BEGIN TRY

    PRINT '========================================';
    PRINT 'Phase G-B3: Creating Drawer Note-Label Link Table';
    PRINT '========================================';
    PRINT '';

    -- ========================================================================
    -- 1) APP_DrawerNoteLabelLink Table
    -- ========================================================================
    -- Pure bridge table for many-to-many relationship
    
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'APP_DrawerNoteLabelLink' AND schema_id = SCHEMA_ID('dbo'))
    BEGIN
        PRINT 'Creating table: APP_DrawerNoteLabelLink...';
        
        CREATE TABLE dbo.APP_DrawerNoteLabelLink (
            -- Foreign Key Columns
            NoteID INT NOT NULL,
            LabelID INT NOT NULL,
            
            -- Composite Primary Key (prevents duplicate links)
            CONSTRAINT PK_DrawerNoteLabelLink PRIMARY KEY (NoteID, LabelID),
            
            -- Foreign Key to APP_DrawerNote
            CONSTRAINT FK_DrawerNoteLabelLink_Note 
                FOREIGN KEY (NoteID) 
                REFERENCES dbo.APP_DrawerNote(NoteID) 
                ON DELETE CASCADE,
            
            -- Foreign Key to APP_DrawerLabel
            CONSTRAINT FK_DrawerNoteLabelLink_Label 
                FOREIGN KEY (LabelID) 
                REFERENCES dbo.APP_DrawerLabel(LabelID) 
                ON DELETE CASCADE,
            
            -- Index for label-based filtering
            INDEX IX_DrawerNoteLabelLink_LabelID NONCLUSTERED (LabelID)
        );
        
        PRINT '✓ APP_DrawerNoteLabelLink table created successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT '✓ APP_DrawerNoteLabelLink table already exists';
        PRINT '';
    END

    -- ========================================================================
    -- Verification Queries
    -- ========================================================================
    PRINT 'Verifying table structure...';
    PRINT '';
    
    SELECT 
        COLUMN_NAME,
        DATA_TYPE,
        IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'APP_DrawerNoteLabelLink'
    ORDER BY ORDINAL_POSITION;
    
    PRINT '';
    PRINT 'Verifying constraints...';
    PRINT '';
    
    SELECT 
        CONSTRAINT_NAME,
        CONSTRAINT_TYPE
    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
    WHERE TABLE_NAME = 'APP_DrawerNoteLabelLink';
    
    PRINT '';
    PRINT 'Verifying foreign keys...';
    PRINT '';
    
    SELECT 
        fk.name AS FK_Name,
        tp.name AS Parent_Table,
        cp.name AS Parent_Column,
        tr.name AS Referenced_Table,
        cr.name AS Referenced_Column
    FROM sys.foreign_keys AS fk
    INNER JOIN sys.foreign_key_columns AS fkc ON fk.object_id = fkc.constraint_object_id
    INNER JOIN sys.tables AS tp ON fkc.parent_object_id = tp.object_id
    INNER JOIN sys.columns AS cp ON fkc.parent_object_id = cp.object_id AND fkc.parent_column_id = cp.column_id
    INNER JOIN sys.tables AS tr ON fkc.referenced_object_id = tr.object_id
    INNER JOIN sys.columns AS cr ON fkc.referenced_object_id = cr.object_id AND fkc.referenced_column_id = cr.column_id
    WHERE tp.name = 'APP_DrawerNoteLabelLink';
    
    PRINT '';
    PRINT '========================================';
    PRINT 'Phase G-B3: Completed Successfully';
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

-- DROP TABLE IF EXISTS dbo.APP_DrawerNoteLabelLink;
-- PRINT 'APP_DrawerNoteLabelLink table dropped';

================================================================================
*/
