/*
====================================================================
REMOVE PATIENT FOREIGN KEY CONSTRAINTS
====================================================================
Purpose: Drop foreign key constraints that prevent reserve patients
         from being used in the system

WARNING: Only run this if check_patient_fk_constraints.sql found
         existing constraints that need to be removed.

Strategy: FK constraints cannot point to two tables (hospital + reserve)
          so we must remove them to allow dual-source pattern.

Author: System
Date: 2026-01-20
====================================================================
*/

PRINT '======================================================================'
PRINT 'REMOVING PATIENT-RELATED FOREIGN KEY CONSTRAINTS'
PRINT '======================================================================'
PRINT ''

-- Dynamically drop all patient-related FK constraints
DECLARE @sql NVARCHAR(MAX) = ''
DECLARE @constraintName NVARCHAR(255)
DECLARE @tableName NVARCHAR(255)

DECLARE constraint_cursor CURSOR FOR
SELECT 
    fk.name AS ForeignKeyName,
    OBJECT_NAME(fk.parent_object_id) AS TableName
FROM 
    sys.foreign_keys AS fk
    INNER JOIN sys.foreign_key_columns AS fkc ON fk.object_id = fkc.constraint_object_id
WHERE 
    COL_NAME(fkc.parent_object_id, fkc.parent_column_id) LIKE '%Patient%'
    OR COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) LIKE '%Patient%'
    OR OBJECT_NAME(fk.referenced_object_id) LIKE '%Patient%'

OPEN constraint_cursor

FETCH NEXT FROM constraint_cursor INTO @constraintName, @tableName

WHILE @@FETCH_STATUS = 0
BEGIN
    PRINT 'Dropping FK constraint: ' + @constraintName + ' from table: ' + @tableName
    
    SET @sql = 'ALTER TABLE [dbo].[' + @tableName + '] DROP CONSTRAINT [' + @constraintName + ']'
    
    BEGIN TRY
        EXEC sp_executesql @sql
        PRINT '  ✓ Successfully dropped: ' + @constraintName
    END TRY
    BEGIN CATCH
        PRINT '  ✗ Error dropping ' + @constraintName + ': ' + ERROR_MESSAGE()
    END CATCH
    
    PRINT ''
    
    FETCH NEXT FROM constraint_cursor INTO @constraintName, @tableName
END

CLOSE constraint_cursor
DEALLOCATE constraint_cursor

PRINT '======================================================================'
PRINT 'FOREIGN KEY REMOVAL COMPLETE'
PRINT '======================================================================'
PRINT ''
PRINT 'Note: This is expected behavior for the dual-source pattern.'
PRINT 'The application will handle referential integrity in the DB layer.'
PRINT '======================================================================'
GO
