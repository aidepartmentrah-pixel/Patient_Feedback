/*
====================================================================
CHECK PATIENT FOREIGN KEY CONSTRAINTS
====================================================================
Purpose: Find all foreign key constraints referencing PatientAdmissionID
         or patient-related columns

This script will identify any FK constraints that might prevent
reserve patients from being used in the system.

Author: System
Date: 2026-01-20
====================================================================
*/

PRINT '======================================================================'
PRINT 'CHECKING FOR PATIENT-RELATED FOREIGN KEY CONSTRAINTS'
PRINT '======================================================================'
PRINT ''

-- Check for FK constraints on PatientAdmissionID
PRINT '1. Foreign Keys Referencing PatientAdmissionID:'
PRINT '----------------------------------------------------------------'

SELECT 
    fk.name AS ForeignKeyName,
    OBJECT_NAME(fk.parent_object_id) AS TableName,
    COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS ColumnName,
    OBJECT_NAME(fk.referenced_object_id) AS ReferencedTable,
    COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS ReferencedColumn
FROM 
    sys.foreign_keys AS fk
    INNER JOIN sys.foreign_key_columns AS fkc ON fk.object_id = fkc.constraint_object_id
WHERE 
    COL_NAME(fkc.parent_object_id, fkc.parent_column_id) LIKE '%Patient%'
    OR COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) LIKE '%Patient%'
    OR OBJECT_NAME(fk.referenced_object_id) LIKE '%Patient%'
ORDER BY 
    TableName, ForeignKeyName
GO

-- Check if APP_IncidentCase has any patient FK
PRINT ''
PRINT '2. APP_IncidentCase Patient-Related Columns:'
PRINT '----------------------------------------------------------------'

SELECT 
    c.name AS ColumnName,
    t.name AS DataType,
    c.max_length AS MaxLength,
    c.is_nullable AS IsNullable
FROM 
    sys.columns c
    INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
WHERE 
    c.object_id = OBJECT_ID('APP_IncidentCase')
    AND c.name LIKE '%Patient%'
ORDER BY 
    c.column_id
GO

-- Check for any FK constraints on APP_IncidentCase
PRINT ''
PRINT '3. All Foreign Keys on APP_IncidentCase:'
PRINT '----------------------------------------------------------------'

SELECT 
    fk.name AS ForeignKeyName,
    COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS ColumnName,
    OBJECT_NAME(fk.referenced_object_id) AS ReferencedTable,
    COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS ReferencedColumn
FROM 
    sys.foreign_keys AS fk
    INNER JOIN sys.foreign_key_columns AS fkc ON fk.object_id = fkc.constraint_object_id
WHERE 
    OBJECT_NAME(fk.parent_object_id) = 'APP_IncidentCase'
ORDER BY 
    ForeignKeyName
GO

-- Summary
PRINT ''
PRINT '======================================================================'
PRINT 'SUMMARY'
PRINT '======================================================================'
PRINT 'If no foreign keys were found referencing patient tables,'
PRINT 'then no constraints need to be removed.'
PRINT ''
PRINT 'If foreign keys exist, they must be removed to allow reserve patients'
PRINT 'to be used in incident cases (dual-source pattern).'
PRINT '======================================================================'
GO
