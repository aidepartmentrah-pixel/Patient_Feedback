/*
 * Fix Foreign Key Constraint for Reserve Doctors
 * 
 * Issue: APP_IncidentCaseDoctor has FK constraint FK_ICDoctor_Doctor 
 *        that only allows DoctorID from APP_LOOKUP_DOCTOR.
 *        This prevents reserve doctors (from APP_RESERVE_DOCTOR) from being used in incidents.
 * 
 * Solution: Drop the existing FK constraint since we can't create a FK that references
 *           two tables. The validation is handled in application code (insert_service.py)
 *           using UNION query to check both tables.
 * 
 * Database: IncidentManager
 * Table: APP_IncidentCaseDoctor
 * Constraint: FK_ICDoctor_Doctor
 */

USE IncidentManager;
GO

-- Check current constraint
PRINT 'Current FK constraint on APP_IncidentCaseDoctor:';
SELECT 
    fk.name AS ConstraintName,
    tp.name AS ParentTable,
    cp.name AS ParentColumn,
    tr.name AS ReferencedTable,
    cr.name AS ReferencedColumn
FROM sys.foreign_keys AS fk
INNER JOIN sys.tables AS tp ON fk.parent_object_id = tp.object_id
INNER JOIN sys.tables AS tr ON fk.referenced_object_id = tr.object_id  
INNER JOIN sys.foreign_key_columns AS fkc ON fk.object_id = fkc.constraint_object_id
INNER JOIN sys.columns AS cp ON fkc.parent_column_id = cp.column_id AND fkc.parent_object_id = cp.object_id
INNER JOIN sys.columns AS cr ON fkc.referenced_column_id = cr.column_id AND fkc.referenced_object_id = cr.object_id
WHERE fk.name = 'FK_ICDoctor_Doctor'
GO

-- Drop the constraint
PRINT 'Dropping FK constraint FK_ICDoctor_Doctor...';

IF EXISTS (SELECT * FROM sys.foreign_keys WHERE name = 'FK_ICDoctor_Doctor')
BEGIN
    ALTER TABLE dbo.APP_IncidentCaseDoctor
    DROP CONSTRAINT FK_ICDoctor_Doctor;
    
    PRINT '✓ FK constraint FK_ICDoctor_Doctor dropped successfully';
END
ELSE
BEGIN
    PRINT '⚠ FK constraint FK_ICDoctor_Doctor not found (may already be dropped)';
END
GO

-- Verify constraint is removed
PRINT '';
PRINT 'Verification - Remaining FK constraints on APP_IncidentCaseDoctor:';
SELECT 
    fk.name AS ConstraintName,
    tp.name AS ParentTable,
    cp.name AS ParentColumn,
    tr.name AS ReferencedTable,
    cr.name AS ReferencedColumn
FROM sys.foreign_keys AS fk
INNER JOIN sys.tables AS tp ON fk.parent_object_id = tp.object_id
INNER JOIN sys.tables AS tr ON fk.referenced_object_id = tr.object_id  
INNER JOIN sys.foreign_key_columns AS fkc ON fk.object_id = fkc.constraint_object_id
INNER JOIN sys.columns AS cp ON fkc.parent_column_id = cp.column_id AND fkc.parent_object_id = cp.object_id
INNER JOIN sys.columns AS cr ON fkc.referenced_column_id = cr.column_id AND fkc.referenced_object_id = cr.object_id
WHERE tp.name = 'APP_IncidentCaseDoctor'
GO

PRINT '';
PRINT '====================================================================';
PRINT 'FK Constraint Removal Complete';
PRINT '';
PRINT 'NOTE: Doctor ID validation is now handled in application code';
PRINT '      (backend/api/services/insert_service.py lines 195-214)';
PRINT '      using UNION query to check both APP_LOOKUP_DOCTOR and APP_RESERVE_DOCTOR';
PRINT '====================================================================';
GO
