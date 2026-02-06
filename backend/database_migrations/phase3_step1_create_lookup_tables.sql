-- ============================================================
-- PHASE 3 - STEP 3.1: CREATE LOOKUP TABLES
-- ============================================================
-- Purpose: Add lookup tables for Phase 3 workflow system (Subcases, Action Items)
-- Safety: This script ONLY ADDS new tables. Does NOT modify existing objects.
-- Date: 2026-01-29
-- ============================================================

USE IncidentManager;
GO

PRINT '============================================================';
PRINT 'PHASE 3 - STEP 3.1: CREATING LOOKUP TABLES';
PRINT '============================================================';
PRINT '';

-- ============================================================
-- TABLE 1: APP_Lookup_SubcaseType
-- ============================================================
PRINT 'Creating TABLE: APP_Lookup_SubcaseType...';

IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'APP_Lookup_SubcaseType' AND SCHEMA_NAME(schema_id) = 'dbo')
BEGIN
    CREATE TABLE dbo.APP_Lookup_SubcaseType (
        CaseTypeCode NVARCHAR(50) NOT NULL PRIMARY KEY,
        CaseTypeNameEn NVARCHAR(100) NOT NULL,
        CaseTypeNameAr NVARCHAR(100) NULL,
        IsActive BIT NOT NULL DEFAULT 1
    );
    
    PRINT '✅ Table APP_Lookup_SubcaseType created successfully.';
END
ELSE
BEGIN
    PRINT '⚠️  Table APP_Lookup_SubcaseType already exists. Skipping creation.';
END
GO

-- Seed APP_Lookup_SubcaseType
PRINT 'Seeding APP_Lookup_SubcaseType...';

IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseType WHERE CaseTypeCode = 'INCIDENT_RESPONSE')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseType (CaseTypeCode, CaseTypeNameEn, CaseTypeNameAr, IsActive)
    VALUES ('INCIDENT_RESPONSE', 'Incident Response', N'معالجة شكوى حادثة', 1);
    PRINT '  ✅ Inserted: INCIDENT_RESPONSE';
END
ELSE
BEGIN
    PRINT '  ⚠️  INCIDENT_RESPONSE already exists. Skipping.';
END

IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseType WHERE CaseTypeCode = 'SEASONAL_REPORT_RESPONSE')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseType (CaseTypeCode, CaseTypeNameEn, CaseTypeNameAr, IsActive)
    VALUES ('SEASONAL_REPORT_RESPONSE', 'Seasonal Report Response', N'معالجة تقرير موسمي', 1);
    PRINT '  ✅ Inserted: SEASONAL_REPORT_RESPONSE';
END
ELSE
BEGIN
    PRINT '  ⚠️  SEASONAL_REPORT_RESPONSE already exists. Skipping.';
END
GO

PRINT '';

-- ============================================================
-- TABLE 2: APP_Lookup_SubcaseStatus
-- ============================================================
PRINT 'Creating TABLE: APP_Lookup_SubcaseStatus...';

IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'APP_Lookup_SubcaseStatus' AND SCHEMA_NAME(schema_id) = 'dbo')
BEGIN
    CREATE TABLE dbo.APP_Lookup_SubcaseStatus (
        StatusCode NVARCHAR(50) NOT NULL PRIMARY KEY,
        StatusNameEn NVARCHAR(100) NOT NULL,
        StatusNameAr NVARCHAR(100) NULL,
        DisplayOrder INT NOT NULL,
        IsFinal BIT NOT NULL DEFAULT 0,
        IsActive BIT NOT NULL DEFAULT 1
    );
    
    PRINT '✅ Table APP_Lookup_SubcaseStatus created successfully.';
END
ELSE
BEGIN
    PRINT '⚠️  Table APP_Lookup_SubcaseStatus already exists. Skipping creation.';
END
GO

-- Seed APP_Lookup_SubcaseStatus
PRINT 'Seeding APP_Lookup_SubcaseStatus...';

-- Status 1: SUBMITTED_TO_SECTION
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'SUBMITTED_TO_SECTION')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('SUBMITTED_TO_SECTION', 'Submitted to Section', N'مُرسل إلى القسم', 1, 0, 1);
    PRINT '  ✅ Inserted: SUBMITTED_TO_SECTION';
END
ELSE
BEGIN
    PRINT '  ⚠️  SUBMITTED_TO_SECTION already exists. Skipping.';
END

-- Status 2: SECTION_DENIED
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'SECTION_DENIED')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('SECTION_DENIED', 'Section Denied', N'مرفوض من القسم', 2, 0, 1);
    PRINT '  ✅ Inserted: SECTION_DENIED';
END
ELSE
BEGIN
    PRINT '  ⚠️  SECTION_DENIED already exists. Skipping.';
END

-- Status 3: SECTION_ACCEPTED_PENDING_DEPT
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'SECTION_ACCEPTED_PENDING_DEPT')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('SECTION_ACCEPTED_PENDING_DEPT', 'Section Accepted - Pending Department', N'موافقة القسم - في انتظار الإدارة', 3, 0, 1);
    PRINT '  ✅ Inserted: SECTION_ACCEPTED_PENDING_DEPT';
END
ELSE
BEGIN
    PRINT '  ⚠️  SECTION_ACCEPTED_PENDING_DEPT already exists. Skipping.';
END

-- Status 4: DEPT_REJECTED
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'DEPT_REJECTED')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('DEPT_REJECTED', 'Department Rejected', N'مرفوض من الإدارة', 4, 0, 1);
    PRINT '  ✅ Inserted: DEPT_REJECTED';
END
ELSE
BEGIN
    PRINT '  ⚠️  DEPT_REJECTED already exists. Skipping.';
END

-- Status 5: DEPT_ACCEPTED_PENDING_ADMIN
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'DEPT_ACCEPTED_PENDING_ADMIN')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('DEPT_ACCEPTED_PENDING_ADMIN', 'Department Accepted - Pending Administration', N'موافقة الإدارة - في انتظار الإدارة العليا', 5, 0, 1);
    PRINT '  ✅ Inserted: DEPT_ACCEPTED_PENDING_ADMIN';
END
ELSE
BEGIN
    PRINT '  ⚠️  DEPT_ACCEPTED_PENDING_ADMIN already exists. Skipping.';
END

-- Status 6: ADMIN_REJECTED
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'ADMIN_REJECTED')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('ADMIN_REJECTED', 'Administration Rejected', N'مرفوض من الإدارة العليا', 6, 0, 1);
    PRINT '  ✅ Inserted: ADMIN_REJECTED';
END
ELSE
BEGIN
    PRINT '  ⚠️  ADMIN_REJECTED already exists. Skipping.';
END

-- Status 7: ADMIN_APPROVED
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'ADMIN_APPROVED')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('ADMIN_APPROVED', 'Administration Approved', N'موافق من الإدارة العليا', 7, 0, 1);
    PRINT '  ✅ Inserted: ADMIN_APPROVED';
END
ELSE
BEGIN
    PRINT '  ⚠️  ADMIN_APPROVED already exists. Skipping.';
END

-- Status 8: CLOSED
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'CLOSED')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('CLOSED', 'Closed', N'مغلق', 8, 1, 1);
    PRINT '  ✅ Inserted: CLOSED';
END
ELSE
BEGIN
    PRINT '  ⚠️  CLOSED already exists. Skipping.';
END

-- Status 9: FORCE_CLOSED
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'FORCE_CLOSED')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('FORCE_CLOSED', 'Force Closed', N'إغلاق إجباري', 9, 1, 1);
    PRINT '  ✅ Inserted: FORCE_CLOSED';
END
ELSE
BEGIN
    PRINT '  ⚠️  FORCE_CLOSED already exists. Skipping.';
END
GO

PRINT '';

-- ============================================================
-- TABLE 3: APP_Lookup_SubcaseActionItemStatus
-- ============================================================
PRINT 'Creating TABLE: APP_Lookup_SubcaseActionItemStatus...';

IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'APP_Lookup_SubcaseActionItemStatus' AND SCHEMA_NAME(schema_id) = 'dbo')
BEGIN
    CREATE TABLE dbo.APP_Lookup_SubcaseActionItemStatus (
        StatusCode NVARCHAR(50) NOT NULL PRIMARY KEY,
        StatusNameEn NVARCHAR(100) NOT NULL,
        StatusNameAr NVARCHAR(100) NULL,
        DisplayOrder INT NOT NULL,
        IsActive BIT NOT NULL DEFAULT 1,
        IsFinal BIT NOT NULL DEFAULT 0
    );
    
    PRINT '✅ Table APP_Lookup_SubcaseActionItemStatus created successfully.';
END
ELSE
BEGIN
    PRINT '⚠️  Table APP_Lookup_SubcaseActionItemStatus already exists. Skipping creation.';
END
GO

-- Seed APP_Lookup_SubcaseActionItemStatus
PRINT 'Seeding APP_Lookup_SubcaseActionItemStatus...';

-- Status 1: DRAFT
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseActionItemStatus WHERE StatusCode = 'DRAFT')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseActionItemStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsActive, IsFinal)
    VALUES ('DRAFT', 'Draft', N'مسودة', 1, 1, 0);
    PRINT '  ✅ Inserted: DRAFT';
END
ELSE
BEGIN
    PRINT '  ⚠️  DRAFT already exists. Skipping.';
END

-- Status 2: SUBMITTED_TO_DEPT
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseActionItemStatus WHERE StatusCode = 'SUBMITTED_TO_DEPT')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseActionItemStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsActive, IsFinal)
    VALUES ('SUBMITTED_TO_DEPT', 'Submitted to Department', N'مُرسل إلى الإدارة', 2, 1, 0);
    PRINT '  ✅ Inserted: SUBMITTED_TO_DEPT';
END
ELSE
BEGIN
    PRINT '  ⚠️  SUBMITTED_TO_DEPT already exists. Skipping.';
END

-- Status 3: DEPT_REJECTED
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseActionItemStatus WHERE StatusCode = 'DEPT_REJECTED')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseActionItemStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsActive, IsFinal)
    VALUES ('DEPT_REJECTED', 'Department Rejected', N'مرفوض من الإدارة', 3, 1, 0);
    PRINT '  ✅ Inserted: DEPT_REJECTED';
END
ELSE
BEGIN
    PRINT '  ⚠️  DEPT_REJECTED already exists. Skipping.';
END

-- Status 4: SUBMITTED_TO_ADMIN
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseActionItemStatus WHERE StatusCode = 'SUBMITTED_TO_ADMIN')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseActionItemStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsActive, IsFinal)
    VALUES ('SUBMITTED_TO_ADMIN', 'Submitted to Administration', N'مُرسل إلى الإدارة العليا', 4, 1, 0);
    PRINT '  ✅ Inserted: SUBMITTED_TO_ADMIN';
END
ELSE
BEGIN
    PRINT '  ⚠️  SUBMITTED_TO_ADMIN already exists. Skipping.';
END

-- Status 5: ADMIN_REJECTED
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseActionItemStatus WHERE StatusCode = 'ADMIN_REJECTED')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseActionItemStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsActive, IsFinal)
    VALUES ('ADMIN_REJECTED', 'Administration Rejected', N'مرفوض من الإدارة العليا', 5, 1, 0);
    PRINT '  ✅ Inserted: ADMIN_REJECTED';
END
ELSE
BEGIN
    PRINT '  ⚠️  ADMIN_REJECTED already exists. Skipping.';
END

-- Status 6: ADMIN_APPROVED
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseActionItemStatus WHERE StatusCode = 'ADMIN_APPROVED')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseActionItemStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsActive, IsFinal)
    VALUES ('ADMIN_APPROVED', 'Administration Approved', N'موافق من الإدارة العليا', 6, 1, 0);
    PRINT '  ✅ Inserted: ADMIN_APPROVED';
END
ELSE
BEGIN
    PRINT '  ⚠️  ADMIN_APPROVED already exists. Skipping.';
END

-- Status 7: IN_PROGRESS
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseActionItemStatus WHERE StatusCode = 'IN_PROGRESS')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseActionItemStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsActive, IsFinal)
    VALUES ('IN_PROGRESS', 'In Progress', N'قيد التنفيذ', 7, 1, 0);
    PRINT '  ✅ Inserted: IN_PROGRESS';
END
ELSE
BEGIN
    PRINT '  ⚠️  IN_PROGRESS already exists. Skipping.';
END

-- Status 8: DONE
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseActionItemStatus WHERE StatusCode = 'DONE')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseActionItemStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsActive, IsFinal)
    VALUES ('DONE', 'Done', N'منجز', 8, 1, 0);
    PRINT '  ✅ Inserted: DONE';
END
ELSE
BEGIN
    PRINT '  ⚠️  DONE already exists. Skipping.';
END

-- Status 9: VERIFIED (FINAL)
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseActionItemStatus WHERE StatusCode = 'VERIFIED')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseActionItemStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsActive, IsFinal)
    VALUES ('VERIFIED', 'Verified', N'موثق', 9, 1, 1);
    PRINT '  ✅ Inserted: VERIFIED';
END
ELSE
BEGIN
    PRINT '  ⚠️  VERIFIED already exists. Skipping.';
END

-- Status 10: CANCELLED (FINAL)
IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseActionItemStatus WHERE StatusCode = 'CANCELLED')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseActionItemStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsActive, IsFinal)
    VALUES ('CANCELLED', 'Cancelled', N'ملغي', 10, 1, 1);
    PRINT '  ✅ Inserted: CANCELLED';
END
ELSE
BEGIN
    PRINT '  ⚠️  CANCELLED already exists. Skipping.';
END
GO

PRINT '';
PRINT '============================================================';
PRINT 'PHASE 3 - STEP 3.1: LOOKUP TABLES COMPLETED';
PRINT '============================================================';
PRINT '';
PRINT 'Summary:';
PRINT '  - APP_Lookup_SubcaseType: Created and seeded (2 rows)';
PRINT '  - APP_Lookup_SubcaseStatus: Created and seeded (9 rows)';
PRINT '  - APP_Lookup_SubcaseActionItemStatus: Created and seeded (10 rows)';
PRINT '';
PRINT '✅ All lookup tables are ready for Phase 3 workflow system.';
PRINT '';
