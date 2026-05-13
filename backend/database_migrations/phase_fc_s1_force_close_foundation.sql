-- ============================================================
-- FC-S1: Force Close Overhaul — Database Foundation
-- Date: 2026-05-12
-- Safe to run multiple times (all checks are idempotent)
-- No data deleted. All new columns are nullable.
-- ============================================================

USE IncidentManager;
GO

PRINT '============================================================';
PRINT 'FC-S1: Force Close Overhaul — Database Foundation';
PRINT '============================================================';
PRINT '';

-- ============================================================
-- PART 1: New Statuses in APP_Lookup_SubcaseStatus
-- ============================================================
PRINT 'PART 1: Adding new subcase statuses...';

IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'FORCE_CLOSED_DRAFT')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('FORCE_CLOSED_DRAFT', 'Force Closed - Draft', N'إغلاق إجباري - مسودة', 10, 0, 1);
    PRINT '  ✅ FORCE_CLOSED_DRAFT added';
END
ELSE
    PRINT '  ⚠️  FORCE_CLOSED_DRAFT already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'FORCE_CLOSED_COMPLETE')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('FORCE_CLOSED_COMPLETE', 'Force Closed - Complete', N'إغلاق إجباري - مكتمل', 11, 1, 1);
    PRINT '  ✅ FORCE_CLOSED_COMPLETE added';
END
ELSE
    PRINT '  ⚠️  FORCE_CLOSED_COMPLETE already exists. Skipping.';

PRINT '';

-- ============================================================
-- PART 2: Force Close header fields on APP_AdministrativeSubcase
-- (idempotent — archive migration may have already added these)
-- ============================================================
PRINT 'PART 2: Force close header fields on APP_AdministrativeSubcase...';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'ForceClosedAt')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD ForceClosedAt DATETIME NULL;
    PRINT '  ✅ ForceClosedAt added';
END
ELSE
    PRINT '  ⚠️  ForceClosedAt already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'ForceClosedByUserID')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD ForceClosedByUserID INT NULL;
    PRINT '  ✅ ForceClosedByUserID added';
END
ELSE
    PRINT '  ⚠️  ForceClosedByUserID already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'ForceCloseReason')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD ForceCloseReason NVARCHAR(MAX) NULL;
    PRINT '  ✅ ForceCloseReason added';
END
ELSE
    PRINT '  ⚠️  ForceCloseReason already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_ForceClosedByUser')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase
    ADD CONSTRAINT FK_AdministrativeSubcase_ForceClosedByUser
    FOREIGN KEY (ForceClosedByUserID) REFERENCES dbo.APP_Users(UserID);
    PRINT '  ✅ FK for ForceClosedByUserID added';
END
ELSE
    PRINT '  ⚠️  FK_AdministrativeSubcase_ForceClosedByUser already exists. Skipping.';

PRINT '';

-- ============================================================
-- PART 3: Per-level ownership fields on APP_AdministrativeSubcase
-- Tracks who entered each level's data, on behalf of whom, and how
-- EntryMode values: NORMAL | ON_BEHALF | FORCE_CLOSE_INTERVENTION
-- ============================================================
PRINT 'PART 3: Per-level ownership fields on APP_AdministrativeSubcase...';

-- Section level
IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'SectionEnteredByUserID')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD SectionEnteredByUserID INT NULL;
    PRINT '  ✅ SectionEnteredByUserID added';
END
ELSE
    PRINT '  ⚠️  SectionEnteredByUserID already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'SectionEnteredForRole')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD SectionEnteredForRole NVARCHAR(50) NULL;
    PRINT '  ✅ SectionEnteredForRole added';
END
ELSE
    PRINT '  ⚠️  SectionEnteredForRole already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'SectionEntryMode')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD SectionEntryMode NVARCHAR(50) NULL;
    PRINT '  ✅ SectionEntryMode added';
END
ELSE
    PRINT '  ⚠️  SectionEntryMode already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'SectionEntryTimestamp')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD SectionEntryTimestamp DATETIME NULL;
    PRINT '  ✅ SectionEntryTimestamp added';
END
ELSE
    PRINT '  ⚠️  SectionEntryTimestamp already exists. Skipping.';

-- Department level
IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'DepartmentEnteredByUserID')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD DepartmentEnteredByUserID INT NULL;
    PRINT '  ✅ DepartmentEnteredByUserID added';
END
ELSE
    PRINT '  ⚠️  DepartmentEnteredByUserID already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'DepartmentEnteredForRole')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD DepartmentEnteredForRole NVARCHAR(50) NULL;
    PRINT '  ✅ DepartmentEnteredForRole added';
END
ELSE
    PRINT '  ⚠️  DepartmentEnteredForRole already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'DepartmentEntryMode')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD DepartmentEntryMode NVARCHAR(50) NULL;
    PRINT '  ✅ DepartmentEntryMode added';
END
ELSE
    PRINT '  ⚠️  DepartmentEntryMode already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'DepartmentEntryTimestamp')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD DepartmentEntryTimestamp DATETIME NULL;
    PRINT '  ✅ DepartmentEntryTimestamp added';
END
ELSE
    PRINT '  ⚠️  DepartmentEntryTimestamp already exists. Skipping.';

-- Administration level
IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'AdministrationEnteredByUserID')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD AdministrationEnteredByUserID INT NULL;
    PRINT '  ✅ AdministrationEnteredByUserID added';
END
ELSE
    PRINT '  ⚠️  AdministrationEnteredByUserID already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'AdministrationEnteredForRole')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD AdministrationEnteredForRole NVARCHAR(50) NULL;
    PRINT '  ✅ AdministrationEnteredForRole added';
END
ELSE
    PRINT '  ⚠️  AdministrationEnteredForRole already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'AdministrationEntryMode')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD AdministrationEntryMode NVARCHAR(50) NULL;
    PRINT '  ✅ AdministrationEntryMode added';
END
ELSE
    PRINT '  ⚠️  AdministrationEntryMode already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'AdministrationEntryTimestamp')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD AdministrationEntryTimestamp DATETIME NULL;
    PRINT '  ✅ AdministrationEntryTimestamp added';
END
ELSE
    PRINT '  ⚠️  AdministrationEntryTimestamp already exists. Skipping.';

PRINT '';

-- ============================================================
-- PART 4: Ownership fields on APP_SubcaseActionItem
-- ============================================================
PRINT 'PART 4: Ownership fields on APP_SubcaseActionItem...';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_SubcaseActionItem') AND name = 'EnteredByUserID')
BEGIN
    ALTER TABLE dbo.APP_SubcaseActionItem ADD EnteredByUserID INT NULL;
    PRINT '  ✅ EnteredByUserID added';
END
ELSE
    PRINT '  ⚠️  EnteredByUserID already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_SubcaseActionItem') AND name = 'EnteredForRole')
BEGIN
    ALTER TABLE dbo.APP_SubcaseActionItem ADD EnteredForRole NVARCHAR(50) NULL;
    PRINT '  ✅ EnteredForRole added';
END
ELSE
    PRINT '  ⚠️  EnteredForRole already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_SubcaseActionItem') AND name = 'EntryMode')
BEGIN
    ALTER TABLE dbo.APP_SubcaseActionItem ADD EntryMode NVARCHAR(50) NULL;
    PRINT '  ✅ EntryMode added';
END
ELSE
    PRINT '  ⚠️  EntryMode already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_SubcaseActionItem') AND name = 'EntryTimestamp')
BEGIN
    ALTER TABLE dbo.APP_SubcaseActionItem ADD EntryTimestamp DATETIME NULL;
    PRINT '  ✅ EntryTimestamp added';
END
ELSE
    PRINT '  ⚠️  EntryTimestamp already exists. Skipping.';

PRINT '';

-- ============================================================
-- VERIFICATION
-- ============================================================
PRINT '============================================================';
PRINT 'VERIFICATION';
PRINT '============================================================';
PRINT '';

PRINT '--- APP_Lookup_SubcaseStatus (all statuses) ---';
SELECT StatusCode, StatusNameEn, DisplayOrder, IsFinal, IsActive
FROM dbo.APP_Lookup_SubcaseStatus
ORDER BY DisplayOrder;

PRINT '';
PRINT '--- APP_AdministrativeSubcase columns (ownership + force close) ---';
SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase')
  AND name IN (
    'ForceClosedAt', 'ForceClosedByUserID', 'ForceCloseReason',
    'SectionEnteredByUserID', 'SectionEnteredForRole', 'SectionEntryMode', 'SectionEntryTimestamp',
    'DepartmentEnteredByUserID', 'DepartmentEnteredForRole', 'DepartmentEntryMode', 'DepartmentEntryTimestamp',
    'AdministrationEnteredByUserID', 'AdministrationEnteredForRole', 'AdministrationEntryMode', 'AdministrationEntryTimestamp'
  )
ORDER BY name;

PRINT '';
PRINT '--- APP_SubcaseActionItem columns (ownership) ---';
SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_SubcaseActionItem')
  AND name IN ('EnteredByUserID', 'EnteredForRole', 'EntryMode', 'EntryTimestamp')
ORDER BY name;

PRINT '';
PRINT '============================================================';
PRINT 'FC-S1: Migration complete.';
PRINT 'Expected: 2 new statuses, 15 new columns on APP_AdministrativeSubcase,';
PRINT '          4 new columns on APP_SubcaseActionItem.';
PRINT '============================================================';
