-- ============================================================
-- AFC-S1: Automatic Force Close Policy — Database Foundation
-- Date: 2026-06-12
-- Safe to run multiple times (all checks are idempotent)
-- No data deleted. No existing columns altered.
-- Pure schema/seed work — no workflow behavior changes.
-- ============================================================

USE IncidentManager;
GO

-- Required for ALTER TABLE ADD ... DEFAULT on tables with filtered indexes
-- (APP_AdministrativeSubcase has filtered indexes from phase3_step2)
SET QUOTED_IDENTIFIER ON;
SET ANSI_NULLS ON;
GO

PRINT '============================================================';
PRINT 'AFC-S1: Automatic Force Close Policy — Database Foundation';
PRINT '============================================================';
PRINT '';

-- ============================================================
-- PART 1: Morbidity flag on APP_IncidentCase
-- ============================================================
PRINT 'PART 1: Morbidity flag on APP_IncidentCase...';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_IncidentCase') AND name = 'IsMorbidity')
BEGIN
    ALTER TABLE dbo.APP_IncidentCase
    ADD IsMorbidity BIT NOT NULL CONSTRAINT DF_IncidentCase_IsMorbidity DEFAULT 0;
    PRINT '  IsMorbidity added';
END
ELSE
    PRINT '  IsMorbidity already exists. Skipping.';

PRINT '';

-- ============================================================
-- PART 2: Workflow deadline fields on APP_AdministrativeSubcase
-- ============================================================
PRINT 'PART 2: Workflow deadline fields on APP_AdministrativeSubcase...';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'SectionDeadlineAt')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD SectionDeadlineAt DATETIME NULL;
    PRINT '  SectionDeadlineAt added';
END
ELSE
    PRINT '  SectionDeadlineAt already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'DepartmentDeadlineAt')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD DepartmentDeadlineAt DATETIME NULL;
    PRINT '  DepartmentDeadlineAt added';
END
ELSE
    PRINT '  DepartmentDeadlineAt already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'AdministrationDeadlineAt')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD AdministrationDeadlineAt DATETIME NULL;
    PRINT '  AdministrationDeadlineAt added';
END
ELSE
    PRINT '  AdministrationDeadlineAt already exists. Skipping.';

PRINT '';

-- ============================================================
-- PART 3: Force-close history fields on APP_AdministrativeSubcase
-- (per-level — distinct from the single ForceClosedAt added by FC-S1)
-- ============================================================
PRINT 'PART 3: Per-level force-close history fields on APP_AdministrativeSubcase...';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'SectionForceClosedAt')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD SectionForceClosedAt DATETIME NULL;
    PRINT '  SectionForceClosedAt added';
END
ELSE
    PRINT '  SectionForceClosedAt already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'DepartmentForceClosedAt')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD DepartmentForceClosedAt DATETIME NULL;
    PRINT '  DepartmentForceClosedAt added';
END
ELSE
    PRINT '  DepartmentForceClosedAt already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'AdministrationForceClosedAt')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD AdministrationForceClosedAt DATETIME NULL;
    PRINT '  AdministrationForceClosedAt added';
END
ELSE
    PRINT '  AdministrationForceClosedAt already exists. Skipping.';

PRINT '';

-- ============================================================
-- PART 4: Late-reply tracking fields on APP_AdministrativeSubcase
-- ============================================================
PRINT 'PART 4: Late-reply tracking fields on APP_AdministrativeSubcase...';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'SectionLateReply')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase
    ADD SectionLateReply BIT NOT NULL CONSTRAINT DF_AdministrativeSubcase_SectionLateReply DEFAULT 0;
    PRINT '  SectionLateReply added';
END
ELSE
    PRINT '  SectionLateReply already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'DepartmentLateReply')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase
    ADD DepartmentLateReply BIT NOT NULL CONSTRAINT DF_AdministrativeSubcase_DepartmentLateReply DEFAULT 0;
    PRINT '  DepartmentLateReply added';
END
ELSE
    PRINT '  DepartmentLateReply already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'AdministrationLateReply')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase
    ADD AdministrationLateReply BIT NOT NULL CONSTRAINT DF_AdministrativeSubcase_AdministrationLateReply DEFAULT 0;
    PRINT '  AdministrationLateReply added';
END
ELSE
    PRINT '  AdministrationLateReply already exists. Skipping.';

PRINT '';

-- ============================================================
-- PART 5: Give More Time / extra-time tracking fields on APP_AdministrativeSubcase
-- ============================================================
PRINT 'PART 5: Extra-time tracking fields on APP_AdministrativeSubcase...';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'SectionExtraTimeGrantedAt')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD SectionExtraTimeGrantedAt DATETIME NULL;
    PRINT '  SectionExtraTimeGrantedAt added';
END
ELSE
    PRINT '  SectionExtraTimeGrantedAt already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'DepartmentExtraTimeGrantedAt')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD DepartmentExtraTimeGrantedAt DATETIME NULL;
    PRINT '  DepartmentExtraTimeGrantedAt added';
END
ELSE
    PRINT '  DepartmentExtraTimeGrantedAt already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'AdministrationExtraTimeGrantedAt')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD AdministrationExtraTimeGrantedAt DATETIME NULL;
    PRINT '  AdministrationExtraTimeGrantedAt added';
END
ELSE
    PRINT '  AdministrationExtraTimeGrantedAt already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'SectionExtraTimeGrantedBy')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD SectionExtraTimeGrantedBy INT NULL;
    PRINT '  SectionExtraTimeGrantedBy added';
END
ELSE
    PRINT '  SectionExtraTimeGrantedBy already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'DepartmentExtraTimeGrantedBy')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD DepartmentExtraTimeGrantedBy INT NULL;
    PRINT '  DepartmentExtraTimeGrantedBy added';
END
ELSE
    PRINT '  DepartmentExtraTimeGrantedBy already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase') AND name = 'AdministrationExtraTimeGrantedBy')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase ADD AdministrationExtraTimeGrantedBy INT NULL;
    PRINT '  AdministrationExtraTimeGrantedBy added';
END
ELSE
    PRINT '  AdministrationExtraTimeGrantedBy already exists. Skipping.';

PRINT '';
PRINT 'PART 5b: Foreign keys for ExtraTimeGrantedBy columns -> APP_Users...';

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_SectionExtraTimeGrantedBy')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase
    ADD CONSTRAINT FK_AdministrativeSubcase_SectionExtraTimeGrantedBy
    FOREIGN KEY (SectionExtraTimeGrantedBy) REFERENCES dbo.APP_Users(UserID);
    PRINT '  FK for SectionExtraTimeGrantedBy added';
END
ELSE
    PRINT '  FK_AdministrativeSubcase_SectionExtraTimeGrantedBy already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_DepartmentExtraTimeGrantedBy')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase
    ADD CONSTRAINT FK_AdministrativeSubcase_DepartmentExtraTimeGrantedBy
    FOREIGN KEY (DepartmentExtraTimeGrantedBy) REFERENCES dbo.APP_Users(UserID);
    PRINT '  FK for DepartmentExtraTimeGrantedBy added';
END
ELSE
    PRINT '  FK_AdministrativeSubcase_DepartmentExtraTimeGrantedBy already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_AdministrationExtraTimeGrantedBy')
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase
    ADD CONSTRAINT FK_AdministrativeSubcase_AdministrationExtraTimeGrantedBy
    FOREIGN KEY (AdministrationExtraTimeGrantedBy) REFERENCES dbo.APP_Users(UserID);
    PRINT '  FK for AdministrationExtraTimeGrantedBy added';
END
ELSE
    PRINT '  FK_AdministrativeSubcase_AdministrationExtraTimeGrantedBy already exists. Skipping.';

PRINT '';

-- ============================================================
-- PART 6: New workflow statuses in APP_Lookup_SubcaseStatus
-- ============================================================
PRINT 'PART 6: Adding new force-close subcase statuses...';

IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'FORCE_CLOSED_AT_SECTION')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('FORCE_CLOSED_AT_SECTION', 'Force Closed at Section', N'تم الإغلاق القسري على مستوى القسم', 14, 1, 1);
    PRINT '  FORCE_CLOSED_AT_SECTION added';
END
ELSE
    PRINT '  FORCE_CLOSED_AT_SECTION already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'FORCE_CLOSED_AT_DEPARTMENT')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('FORCE_CLOSED_AT_DEPARTMENT', 'Force Closed at Department', N'تم الإغلاق القسري على مستوى الإدارة المعنية', 15, 1, 1);
    PRINT '  FORCE_CLOSED_AT_DEPARTMENT added';
END
ELSE
    PRINT '  FORCE_CLOSED_AT_DEPARTMENT already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus WHERE StatusCode = 'FORCE_CLOSED_AT_ADMINISTRATION')
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES ('FORCE_CLOSED_AT_ADMINISTRATION', 'Force Closed at Administration', N'تم الإغلاق القسري على مستوى الإدارة العليا', 16, 1, 1);
    PRINT '  FORCE_CLOSED_AT_ADMINISTRATION added';
END
ELSE
    PRINT '  FORCE_CLOSED_AT_ADMINISTRATION already exists. Skipping.';

PRINT '';

-- ============================================================
-- PART 7: New settings in APP_SystemSettings
-- ============================================================
PRINT 'PART 7: Adding Automatic Force Close Policy settings...';

IF NOT EXISTS (SELECT 1 FROM dbo.APP_SystemSettings WHERE SettingKey = 'automatic_force_close_enabled')
BEGIN
    INSERT INTO dbo.APP_SystemSettings (SettingKey, SettingValue, SettingLabel, SettingLabelAr, SettingType, Description, DescriptionAr)
    VALUES (
        'automatic_force_close_enabled', 'false',
        'Automatic Force Close Enabled', N'تفعيل الإغلاق القسري التلقائي',
        'boolean',
        'Enable automatic force-close processing for overdue workflow stages',
        N'تفعيل معالجة الإغلاق القسري التلقائي لمراحل سير العمل المتأخرة'
    );
    PRINT '  automatic_force_close_enabled added';
END
ELSE
    PRINT '  automatic_force_close_enabled already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM dbo.APP_SystemSettings WHERE SettingKey = 'section_deadline_days')
BEGIN
    INSERT INTO dbo.APP_SystemSettings (SettingKey, SettingValue, SettingLabel, SettingLabelAr, SettingType, Description, DescriptionAr)
    VALUES (
        'section_deadline_days', '10',
        'Section Deadline (Days)', N'مهلة القسم (بالأيام)',
        'number',
        'Number of days a section has to respond before its workflow stage becomes overdue',
        N'عدد الأيام المتاحة للقسم للرد قبل اعتبار مرحلة سير العمل متأخرة'
    );
    PRINT '  section_deadline_days added';
END
ELSE
    PRINT '  section_deadline_days already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM dbo.APP_SystemSettings WHERE SettingKey = 'department_deadline_days')
BEGIN
    INSERT INTO dbo.APP_SystemSettings (SettingKey, SettingValue, SettingLabel, SettingLabelAr, SettingType, Description, DescriptionAr)
    VALUES (
        'department_deadline_days', '7',
        'Department Deadline (Days)', N'مهلة الإدارة المعنية (بالأيام)',
        'number',
        'Number of days a department has to respond before its workflow stage becomes overdue',
        N'عدد الأيام المتاحة للإدارة المعنية للرد قبل اعتبار مرحلة سير العمل متأخرة'
    );
    PRINT '  department_deadline_days added';
END
ELSE
    PRINT '  department_deadline_days already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM dbo.APP_SystemSettings WHERE SettingKey = 'administration_deadline_days')
BEGIN
    INSERT INTO dbo.APP_SystemSettings (SettingKey, SettingValue, SettingLabel, SettingLabelAr, SettingType, Description, DescriptionAr)
    VALUES (
        'administration_deadline_days', '7',
        'Administration Deadline (Days)', N'مهلة الإدارة العليا (بالأيام)',
        'number',
        'Number of days the higher administration has to respond before its workflow stage becomes overdue',
        N'عدد الأيام المتاحة للإدارة العليا للرد قبل اعتبار مرحلة سير العمل متأخرة'
    );
    PRINT '  administration_deadline_days added';
END
ELSE
    PRINT '  administration_deadline_days already exists. Skipping.';

PRINT '';

-- ============================================================
-- VERIFICATION
-- ============================================================
PRINT '============================================================';
PRINT 'VERIFICATION';
PRINT '============================================================';
PRINT '';

PRINT '--- APP_IncidentCase.IsMorbidity ---';
SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_IncidentCase') AND name = 'IsMorbidity';

PRINT '';
PRINT '--- APP_AdministrativeSubcase new columns ---';
SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase')
  AND name IN (
    'SectionDeadlineAt', 'DepartmentDeadlineAt', 'AdministrationDeadlineAt',
    'SectionForceClosedAt', 'DepartmentForceClosedAt', 'AdministrationForceClosedAt',
    'SectionLateReply', 'DepartmentLateReply', 'AdministrationLateReply',
    'SectionExtraTimeGrantedAt', 'DepartmentExtraTimeGrantedAt', 'AdministrationExtraTimeGrantedAt',
    'SectionExtraTimeGrantedBy', 'DepartmentExtraTimeGrantedBy', 'AdministrationExtraTimeGrantedBy'
  )
ORDER BY name;

PRINT '';
PRINT '--- APP_Lookup_SubcaseStatus (all statuses) ---';
SELECT StatusCode, StatusNameEn, DisplayOrder, IsFinal, IsActive
FROM dbo.APP_Lookup_SubcaseStatus
ORDER BY DisplayOrder;

PRINT '';
PRINT '--- APP_SystemSettings (new settings) ---';
SELECT SettingKey, SettingValue, SettingType, IsActive
FROM dbo.APP_SystemSettings
WHERE SettingKey IN (
    'automatic_force_close_enabled', 'section_deadline_days',
    'department_deadline_days', 'administration_deadline_days'
)
ORDER BY SettingKey;

PRINT '';
PRINT '============================================================';
PRINT 'AFC-S1: Migration complete.';
PRINT 'Expected: 1 new column on APP_IncidentCase,';
PRINT '          15 new columns + 3 new FKs on APP_AdministrativeSubcase,';
PRINT '          3 new statuses in APP_Lookup_SubcaseStatus,';
PRINT '          4 new settings in APP_SystemSettings.';
PRINT '============================================================';
