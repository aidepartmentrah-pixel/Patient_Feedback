-- =============================================================
-- Phase RCA Step 1: New RCA Tables + Category Seed
-- Date: 2026-06-10
-- =============================================================

-- =============================================================
-- TABLE 1: APP_RCAFactorCategory
-- Stores the 7 RCA factor groups
-- =============================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables WHERE name = 'APP_RCAFactorCategory'
)
BEGIN
    CREATE TABLE dbo.APP_RCAFactorCategory (
        CategoryID          INT IDENTITY(1,1) PRIMARY KEY,
        CategoryCode        NVARCHAR(100)  NOT NULL,
        CategoryNameEn      NVARCHAR(255)  NOT NULL,
        CategoryNameAr      NVARCHAR(255)  NULL,
        SortOrder           INT            NOT NULL DEFAULT 0,
        IsActive            BIT            NOT NULL DEFAULT 1,
        CreatedAt           DATETIME2      NOT NULL DEFAULT SYSUTCDATETIME(),
        CreatedByUserID     INT            NULL,
        UpdatedAt           DATETIME2      NULL,
        UpdatedByUserID     INT            NULL,

        CONSTRAINT UQ_APP_RCAFactorCategory_Code UNIQUE (CategoryCode)
    );
END;

-- =============================================================
-- TABLE 2: APP_RCASuggestion
-- Stores cause and corrective-action suggestions
-- =============================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables WHERE name = 'APP_RCASuggestion'
)
BEGIN
    CREATE TABLE dbo.APP_RCASuggestion (
        SuggestionID        INT IDENTITY(1,1) PRIMARY KEY,
        CategoryID          INT             NOT NULL,
        SuggestionType      NVARCHAR(50)    NOT NULL,
        SuggestionTextEn    NVARCHAR(500)   NULL,
        SuggestionTextAr    NVARCHAR(500)   NOT NULL,
        DescriptionEn       NVARCHAR(MAX)   NULL,
        DescriptionAr       NVARCHAR(MAX)   NULL,
        SortOrder           INT             NOT NULL DEFAULT 0,
        IsActive            BIT             NOT NULL DEFAULT 1,
        CreatedAt           DATETIME2       NOT NULL DEFAULT SYSUTCDATETIME(),
        CreatedByUserID     INT             NULL,
        UpdatedAt           DATETIME2       NULL,
        UpdatedByUserID     INT             NULL,

        CONSTRAINT FK_APP_RCASuggestion_Category
            FOREIGN KEY (CategoryID)
            REFERENCES dbo.APP_RCAFactorCategory(CategoryID),

        CONSTRAINT CK_APP_RCASuggestion_Type
            CHECK (SuggestionType IN ('CAUSE', 'ACTION_ITEM'))
    );
END;

-- =============================================================
-- TABLE 3: APP_SubcaseRCASuggestionSelection
-- Stores which suggestions a section user selected per subcase
-- =============================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.tables WHERE name = 'APP_SubcaseRCASuggestionSelection'
)
BEGIN
    CREATE TABLE dbo.APP_SubcaseRCASuggestionSelection (
        SelectionID         INT IDENTITY(1,1) PRIMARY KEY,
        SubcaseID           INT             NOT NULL,
        SuggestionID        INT             NOT NULL,
        SelectedByUserID    INT             NULL,
        SelectedAt          DATETIME2       NOT NULL DEFAULT SYSUTCDATETIME(),

        CONSTRAINT FK_APP_SubcaseRCASuggestionSelection_Subcase
            FOREIGN KEY (SubcaseID)
            REFERENCES dbo.APP_AdministrativeSubcase(SubcaseID),

        CONSTRAINT FK_APP_SubcaseRCASuggestionSelection_Suggestion
            FOREIGN KEY (SuggestionID)
            REFERENCES dbo.APP_RCASuggestion(SuggestionID),

        CONSTRAINT UQ_APP_SubcaseRCASuggestionSelection
            UNIQUE (SubcaseID, SuggestionID)
    );
END;

-- =============================================================
-- SEED: 7 RCA Factor Categories
-- Only insert if the table is empty (idempotent)
-- =============================================================
IF NOT EXISTS (SELECT 1 FROM dbo.APP_RCAFactorCategory)
BEGIN
    INSERT INTO dbo.APP_RCAFactorCategory
        (CategoryCode, CategoryNameEn, CategoryNameAr, SortOrder, IsActive)
    VALUES
        ('TEAM_FACTORS',          'Team Factors',                        N'عوامل الفريق',                         1, 1),
        ('WORK_ENV_FACTORS',      'Work Environment Factors',            N'عوامل بيئة العمل',                     2, 1),
        ('TASK_POLICY_FACTORS',   'Task & Policy Factors',               N'عوامل المهام والسياسات',               3, 1),
        ('TECH_EQUIP_FACTORS',    'Technology & Equipment Factors',      N'عوامل التكنولوجيا والمعدات',           4, 1),
        ('TRAINING_EDU_FACTORS',  'Training & Education Factors',        N'عوامل التدريب والتعليم',               5, 1),
        ('ORG_STRATEGIC_FACTORS', 'Organizational & Strategic Factors',  N'العوامل التنظيمية والاستراتيجية',     6, 1),
        ('PATIENT_FACTORS',       'Patient Factors',                     N'عوامل المريض',                         7, 1);
END;
