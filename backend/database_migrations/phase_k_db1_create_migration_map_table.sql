-- =============================================
-- PHASE K — KDB1 — CREATE DATA MIGRATION MAPPING TABLE
-- =============================================
-- Purpose: Track legacy → APP case migration
-- Enables: Idempotent migration and duplicate prevention
-- =============================================

-- Create the mapping table
CREATE TABLE dbo.APP_DataMigration_Map (
    MapID int IDENTITY(1,1) NOT NULL,
    legacy_case_id int NOT NULL,
    new_case_id int NOT NULL,
    migrated_by_user_id int NOT NULL,
    migrated_at datetime2 NOT NULL DEFAULT GETDATE(),
    
    CONSTRAINT PK_APP_DataMigration_Map PRIMARY KEY CLUSTERED (MapID),
    CONSTRAINT UQ_APP_DataMigration_Map_LegacyCase UNIQUE (legacy_case_id),
    
    CONSTRAINT FK_APP_DataMigration_Map_NewCase 
        FOREIGN KEY (new_case_id) 
        REFERENCES dbo.APP_IncidentCase(IncidentRequestCaseID),
    
    CONSTRAINT FK_APP_DataMigration_Map_User 
        FOREIGN KEY (migrated_by_user_id) 
        REFERENCES dbo.APP_Users(UserID)
);

-- Create explicit index on legacy_case_id for fast lookups
CREATE NONCLUSTERED INDEX IX_APP_DataMigration_Map_Legacy
ON dbo.APP_DataMigration_Map (legacy_case_id);

-- Verification output
PRINT 'APP_DataMigration_Map table created successfully';
PRINT 'Primary key: MapID';
PRINT 'Unique constraint: legacy_case_id';
PRINT 'Foreign keys: new_case_id → APP_IncidentCase, migrated_by_user_id → APP_Users';
PRINT 'Index: IX_APP_DataMigration_Map_Legacy on legacy_case_id';
