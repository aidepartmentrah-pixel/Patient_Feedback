-- ============================================================
-- Migration: Add Force Close Tracking Fields
-- Purpose: Add audit trail columns for force-close functionality
-- Date: 2026-02-10
-- ============================================================

-- Add force close tracking columns to APP_AdministrativeSubcase
ALTER TABLE dbo.APP_AdministrativeSubcase
ADD ForceClosedAt DATETIME NULL,
    ForceClosedByUserID INT NULL,
    ForceCloseReason NVARCHAR(MAX) NULL;

-- Add foreign key constraint for ForceClosedByUserID
ALTER TABLE dbo.APP_AdministrativeSubcase
ADD CONSTRAINT FK_AdministrativeSubcase_ForceClosedByUser
FOREIGN KEY (ForceClosedByUserID) REFERENCES dbo.APP_Users(UserID);

-- Add force close tracking columns to APP_IncidentCase (optional - for main incident tracking)
ALTER TABLE dbo.APP_IncidentCase
ADD ForceClosedAt DATETIME NULL,
    ForceClosedByUserID INT NULL,
    ForceCloseReason NVARCHAR(MAX) NULL;

-- Add foreign key constraint for ForceClosedByUserID
ALTER TABLE dbo.APP_IncidentCase
ADD CONSTRAINT FK_IncidentCase_ForceClosedByUser
FOREIGN KEY (ForceClosedByUserID) REFERENCES dbo.APP_Users(UserID);

-- Create index for efficient querying of force-closed cases
CREATE NONCLUSTERED INDEX IX_AdministrativeSubcase_ForceClosedAt
ON dbo.APP_AdministrativeSubcase(ForceClosedAt)
WHERE ForceClosedAt IS NOT NULL;

CREATE NONCLUSTERED INDEX IX_IncidentCase_ForceClosedAt
ON dbo.APP_IncidentCase(ForceClosedAt)
WHERE ForceClosedAt IS NOT NULL;

PRINT 'Migration completed: Force close tracking fields added successfully';
