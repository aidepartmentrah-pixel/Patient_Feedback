-- Migration Script: Upgrade APP_ActionItem for Follow-Up Page
-- Adds missing columns required by the Follow-Up functionality
-- Date: 2026-01-21

USE IncidentManager;
GO

-- Add new columns for follow-up action management
ALTER TABLE dbo.APP_ActionItem
ADD 
    DepartmentID INT NULL,
    AssignedTo NVARCHAR(255) NULL,
    Priority NVARCHAR(20) NOT NULL DEFAULT 'medium',
    Status NVARCHAR(20) NOT NULL DEFAULT 'pending',
    CompletedDate DATE NULL,
    Notes NVARCHAR(MAX) NULL,
    LastUpdatedAt DATETIME NULL,
    LastUpdatedByUserID INT NULL;
GO

-- Add CHECK constraints for valid values
ALTER TABLE dbo.APP_ActionItem
ADD CONSTRAINT CHK_ActionItem_Priority 
    CHECK (Priority IN ('low', 'medium', 'high'));
GO

ALTER TABLE dbo.APP_ActionItem
ADD CONSTRAINT CHK_ActionItem_Status 
    CHECK (Status IN ('pending', 'delayed', 'completed'));
GO

-- Set default values for LastUpdatedAt to match CreatedAt for existing rows
UPDATE dbo.APP_ActionItem
SET LastUpdatedAt = CreatedAt,
    LastUpdatedByUserID = CreatedByUserID
WHERE LastUpdatedAt IS NULL;
GO

-- Migrate IsDone to Status for existing records
UPDATE dbo.APP_ActionItem
SET Status = CASE 
    WHEN IsDone = 1 THEN 'completed'
    ELSE 'pending'
END
WHERE Status = 'pending';  -- Only update if still at default
GO

-- Set CompletedDate from DateSubmitted for completed items
UPDATE dbo.APP_ActionItem
SET CompletedDate = DateSubmitted
WHERE IsDone = 1 AND DateSubmitted IS NOT NULL AND CompletedDate IS NULL;
GO

-- Add foreign key for DepartmentID (if not already exists)
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ActionItem_Department')
BEGIN
    ALTER TABLE dbo.APP_ActionItem
    ADD CONSTRAINT FK_ActionItem_Department
        FOREIGN KEY (DepartmentID) REFERENCES dbo.Department(DepartmentID)
        ON DELETE SET NULL;
END
GO

PRINT 'APP_ActionItem table migration completed successfully!';
PRINT 'New columns added: DepartmentID, AssignedTo, Priority, Status, CompletedDate, Notes, LastUpdatedAt, LastUpdatedByUserID';
