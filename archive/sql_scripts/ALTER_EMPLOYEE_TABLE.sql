-- ============================================
-- ALTER APP_IncidentCaseEmployee TABLE
-- Add columns to link employees to incidents
-- ============================================

USE IncidentManager;
GO

-- Add incident linkage columns
ALTER TABLE dbo.APP_IncidentCaseEmployee
ADD IncidentRequestCaseID INT NULL;

ALTER TABLE dbo.APP_IncidentCaseEmployee
ADD IsPrimary BIT DEFAULT 0;

ALTER TABLE dbo.APP_IncidentCaseEmployee
ADD AssignedAt DATETIME DEFAULT GETDATE();

ALTER TABLE dbo.APP_IncidentCaseEmployee
ADD AssignedByUserID INT NULL;

-- Add foreign key constraint
ALTER TABLE dbo.APP_IncidentCaseEmployee
ADD CONSTRAINT FK_IncidentCaseEmployee_Incident
FOREIGN KEY (IncidentRequestCaseID) 
REFERENCES dbo.APP_IncidentCase(IncidentRequestCaseID);

-- Add index for performance
CREATE NONCLUSTERED INDEX IX_IncidentCaseEmployee_IncidentID
ON dbo.APP_IncidentCaseEmployee(IncidentRequestCaseID);

CREATE NONCLUSTERED INDEX IX_IncidentCaseEmployee_EmployeeID
ON dbo.APP_IncidentCaseEmployee(EmployeeID);

PRINT 'APP_IncidentCaseEmployee table altered successfully';
GO
