-- Create APP_LOOKUP_SOURCE table for feedback sources
-- This table stores different sources of patient feedback

USE IncidentManager;
GO

-- Create the lookup table
CREATE TABLE dbo.APP_LOOKUP_SOURCE (
    SourceID INT PRIMARY KEY IDENTITY(1,1),
    SourceName NVARCHAR(100) NOT NULL,
    SourceNameAr NVARCHAR(100) NOT NULL,
    DisplayOrder INT NOT NULL,
    IsActive BIT DEFAULT 1,
    CreatedAt DATETIME DEFAULT GETDATE(),
    UpdatedAt DATETIME DEFAULT GETDATE()
);
GO

-- Insert the source options
INSERT INTO dbo.APP_LOOKUP_SOURCE (SourceName, SourceNameAr, DisplayOrder, IsActive)
VALUES 
    ('Tours', N'جولات', 1, 1),
    ('Attendance', N'حضور', 2, 1),
    ('Hotline', N'خط ساخن', 3, 1),
    ('Box', N'صندوق', 4, 1),
    ('Supervisor', N'مشرف', 5, 1),
    ('Employee', N'موظف', 6, 1),
    ('Office WhatsApp', N'واتساب مكتب', 7, 1),
    ('Social Media', N'وسائل التواصل', 8, 1);
GO

-- Add SourceID column to APP_IncidentCase table if it doesn't exist
IF NOT EXISTS (
    SELECT 1 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID('dbo.APP_IncidentCase') 
    AND name = 'SourceID'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase
    ADD SourceID INT NULL;
    
    -- Add foreign key constraint
    ALTER TABLE dbo.APP_IncidentCase
    ADD CONSTRAINT FK_APP_IncidentCase_Source
    FOREIGN KEY (SourceID) REFERENCES dbo.APP_LOOKUP_SOURCE(SourceID);
END
GO

-- Verify the data
SELECT * FROM dbo.APP_LOOKUP_SOURCE ORDER BY DisplayOrder;
GO

PRINT 'APP_LOOKUP_SOURCE table created successfully with 8 source options';
