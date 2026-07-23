-- Schema-only addition (no data) supporting Stage C of the organizational
-- unit / user account migration from the old HCAT system. Maps a source
-- system's user identity to the LocalUserID that gets assigned when
-- APP_Users.UserID (IDENTITY) is generated at insert time -- the mapping
-- cannot be known ahead of insertion, so this table is populated by
-- database/sqlserver/seed/provision.py, not by this install script.
--
-- Identified by (SourceSystem, SourceUserID) together, not SourceUserID
-- alone, so a future second source system cannot collide on ID numbering.

IF OBJECT_ID('dbo.APP_UserSourceIDMap', 'U') IS NULL
CREATE TABLE [dbo].[APP_UserSourceIDMap] (
    [SourceSystem]   nvarchar(100) NOT NULL,
    [SourceUserID]   int NOT NULL,
    [LocalUserID]    int NOT NULL,
    [MappedAt]       datetime2 NOT NULL DEFAULT (sysdatetime()),
    CONSTRAINT PK_APP_UserSourceIDMap PRIMARY KEY (SourceSystem, SourceUserID),
    CONSTRAINT UQ_APP_UserSourceIDMap_LocalUserID UNIQUE (LocalUserID),
    CONSTRAINT FK_APP_UserSourceIDMap_APP_Users FOREIGN KEY (LocalUserID)
        REFERENCES dbo.APP_Users(UserID)
);
GO
