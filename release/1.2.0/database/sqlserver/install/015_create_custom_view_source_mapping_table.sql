-- Tracks which locally-provisioned APP_CUSTOM_VIEWS row corresponds to which
-- source-system view, the same way APP_UserSourceIDMap does for users
-- (012_create_user_source_mapping_table.sql) -- ViewID is IDENTITY, so local
-- IDs don't exist until provision.py inserts a row and captures
-- SCOPE_IDENTITY(). Identifies a source view by SourceSystem + SourceViewID
-- together, not SourceViewID alone, for the same future-second-source-system
-- reason as the user mapping table.

IF OBJECT_ID('dbo.APP_CustomViewSourceIDMap', 'U') IS NULL
CREATE TABLE [dbo].[APP_CustomViewSourceIDMap] (
    [SourceSystem]   nvarchar(100) NOT NULL,
    [SourceViewID]   int NOT NULL,
    [LocalViewID]    int NOT NULL,
    [MappedAt]       datetime2 NOT NULL DEFAULT (sysdatetime()),
    CONSTRAINT PK_APP_CustomViewSourceIDMap PRIMARY KEY (SourceSystem, SourceViewID),
    CONSTRAINT UQ_APP_CustomViewSourceIDMap_LocalViewID UNIQUE (LocalViewID),
    CONSTRAINT FK_APP_CustomViewSourceIDMap_APP_CUSTOM_VIEWS FOREIGN KEY (LocalViewID)
        REFERENCES dbo.APP_CUSTOM_VIEWS(ViewID)
);
GO
