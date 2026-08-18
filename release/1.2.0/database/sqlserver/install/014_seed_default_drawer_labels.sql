-- Drawer Notes requires at least one active label to attach to a note (see
-- api_v2/services/drawer_note_service.py's "Notes must always have at least
-- ONE label" business rule). On a fresh install APP_DrawerLabel has zero
-- rows, so the label picker in the UI is empty and the feature is
-- unusable out of the box.
--
-- Drawer Notes/Labels are a feature unique to this system -- confirmed via
-- read-only probing that the old HCAT production system has no equivalent
-- endpoint (drawer-notes/drawer-labels both 404, even authenticated), so
-- there is no real historical data to migrate here. These are reasonable
-- generic defaults to seed a usable starting point; hospital staff can
-- rename/add more via the existing label management UI.

IF NOT EXISTS (SELECT 1 FROM dbo.APP_DrawerLabel WHERE LabelName = N'Follow-up Required')
INSERT INTO dbo.APP_DrawerLabel (LabelName, IsActive) VALUES (N'Follow-up Required', 1);
GO

IF NOT EXISTS (SELECT 1 FROM dbo.APP_DrawerLabel WHERE LabelName = N'Resolved')
INSERT INTO dbo.APP_DrawerLabel (LabelName, IsActive) VALUES (N'Resolved', 1);
GO

IF NOT EXISTS (SELECT 1 FROM dbo.APP_DrawerLabel WHERE LabelName = N'Escalated')
INSERT INTO dbo.APP_DrawerLabel (LabelName, IsActive) VALUES (N'Escalated', 1);
GO

IF NOT EXISTS (SELECT 1 FROM dbo.APP_DrawerLabel WHERE LabelName = N'Internal Note')
INSERT INTO dbo.APP_DrawerLabel (LabelName, IsActive) VALUES (N'Internal Note', 1);
GO
