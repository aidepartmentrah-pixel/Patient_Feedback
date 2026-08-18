-- Confirms role DEFINITIONS were seeded (APP_Roles), while explicitly documenting
-- that user ACCOUNTS are intentionally never seeded by this package.
SELECT COUNT(*) AS role_definition_count FROM dbo.APP_Roles;
-- Expect: > 0 (role definitions travel with the package as configuration seed data).

SELECT COUNT(*) AS user_account_count FROM dbo.APP_Users;
-- Expect: 0 on a freshly installed database. User accounts must be created
-- per-deployment (through the application's own admin/bootstrap flow), never
-- shipped as seed data -- see docs/DATABASE_STRUCTURE_REPORT.md.
