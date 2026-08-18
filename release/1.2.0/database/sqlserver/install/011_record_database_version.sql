IF NOT EXISTS (SELECT 1 FROM dbo.SchemaMigrationHistory WHERE MigrationName = 'baseline_install_1.0.0')
    INSERT INTO dbo.SchemaMigrationHistory (MigrationName, Checksum, AppliedAt, AppliedBy, ApplicationVersion, Success)
    VALUES ('baseline_install_1.0.0', NULL, SYSUTCDATETIME(), SUSER_SNAME(), '1.0.0', 1);
