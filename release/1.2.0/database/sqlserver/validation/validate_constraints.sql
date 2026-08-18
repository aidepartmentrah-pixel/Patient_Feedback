-- Confirms primary keys, foreign keys, and indexes were created as expected.
SELECT
    (SELECT COUNT(*) FROM sys.key_constraints WHERE type = 'PK') AS primary_key_count,
    (SELECT COUNT(*) FROM sys.foreign_keys) AS foreign_key_count,
    (SELECT COUNT(*) FROM sys.indexes WHERE is_primary_key = 0 AND name IS NOT NULL) AS non_pk_index_count;
-- Compare against docs/DATABASE_STRUCTURE_REPORT.md baseline: 81 PKs (2 fewer than the
-- 83-table inventory since the 2 obsolete tables are excluded from fresh installs),
-- foreign keys and indexes counted after excluding any FK/index touching the obsolete tables.
