# Database Structure Report

**Database:** `IncidentManager` (SQL Server, local SQLEXPRESS instance)  
**Inspected:** 2026-07-21, live production, read-only queries against `INFORMATION_SCHEMA`/`sys.*` catalog views only.  
**Tables:** 83 total (81 travel with a fresh install; 2 are obsolete, see below).

## Fresh-install vs existing-install boundary

This package draws a hard line, per explicit decision on 2026-07-21:

- **Fresh-install schema + seed** (`install/`): creates table structure for all non-obsolete
  tables, and seeds ONLY universal lookup data and installation-configuration data.
  No patient, complaint, ML-training, or user-account data is ever included.
- **Existing-install data preservation**: an already-running installation's real data
  (incidents, ML training history, patient reserve records, user accounts, etc.) moves
  between servers only via full database backup/restore (`scripts/backup_database.sql`,
  `scripts/restore_database.sql`), never via the seed scripts.
- **Obsolete object retirement**: schema changes that remove things are handled through
  `retirement/`, reviewed and explicitly approved before execution — never bundled into
  the normal install/migration path.

## Classification manifest

Every table, one row each. `Seeded?` reflects what `install/` actually inserts on a fresh install.

| Schema | Table | Columns | Current Rows | Category | Seeded on fresh install? |
|---|---|---|---|---|---|
| dbo | APP_LOOKUP_BUILDING | 3 | 2 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_CASE_STAGE | 3 | 6 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_CASE_STATUS | 7 | 5 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_CATEGORY | 4 | 7 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_CLASSIFICATION | 5 | 78 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_CLINICAL_RISK_TYPE | 6 | 3 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_DOMAIN | 4 | 3 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_EXPLANATION_STATUS | 2 | 4 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_FEEDBACK_INTENT_TYPE | 7 | 2 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_HARM_LEVEL | 3 | 5 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_RECORD_TYPE | 2 | 2 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_SEVERITY | 9 | 3 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_SOURCE | 7 | 8 | Universal lookup seed | Yes |
| dbo | APP_LOOKUP_SUBCATEGORY | 3 | 27 | Universal lookup seed | Yes |
| dbo | APP_Lookup_SatisfactionStatus | 5 | 3 | Universal lookup seed | Yes |
| dbo | APP_Lookup_SubcaseActionItemStatus | 6 | 10 | Universal lookup seed | Yes |
| dbo | APP_Lookup_SubcaseStatus | 6 | 16 | Universal lookup seed | Yes |
| dbo | APP_Lookup_SubcaseType | 4 | 2 | Universal lookup seed | Yes |
| ml | EmbeddingModelVersion | 13 | 1 | Universal lookup seed | Yes |
| dbo | APP_DepartmentEvaluationRule | 14 | 0 | Installation configuration seed | Yes |
| dbo | APP_DepartmentPolicy | 4 | 3 | Installation configuration seed | Yes |
| dbo | APP_OrgUnitPolicy | 18 | 179 | Installation configuration seed | Yes |
| dbo | APP_Roles | 4 | 6 | Installation configuration seed | Yes |
| dbo | VW_Doctors | 7 | 5 | Obsolete object - remove via reviewed migration | No |
| dbo | VW_PatientAdmission | 60 | 100 | Obsolete object - remove via reviewed migration | No |
| ml | CaseTrainingRecord | 23 | 26 | Patient-derived/ML historical data - never seed | No |
| ml | HistoricalTrainingExample | 32 | 961 | Patient-derived/ML historical data - never seed | No |
| dbo | SchemaMigrationHistory | 7 | 2 | Derived or transient data - start empty | No |
| ml | EmbeddingProcessingJob | 15 | 26 | Derived or transient data - start empty | No |
| ml | ImportBatch | 15 | 0 | Derived or transient data - start empty | No |
| ml | ImportSourceRecordMap | 6 | 0 | Derived or transient data - start empty | No |
| ml | LegacyDbSizeHistory | 5 | 1 | Derived or transient data - start empty | No |
| ml | LegacyModelMetricHistory | 11 | 109 | Derived or transient data - start empty | No |
| ml | LegacyTrainingRunHistory | 8 | 22 | Derived or transient data - start empty | No |
| dbo | APP_DrawerLabel ⚠️ | 4 | 15 | Schema only - possibly vestigial, flagged for review | No |
| dbo | APP_RCAFactorCategory ⚠️ | 10 | 7 | Schema only - possibly vestigial, flagged for review | No |
| dbo | APP_RCASuggestion ⚠️ | 14 | 42 | Schema only - possibly vestigial, flagged for review | No |
| dbo | Instance ⚠️ | 13 | 0 | Schema only - possibly vestigial, flagged for review | No |
| dbo | Parameter ⚠️ | 7 | 0 | Schema only - possibly vestigial, flagged for review | No |
| dbo | Role ⚠️ | 3 | 0 | Schema only - possibly vestigial, flagged for review | No |
| dbo | APP_ActionItem | 11 | 0 | Operational data - never seed | No |
| dbo | APP_AdministrativeSubcase | 50 | 501 | Operational data - never seed | No |
| dbo | APP_CUSTOM_VIEWS | 54 | 32 | Operational data - never seed | No |
| dbo | APP_DataMigration_Map | 5 | 0 | Operational data - never seed | No |
| dbo | APP_DrawerNote | 9 | 17 | Operational data - never seed | No |
| dbo | APP_DrawerNoteLabelLink | 2 | 24 | Operational data - never seed | No |
| dbo | APP_ExternalApiSettings | 11 | 1 | Operational data - never seed | No |
| dbo | APP_HardwareConfig | 13 | 23 | Operational data - never seed | No |
| dbo | APP_Incident | 14 | 131 | Operational data - never seed | No |
| dbo | APP_IncidentCase | 32 | 174 | Operational data - never seed | No |
| dbo | APP_IncidentCaseDoctor | 7 | 18 | Operational data - never seed | No |
| dbo | APP_IncidentCaseEmployee | 14 | 18 | Operational data - never seed | No |
| dbo | APP_IncidentCaseFeedback | 36 | 16 | Operational data - never seed | No |
| dbo | APP_IncidentCaseSatisfaction | 9 | 2 | Operational data - never seed | No |
| dbo | APP_IncidentCaseTargetDepartment | 6 | 171 | Operational data - never seed | No |
| dbo | APP_LOOKUP_DOCTOR | 6 | 23 | Operational data - never seed | No |
| dbo | APP_PublicationBatch | 7 | 3 | Operational data - never seed | No |
| dbo | APP_PublicationBatchCase | 6 | 13 | Operational data - never seed | No |
| dbo | APP_RESERVE_DOCTOR | 6 | 30 | Operational data - never seed | No |
| dbo | APP_RESERVE_PATIENT | 61 | 27 | Operational data - never seed | No |
| dbo | APP_ReportConfig | 4 | 9 | Operational data - never seed | No |
| dbo | APP_SeasonCase | 8 | 0 | Operational data - never seed | No |
| dbo | APP_SeasonalOrgUnitReport | 19 | 509 | Operational data - never seed | No |
| dbo | APP_SeasonalOrgUnitReport_ClassificationStats | 12 | 222 | Operational data - never seed | No |
| dbo | APP_SeasonalOrgUnitReport_PolicySnapshot | 12 | 12 | Operational data - never seed | No |
| dbo | APP_SubcaseActionItem | 18 | 7 | Operational data - never seed | No |
| dbo | APP_SubcaseActionItemChangeNotice | 13 | 0 | Operational data - never seed | No |
| dbo | APP_SubcaseDecisionAcknowledgment | 5 | 5 | Operational data - never seed | No |
| dbo | APP_SubcaseRCASuggestionSelection | 5 | 4 | Operational data - never seed | No |
| dbo | APP_SupervisorActionItem | 17 | 5 | Operational data - never seed | No |
| dbo | APP_SupervisorActionItemAuditLog | 6 | 5 | Operational data - never seed | No |
| dbo | APP_SystemSettings | 12 | 8 | Operational data - never seed | No |
| dbo | APP_UserRoleScope | 5 | 164 | Operational data - never seed | No |
| dbo | APP_Users | 8 | 165 | Operational data - never seed | No |
| dbo | AdminsrationUnit | 9 | 179 | Operational data - never seed | No |
| dbo | AdminsrationUnitHistory | 9 | 0 | Operational data - never seed | No |
| dbo | IncidentRequest | 31 | 25 | Operational data - never seed | No |
| dbo | IncidentRequestCase | 23 | 7 | Operational data - never seed | No |
| dbo | IncidentRequestCaseAction | 34 | 5 | Operational data - never seed | No |
| dbo | Season | 8 | 32 | Operational data - never seed | No |
| dbo | UserRole | 4 | 0 | Operational data - never seed | No |
| dbo | Users | 12 | 0 | Operational data - never seed | No |
| dbo | VW_HrEmployeeProfileView | 9 | 10 | Operational data - never seed | No |

⚠️ = flagged for follow-up review, not yet explicitly approved for seeding — currently
defaulted to "never seed" out of caution. See notes below.

### Flagged for follow-up review

- **`dbo.APP_RCAFactorCategory`** — small taxonomy-shaped table (7 rows) - looks lookup-like but not explicitly approved for seeding
- **`dbo.APP_RCASuggestion`** — unclear if curated suggestion bank or per-case generated text (42 rows) - treated as operational until confirmed
- **`dbo.APP_DrawerLabel`** — small taxonomy-shaped table (15 rows) - looks lookup-like but not explicitly approved for seeding
- **`dbo.Instance`** — 0 rows, legacy UniqueID-PK naming convention unlike rest of schema - possibly vestigial
- **`dbo.Parameter`** — 0 rows, legacy UniqueID-PK naming convention unlike rest of schema - possibly vestigial
- **`dbo.Role`** — 0 rows, legacy UniqueID-PK naming convention, distinct from the actively-used APP_Roles - possibly vestigial

## Known cosmetic quirks (documented, not fixed)

- **Login/user name mismatch**: the SQL Server *login* used by the application is
  `HCAT_Insight` (matches `config/db_settings.json`, confirmed via `SUSER_SNAME()`),
  but the *database user* it's mapped to inside `IncidentManager` is named
  `HACT_Insight` (letters transposed, confirmed via `USER_NAME()`). Cosmetic only —
  authentication and permissions both work correctly. Left as-is.

## Schema objects other than tables

Zero views, stored procedures, functions, or triggers exist in the live database as
of this inspection — all business logic lives in the Python backend. The
`005_create_views.sql`/`006_create_stored_procedures.sql`/`007_create_triggers.sql`
install files are intentionally empty placeholders.

## Existing migration history

`dbo.SchemaMigrationHistory` already existed in production before this package was
built, with 2 rows: `phase_ml_s1_create_ml_schema_and_tables` and
`phase_ml_s8_historical_migration_schema` (both applied 2026-07-16). This package's
`install/011_record_database_version.sql` extends that same table/convention rather
than introducing a new one — see `migrations/README.md`.