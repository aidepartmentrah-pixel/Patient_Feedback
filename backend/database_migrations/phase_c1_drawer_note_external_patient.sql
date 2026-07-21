-- =====================================================
-- Session C1: External Patient Linkage for Drawer Notes
-- =====================================================
-- Purpose: Let a drawer note link to a patient that came from the Hospital
-- Directory API (external), not just APP_RESERVE_PATIENT (reserve).
--
-- The external API's patient identity is a (patient_id, visit_id) string
-- pair, not an int, so it CANNOT be stored in the existing
-- PatientAdmissionID int column (see backend/core/hospital_directory_client.py
-- encode_external_patient_id — that encoding is only used at the service
-- layer to route lookups, never persisted to SQL). Instead, two new nullable
-- columns are added:
--   - ExternalPatientID   — the opaque "ext__{patient_id}__{visit_id}" id
--   - ExternalPatientName — the patient's full_name, SNAPSHOTTED at note
--     creation time (same pattern APP_IncidentCaseEmployee already uses for
--     HR employee names — see Investigation 2 §3) so listing/reading notes
--     never needs a live API call.
--
-- A note still links to AT MOST ONE patient, via EITHER
-- PatientAdmissionID (reserve, unchanged) OR ExternalPatientID (new) —
-- enforced in application code (api_v2/services/drawer_note_service.py),
-- not a CHECK constraint, to match this table's existing "no DB-level
-- business rules" convention (see phase_g_b1_create_drawer_note_table.sql).
--
-- Existing notes and the reserve-linkage path are completely unaffected —
-- both new columns are nullable and default NULL.
-- =====================================================

IF NOT EXISTS (
    SELECT * FROM sys.columns
    WHERE object_id = OBJECT_ID('dbo.APP_DrawerNote') AND name = 'ExternalPatientID'
)
BEGIN
    ALTER TABLE dbo.APP_DrawerNote ADD ExternalPatientID NVARCHAR(128) NULL;
    PRINT 'Column ExternalPatientID added to APP_DrawerNote.';
END
ELSE
BEGIN
    PRINT 'Column ExternalPatientID already exists on APP_DrawerNote.';
END
GO

IF NOT EXISTS (
    SELECT * FROM sys.columns
    WHERE object_id = OBJECT_ID('dbo.APP_DrawerNote') AND name = 'ExternalPatientName'
)
BEGIN
    ALTER TABLE dbo.APP_DrawerNote ADD ExternalPatientName NVARCHAR(300) NULL;
    PRINT 'Column ExternalPatientName added to APP_DrawerNote.';
END
ELSE
BEGIN
    PRINT 'Column ExternalPatientName already exists on APP_DrawerNote.';
END
GO

-- Verify the columns
SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'APP_DrawerNote'
ORDER BY ORDINAL_POSITION;
GO
