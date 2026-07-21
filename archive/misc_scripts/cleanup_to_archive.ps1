# ============================================================
# CLEANUP SCRIPT - Move Development Files to Archive
# ============================================================
# This script moves test files, documentation, debug scripts,
# and other development artifacts to an archive folder.
# 
# REVIEW THIS SCRIPT BEFORE RUNNING!
# Run with: .\cleanup_to_archive.ps1
# ============================================================

$projectRoot = "C:\Users\IT\Documents\GitHub Repository\Patient_Feedback"
$archiveDir = Join-Path $projectRoot "archive"

# Create archive directory structure
$subfolders = @(
    "documentation",
    "test_scripts", 
    "debug_scripts",
    "migration_scripts",
    "output_files",
    "sql_scripts",
    "misc_scripts"
)

Write-Host "Creating archive directory structure..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
foreach ($folder in $subfolders) {
    New-Item -ItemType Directory -Path (Join-Path $archiveDir $folder) -Force | Out-Null
}

# ============================================================
# 1. DOCUMENTATION FILES (.md files - completion reports, etc.)
# ============================================================
$docFiles = @(
    "48. Error.md",
    "ADMIN_ROUTER_PROTECTION_COMPLETE.md",
    "API_V2_CONTRACT_FREEZE.md",
    "B-I18_REGISTRATION_COMPLETE.md",
    "BACKEND_EXPLANATION_FIX_SUMMARY.md",
    "BACKEND_MIGRATION_PROGRESS_SUMMARY.md",
    "BACKEND_ORG_UNIT_SECTION_CREATION_REPORT.md",
    "BACKEND_STEP_5_COMPLETION_REPORT.md",
    "BACKEND_STEP_6_COMPLETION_REPORT.md",
    "BACKEND_STEP_7_COMPLETION_REPORT.md",
    "BULK_DELETE_USERS_IMPLEMENTATION_COMPLETE.md",
    "B_B1_COMPLETION_REPORT.md",
    "B_B2_COMPLETION_REPORT.md",
    "B_B4_COMPLETION_REPORT.md",
    "B_B5_COMPLETION_REPORT.md",
    "CRISIS_SOLVED.md",
    "DATE_RANGE_EXPORT_FIX.md",
    "EMPLOYEE_LINKAGE_COMPLETE.md",
    "EMPLOYEE_VALIDATION_FIX_COMPLETE.md",
    "EMPLOYEE_VALIDATION_ISSUE.md",
    "ENDPOINTS_VISUAL_GUIDE.md",
    "EXPLANATION_SERVICES_QUICK_REFERENCE.md",
    "FOLLOW_UP_API_SCHEMA.md",
    "FORCE_CLOSE_IMPLEMENTATION_COMPLETE.md",
    "FORCE_CLOSE_TESTING_REPORT.md",
    "FRONTEND_ADD_PATIENT_DOCUMENTATION.md",
    "GUARDED_ENDPOINTS_COMPLETION_REPORT.md",
    "GUARDS_QUICK_REFERENCE.md",
    "GUARDS_VISUAL_GUIDE.md",
    "G_B10_COMPLETION_REPORT.md",
    "G_B11_COMPLETION_REPORT.md",
    "G_B12_COMPLETION_REPORT.md",
    "G_B8_COMPLETION_REPORT.md",
    "G_B9_COMPLETION_REPORT.md",
    "HARDWARE_CONFIG_GUIDE.md",
    "HISTORY_AGGREGATE_REPORTS_COMPLETION.md",
    "HISTORY_SEARCH_COMPLETION_REPORT.md",
    "INSERT_PAGE_DOCTOR_SEARCH_FIX.md",
    "K_SVC4_COMPLETION_REPORT.md",
    "K_SVC5_COMPLETION_REPORT.md",
    "MIGRATION_PROGRESS_ENDPOINT_IMPLEMENTATION.md",
    "MIGRATION_PROGRESS_QUICK_REF.md",
    "MODULE_5_2_EXECUTION_REPORT.md",
    "MULTI_EXPORT_FEATURE.md",
    "OFFLINE_ONLINE_SWITCHING_GUIDE.md",
    "ORGANIZATION_SELECTOR_GUIDE.md",
    "ORGANIZATION_SELECTOR_SOLUTION.md",
    "ORG_SELECTION_QUICK_REF.md",
    "ORG_SELECTOR_IMPLEMENTATION_COMPLETE.md",
    "ORG_SELECTOR_QUICK_REF.md",
    "PDF_GENERATION_REBUILD.md",
    "PHASE1_COMPLETION_REPORT.md",
    "PHASE1_PROGRESS_TRACKING_COMPLETE.md",
    "PHASE1_TEST_RESULTS.md",
    "PHASE2_COMPLETE_SUMMARY.md",
    "PHASE2_ML_DB_GROWTH_COMPLETE.md",
    "PHASE2_PROMPT5_COMPLETE.md",
    "PHASE3_5_TESTING_COMPLETE.md",
    "PHASE3_FIX_PLAN.md",
    "PHASE4_STEP4_1_COMPLETE.md",
    "PHASE5_API_ENDPOINTS_COMPLETION_REPORT.md",
    "PHASE5_MODULE_TEST_PROCEDURES.md",
    "PHASE5_QUICK_REFERENCE.md",
    "PHASE5_SUMMARY.md",
    "PHASE5_TESTING_COMPLETE_REPORT.md",
    "PHASE6_COMPLETION_REPORT.md",
    "PHASE_2_5_RUNTIME_TEST_REPORT.md",
    "PHASE_2_5_STRUCTURAL_AUDIT.md",
    "PHASE_3_5_COMPLETE_SUMMARY.md",
    "PHASE_DR_B_COMPLETION_REPORT.md",
    "PHASE_G_B4_COMPLETION_REPORT.md",
    "PHASE_G_B5_COMPLETION_REPORT.md",
    "PHASE_G_B6_COMPLETION_REPORT.md",
    "PHASE_G_B7_COMPLETION_REPORT.md",
    "PHASE_M_COMPLETION_REPORT.md",
    "PHASE_M_M1_M2_COMPLETE.md",
    "PROMPT2_PART2_COMPLETE_SUMMARY.md",
    "PROMPT3_COMPLETE_SUMMARY.md",
    "QUICK_REFERENCE_EXPLANATIONS.md",
    "RBAC_DIAGNOSIS.md",
    "README_ORG_SELECTORS.md",
    "REPORTING_SERVICES_ANALYSIS.md",
    "SCOPING_CURRENT_STATE.md",
    "SEASONAL_COMPARISON_FEATURE.md",
    "SEASONAL_QUARTER_TRIMESTER_IMPLEMENTATION.md",
    "SEASONAL_QUICK_REFERENCE.md",
    "SIGNIN_PLANNING_QA.md",
    "STEP_3_10_ARCHITECTURE.md",
    "STEP_3_10_COMPLETION_REPORT.md",
    "STEP_3_10_STRICT_VERIFICATION.md",
    "STEP_3_13_INTEGRATION_TEST_RESULTS.md",
    "STEP_3_5_0_API_SURFACE_AUDIT.md",
    "STEP_3_5_5_INSIGHT_DELAY_DECISION.md",
    "SYSTEM_SETTINGS_IMPLEMENTATION.md",
    "SYSTEM_SETTINGS_QUICK_START.md",
    "THREE_TYPE_EXPLANATION_SYSTEM.md",
    "UI_ENDPOINT_MAPPING.md",
    "USER_CREDENTIALS_REFERENCE.md",
    "USER_EDIT_FEATURE_IMPLEMENTATION.md",
    "USER_EDIT_FEATURE_TESTING_COMPLETE.md",
    "USER_WORKLOAD_ENDPOINT_COMPLETE.md",
    "USER_WORKLOAD_IMPLEMENTATION_NOTES.md",
    "USER_WORKLOAD_QUICK_REFERENCE.md",
    "VM_DEPLOYMENT_REQUIREMENTS.md",
    "WORKER_REPORTING_DISCUSSION.md",
    "WORKER_ROLE_ANALYSIS.md",
    "WORKFLOW_CONTRACT_CHANGE_REJECTION_REWORK_LOOP.md",
    "WORKFLOW_VALIDATION_REPORT.md"
)

Write-Host "`nMoving documentation files..." -ForegroundColor Yellow
$movedDocs = 0
foreach ($file in $docFiles) {
    $sourcePath = Join-Path $projectRoot $file
    if (Test-Path $sourcePath) {
        Move-Item -Path $sourcePath -Destination (Join-Path $archiveDir "documentation") -Force
        $movedDocs++
    }
}
Write-Host "  Moved $movedDocs documentation files" -ForegroundColor Green

# ============================================================
# 2. TEST SCRIPTS (test_*.py files)
# ============================================================
Write-Host "`nMoving test scripts..." -ForegroundColor Yellow
$testFiles = Get-ChildItem -Path $projectRoot -Filter "test_*.py" -File
$movedTests = 0
foreach ($file in $testFiles) {
    Move-Item -Path $file.FullName -Destination (Join-Path $archiveDir "test_scripts") -Force
    $movedTests++
}
# Also move TEST_TRAINING_API.py
$trainingApi = Join-Path $projectRoot "TEST_TRAINING_API.py"
if (Test-Path $trainingApi) {
    Move-Item -Path $trainingApi -Destination (Join-Path $archiveDir "test_scripts") -Force
    $movedTests++
}
Write-Host "  Moved $movedTests test scripts" -ForegroundColor Green

# ============================================================
# 3. DEBUG/CHECK/VERIFY/DIAGNOSE SCRIPTS
# ============================================================
$debugScripts = @(
    "check_action_item_columns.py",
    "check_action_table.py",
    "check_actual_schema.py",
    "check_attributes.py",
    "check_doctor_view.py",
    "check_employee_incident_link.py",
    "check_employee_table_schema.py",
    "check_hr_employees.py",
    "check_incident_required_fields.py",
    "check_incident_schema.py",
    "check_org_unit_column.py",
    "check_patients.py",
    "check_patient_tables.py",
    "check_policy_tables.py",
    "check_satisfaction_schema.py",
    "check_season_table.py",
    "check_season_table_name.py",
    "check_statuses.py",
    "check_status_codes.py",
    "check_users.py",
    "check_users_table.py",
    "check_worker_schema.py",
    "compare_columns.py",
    "compare_employee_data.py",
    "debug_doctor_service.py",
    "debug_patient_word.py",
    "debug_rca_schema.py",
    "debug_seasonal_empty.py",
    "debug_test_a2.py",
    "demo_strict_verification.py",
    "diagnose_phase3_stack.py",
    "diagnose_returned_to_section.py",
    "find_incident_fields.py",
    "find_org_unit_table.py",
    "find_test_credentials.py",
    "find_user_table.py",
    "get_valid_ids.py",
    "get_valid_test_data.py",
    "get_valid_test_ids.py",
    "list_all_users.py",
    "query_org_structure.py",
    "query_org_units.py",
    "query_rbac_data.py",
    "verify_all_word_reports.py",
    "verify_doctor_word_reports.py",
    "verify_employee_linkage.py",
    "verify_patient_word_report.py",
    "verify_seasonal_data.py",
    "verify_step3_10.py"
)

Write-Host "`nMoving debug/check/verify scripts..." -ForegroundColor Yellow
$movedDebug = 0
foreach ($file in $debugScripts) {
    $sourcePath = Join-Path $projectRoot $file
    if (Test-Path $sourcePath) {
        Move-Item -Path $sourcePath -Destination (Join-Path $archiveDir "debug_scripts") -Force
        $movedDebug++
    }
}
Write-Host "  Moved $movedDebug debug scripts" -ForegroundColor Green

# ============================================================
# 4. MIGRATION/EXECUTION SCRIPTS
# ============================================================
$migrationScripts = @(
    "add_feedback_text_column.py",
    "add_universal_section_role.py",
    "assign_universal_role.py",
    "create_settings_table.py",
    "create_test_inbox_data.py",
    "create_universal_section_user.py",
    "execute_action_migration.py",
    "execute_check_fk.py",
    "execute_create_table.py",
    "execute_force_close_migration.py",
    "execute_phase_g_b1_migration.py",
    "execute_phase_g_b2_migration.py",
    "execute_phase_g_b3_migration.py",
    "phase1_check_current_state.py",
    "phase1_migrate_employee_table.py",
    "phase1_test.py",
    "phase2_test.py",
    "phase3_test.py",
    "phase4_test.py",
    "phase5_test.py",
    "phase6_test.py",
    "run_employee_migration.py",
    "scope_guards_usage_examples.py",
    "setup_test_data_step3_15.py",
    "simulate_returned_subcase.py",
    "use_ai_models.py"
)

Write-Host "`nMoving migration/execution scripts..." -ForegroundColor Yellow
$movedMigration = 0
foreach ($file in $migrationScripts) {
    $sourcePath = Join-Path $projectRoot $file
    if (Test-Path $sourcePath) {
        Move-Item -Path $sourcePath -Destination (Join-Path $archiveDir "migration_scripts") -Force
        $movedMigration++
    }
}
Write-Host "  Moved $movedMigration migration scripts" -ForegroundColor Green

# ============================================================
# 5. SQL SCRIPTS
# ============================================================
$sqlScripts = @(
    "ALTER_EMPLOYEE_TABLE.sql",
    "CREATE_BULK_ADMIN_USERS.sql",
    "migration_add_force_close_tracking.sql"
)

Write-Host "`nMoving SQL scripts..." -ForegroundColor Yellow
$movedSql = 0
foreach ($file in $sqlScripts) {
    $sourcePath = Join-Path $projectRoot $file
    if (Test-Path $sourcePath) {
        Move-Item -Path $sourcePath -Destination (Join-Path $archiveDir "sql_scripts") -Force
        $movedSql++
    }
}
Write-Host "  Moved $movedSql SQL scripts" -ForegroundColor Green

# ============================================================
# 6. OUTPUT FILES (.docx, .txt, .log, .zip)
# ============================================================
Write-Host "`nMoving output files..." -ForegroundColor Yellow
$movedOutput = 0

# Move .docx files
$docxFiles = Get-ChildItem -Path $projectRoot -Filter "*.docx" -File
foreach ($file in $docxFiles) {
    Move-Item -Path $file.FullName -Destination (Join-Path $archiveDir "output_files") -Force
    $movedOutput++
}

# Move specific .txt output files
$txtOutputFiles = @(
    "production_test_output.txt",
    "test_output.txt",
    "test_phase4_results.txt",
    "test_rejection_output.txt",
    "troubleshooting_output.txt",
    "troubleshoot_output.log",
    "verify_insert_output.txt",
    "verify_output.txt"
)
foreach ($file in $txtOutputFiles) {
    $sourcePath = Join-Path $projectRoot $file
    if (Test-Path $sourcePath) {
        Move-Item -Path $sourcePath -Destination (Join-Path $archiveDir "output_files") -Force
        $movedOutput++
    }
}

# Move .zip test archives
$zipFiles = Get-ChildItem -Path $projectRoot -Filter "*.zip" -File
foreach ($file in $zipFiles) {
    Move-Item -Path $file.FullName -Destination (Join-Path $archiveDir "output_files") -Force
    $movedOutput++
}
Write-Host "  Moved $movedOutput output files" -ForegroundColor Green

# ============================================================
# 7. MISC FILES
# ============================================================
$miscFiles = @(
    "project_paths.py",
    "__init__.py"
)

Write-Host "`nMoving misc files..." -ForegroundColor Yellow
$movedMisc = 0
foreach ($file in $miscFiles) {
    $sourcePath = Join-Path $projectRoot $file
    if (Test-Path $sourcePath) {
        Move-Item -Path $sourcePath -Destination (Join-Path $archiveDir "misc_scripts") -Force
        $movedMisc++
    }
}
Write-Host "  Moved $movedMisc misc files" -ForegroundColor Green

# ============================================================
# 8. MOVE FOLDERS (optional - uncomment if desired)
# ============================================================
Write-Host "`nMoving development folders..." -ForegroundColor Yellow
$movedFolders = 0

# Folders that can be archived
# KEEPING: GETTING_Schema, model_training, models_directory, tools
$foldersToMove = @(
    "data_exploration",
    "test_phase8_outputs",
    "Explanatory Notes"
)

foreach ($folder in $foldersToMove) {
    $sourcePath = Join-Path $projectRoot $folder
    if (Test-Path $sourcePath) {
        Move-Item -Path $sourcePath -Destination $archiveDir -Force
        $movedFolders++
    }
}
Write-Host "  Moved $movedFolders folders" -ForegroundColor Green

# ============================================================
# SUMMARY
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Documentation files moved: $movedDocs"
Write-Host "Test scripts moved:        $movedTests"
Write-Host "Debug scripts moved:       $movedDebug"
Write-Host "Migration scripts moved:   $movedMigration"
Write-Host "SQL scripts moved:         $movedSql"
Write-Host "Output files moved:        $movedOutput"
Write-Host "Misc files moved:          $movedMisc"
Write-Host "Folders moved:             $movedFolders"
Write-Host "------------------------------------------------------------"
$total = $movedDocs + $movedTests + $movedDebug + $movedMigration + $movedSql + $movedOutput + $movedMisc + $movedFolders
Write-Host "TOTAL ITEMS MOVED:         $total" -ForegroundColor Yellow
Write-Host "`nAll files moved to: $archiveDir" -ForegroundColor Cyan
Write-Host "`nRemaining in project root:"
Write-Host "  - backend/     (API code)"
Write-Host "  - frontend/    (Frontend code)"
Write-Host "  - api/         (API code)"
Write-Host "  - venv/        (Virtual environment)"
Write-Host "  - requirements.txt"
Write-Host "  - .gitignore"
Write-Host "============================================================`n"
