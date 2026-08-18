#!/usr/bin/env bash
# Post-install / post-update validation pass: container status, backend and
# frontend reachability, and the database validation scripts from
# database/sqlserver/validation/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
set +e; set -uo pipefail  # not -e: we want to run every check and report all results
load_env

PASS=0
FAIL=0

check() {
    local desc="$1"; local result="$2"
    if [ "$result" = "0" ]; then
        echo "  [PASS] $desc"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $desc"
        FAIL=$((FAIL + 1))
    fi
}

# Scoped MSYS_NO_PATHCONV=1 (no-op on the real Debian target) so Git-Bash-on-
# Windows doesn't mangle the absolute /opt/mssql-tools18/... container path
# into a Windows path when an engineer tests this on a dev machine. Scoped to
# just this one command, not exported globally, so it doesn't also break
# --env-file's *host*-path argument elsewhere in this script.
sqlcmd_exec() {
    MSYS_NO_PATHCONV=1 docker exec "$@"
}

echo "=== 1. Container status ==="
compose ps

echo ""
echo "=== 2. Container health ==="
for svc in sqlserver backend frontend; do
    cid="$(compose ps -q "$svc")"
    status="$(docker inspect --format='{{.State.Health.Status}}' "$cid" 2>/dev/null || echo "unknown")"
    check "$svc is healthy (was: $status)" "$([ "$status" = "healthy" ] && echo 0 || echo 1)"
done

echo ""
echo "=== 3. Backend API reachability ==="
curl -sf -m 10 "http://localhost:${BACKEND_HOST_PORT:-8100}/api/status" >/dev/null
check "backend /api/status responds" "$?"

echo ""
echo "=== 4. Frontend reachability ==="
curl -sf -m 10 -o /dev/null "http://localhost:${FRONTEND_HOST_PORT:-8101}/"
check "frontend / responds" "$?"

curl -sf -m 10 -o /dev/null "http://localhost:${FRONTEND_HOST_PORT:-8101}/api/status"
check "frontend -> backend reverse proxy works" "$?"

echo ""
echo "=== 5. Database validation ==="
CONTAINER="${PROJECT_NAME:-pfms}-sqlserver"
CONTAINER_BACKEND="${PROJECT_NAME:-pfms}-backend"
DB_NAME="${DB_DATABASE:-IncidentManager}"
for sql_file in "$LIVE_ROOT"/database/sqlserver/validation/*.sql; do
    name="$(basename "$sql_file")"
    sqlcmd_exec -i "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" \
        < "$sql_file" >"/tmp/verify_${name}.out" 2>&1
    check "$name ran without error (see /tmp/verify_${name}.out for details)" "$?"
done

echo ""
echo "=== 6. Organizational unit / user provisioning integrity ==="
MANIFEST="$LIVE_ROOT/database/sqlserver/seed/provisioning.v1.manifest.json"
if [ ! -f "$MANIFEST" ]; then
    check "provisioning.v1.manifest.json exists" 1
else
    check "provisioning.v1.manifest.json exists" 0
    expected_org_units="$(grep -oP '"org_units_total":\s*\K[0-9]+' "$MANIFEST")"
    expected_users_total="$(grep -oP '"users_total":\s*\K[0-9]+' "$MANIFEST")"
    expected_users_active="$(grep -oP '"users_active":\s*\K[0-9]+' "$MANIFEST")"

    actual_org_units="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.AdminsrationUnit" | tr -d '[:space:]')"
    actual_users_total="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.APP_Users" | tr -d '[:space:]')"
    actual_users_active="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.APP_Users WHERE IsActive = 1" | tr -d '[:space:]')"
    actual_scope_orphans="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.APP_UserRoleScope s LEFT JOIN dbo.APP_Users u ON s.UserID = u.UserID WHERE u.UserID IS NULL" | tr -d '[:space:]')"
    actual_sourcemap_orphans="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.APP_UserSourceIDMap m LEFT JOIN dbo.APP_Users u ON m.LocalUserID = u.UserID WHERE u.UserID IS NULL" | tr -d '[:space:]')"

    check "org unit count matches manifest (expected=$expected_org_units actual=$actual_org_units)" \
        "$([ "$expected_org_units" = "$actual_org_units" ] && echo 0 || echo 1)"
    check "total user count matches manifest (expected=$expected_users_total actual=$actual_users_total)" \
        "$([ "$expected_users_total" = "$actual_users_total" ] && echo 0 || echo 1)"
    check "active user count matches manifest (expected=$expected_users_active actual=$actual_users_active)" \
        "$([ "$expected_users_active" = "$actual_users_active" ] && echo 0 || echo 1)"
    check "no orphaned APP_UserRoleScope rows (found=$actual_scope_orphans)" \
        "$([ "$actual_scope_orphans" = "0" ] && echo 0 || echo 1)"
    check "no orphaned APP_UserSourceIDMap rows (found=$actual_sourcemap_orphans)" \
        "$([ "$actual_sourcemap_orphans" = "0" ] && echo 0 || echo 1)"
fi

echo ""
echo "=== 7. Custom Table Views integrity ==="
if [ ! -f "$MANIFEST" ]; then
    check "provisioning.v1.manifest.json exists (custom views check)" 1
else
    expected_custom_views="$(grep -oP '"custom_views_total":\s*\K[0-9]+' "$MANIFEST")"
    actual_custom_views="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.APP_CUSTOM_VIEWS" | tr -d '[:space:]')"
    actual_custom_view_orphans="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.APP_CustomViewSourceIDMap m LEFT JOIN dbo.APP_CUSTOM_VIEWS v ON m.LocalViewID = v.ViewID WHERE v.ViewID IS NULL" | tr -d '[:space:]')"

    if [ -z "$expected_custom_views" ]; then
        echo "  (manifest predates custom_views_total -- skipping count check, only checking presence)"
        check "at least one Custom Table View exists (found=$actual_custom_views)" \
            "$([ "$actual_custom_views" -gt 0 ] && echo 0 || echo 1)"
    else
        check "custom view count matches manifest (expected=$expected_custom_views actual=$actual_custom_views)" \
            "$([ "$expected_custom_views" = "$actual_custom_views" ] && echo 0 || echo 1)"
    fi
    check "no orphaned APP_CustomViewSourceIDMap rows (found=$actual_custom_view_orphans)" \
        "$([ "$actual_custom_view_orphans" = "0" ] && echo 0 || echo 1)"
fi

echo ""
echo "=== 8. Drawer Notes labels ==="
actual_drawer_labels="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.APP_DrawerLabel WHERE IsActive = 1" | tr -d '[:space:]')"
check "at least one active Drawer Note label exists (found=$actual_drawer_labels)" \
    "$([ "$actual_drawer_labels" -gt 0 ] && echo 0 || echo 1)"

echo ""
echo "=== 9. Speech-to-Text model asset transfer ==="
# The 4 files a CTranslate2 Faster-Whisper model actually needs to load (see
# scripts/export_whisper_model.sh) -- checked individually because a
# directory can be non-empty (partial/truncated extraction, or just the
# .cache/huggingface/ download metadata left behind) without being loadable.
WHISPER_HOST_DIR="$LIVE_ROOT/assets/whisper-model-medium"
for f in config.json model.bin tokenizer.json vocabulary.txt; do
    check "host: $f present and non-empty at $WHISPER_HOST_DIR" \
        "$([ -s "$WHISPER_HOST_DIR/$f" ] && echo 0 || echo 1)"
done
# Also confirm the read-only bind mount into the backend container actually
# sees the same files -- catches a wrong host path or a stale container that
# was started before extraction completed.
for f in config.json model.bin tokenizer.json vocabulary.txt; do
    check "container: $f visible at /models/whisper-medium (mount OK)" \
        "$(sqlcmd_exec "$CONTAINER_BACKEND" test -s "/models/whisper-medium/$f" >/dev/null 2>&1 && echo 0 || echo 1)"
done

echo ""
echo "=== 10. Force Close Policy & RCA Suggestions default configuration ==="
actual_force_close_settings="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.APP_SystemSettings WHERE SettingKey IN ('automatic_force_close_enabled','section_deadline_days','department_deadline_days','administration_deadline_days')" | tr -d '[:space:]')"
check "all 4 Force Close Policy settings present (found=$actual_force_close_settings, expected=4)" \
    "$([ "$actual_force_close_settings" = "4" ] && echo 0 || echo 1)"

actual_rca_categories="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.APP_RCAFactorCategory" | tr -d '[:space:]')"
check "at least one RCA factor category exists (found=$actual_rca_categories)" \
    "$([ "$actual_rca_categories" -gt 0 ] && echo 0 || echo 1)"

actual_rca_suggestions="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.APP_RCASuggestion" | tr -d '[:space:]')"
check "at least one RCA suggestion exists (found=$actual_rca_suggestions)" \
    "$([ "$actual_rca_suggestions" -gt 0 ] && echo 0 || echo 1)"

actual_rca_unpaired="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 -Q "SET NOCOUNT ON; SELECT COUNT(*) FROM dbo.APP_RCASuggestion WHERE PairedSuggestionID IS NULL" | tr -d '[:space:]')"
check "no unpaired RCA cause/action suggestions (found=$actual_rca_unpaired)" \
    "$([ "$actual_rca_unpaired" = "0" ] && echo 0 || echo 1)"

echo ""
echo "=== Summary: $PASS passed, $FAIL failed ==="
echo "NOTE: this script covers infrastructure/database integrity only. Run"
echo "      scripts/qualify_offline_installation.sh for functional checks"
echo "      (ML classification, Speech-to-Text, dashboard scope, patient"
echo "      search, NER removal, publication batches)."
[ "$FAIL" -eq 0 ]
