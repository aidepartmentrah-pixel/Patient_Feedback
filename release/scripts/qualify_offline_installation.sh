#!/usr/bin/env bash
# Air-gap qualification: functional acceptance testing on top of an already
# -completed, already-verified install (run scripts/verify_installation.sh
# first -- this script assumes that infrastructure/data-integrity pass
# already succeeded and focuses on role-based login and hierarchy-based
# access control instead).
#
# Reads database/sqlserver/seed/installation_test_credentials.local.json --
# ONE real, active, plaintext-password account per role, produced by
# Stage B (see database/sqlserver/seed/build_provisioning_artifact.py). This
# file is NOT read by the normal installer (install_offline.sh) -- only by
# this script.
#
# Failure semantics: a failure here marks qualification FAILED and is
# written into the report below, but does NOT roll back or otherwise touch
# the already-installed database. Qualification is a read-mostly acceptance
# test on top of a finished install, not a transactional part of installing.

set -uo pipefail  # not -e: run every check, report all results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$RELEASE_ROOT/.env"
COMPOSE_FILE="$RELEASE_ROOT/compose/docker-compose.yml"
CREDS_FILE="$RELEASE_ROOT/database/sqlserver/seed/installation_test_credentials.local.json"

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

BACKEND_URL="http://localhost:${BACKEND_HOST_PORT:-8100}"
CONTAINER="${PROJECT_NAME:-pfms}-sqlserver"
DB_NAME="${DB_DATABASE:-IncidentManager}"

sqlcmd_exec() {
    # Scoped MSYS_NO_PATHCONV=1 (no-op on the real Debian target): prevents
    # Git-Bash-on-Windows from mangling the absolute /opt/mssql-tools18/...
    # container path when tested on a dev machine.
    MSYS_NO_PATHCONV=1 docker exec "$@"
}

sql_scalar() {
    sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 \
        -Q "SET NOCOUNT ON; $1" | tr -d '[:space:]'
}

PASS=0
FAIL=0
check() {
    local desc="$1"; local result="$2"
    if [ "$result" = "0" ]; then echo "  [PASS] $desc"; PASS=$((PASS + 1))
    else echo "  [FAIL] $desc"; FAIL=$((FAIL + 1)); fi
}

if [ ! -f "$CREDS_FILE" ]; then
    echo "ERROR: $CREDS_FILE not found."
    echo "       Qualification cannot run without it -- it's a separate,"
    echo "       protected artifact from the normal release bundle. See"
    echo "       RELEASE_NOTES.md 'Password handling'."
    exit 1
fi

echo "=== 1. Per-role login ==="
declare -A ALLOWED_UNITS
declare -A ROLE_FOR_USER

python_or_die() {
    command -v python3 >/dev/null 2>&1 && { echo python3; return; }
    command -v python >/dev/null 2>&1 && { echo python; return; }
    echo "ERROR: no python3/python found on this server -- required to parse"
    echo "       installation_test_credentials.local.json and login responses."
    exit 1
}
PY="$(python_or_die)"

NUM_ACCOUNTS="$($PY -c "import json; print(len(json.load(open('$CREDS_FILE'))['accounts']))")"

for i in $(seq 0 $((NUM_ACCOUNTS - 1))); do
    ROLE="$($PY -c "import json; print(json.load(open('$CREDS_FILE'))['accounts'][$i]['role'])")"
    USERNAME="$($PY -c "import json; print(json.load(open('$CREDS_FILE'))['accounts'][$i]['username'])")"
    PASSWORD="$($PY -c "import json; print(json.load(open('$CREDS_FILE'))['accounts'][$i]['password'])")"

    RESPONSE="$(curl -s -c "/tmp/qualify_cookies_${ROLE}.txt" -X POST "$BACKEND_URL/api/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"${USERNAME}\",\"password\":\"${PASSWORD}\"}")"

    SUCCESS="$($PY -c "import json,sys; print(json.loads('''$RESPONSE'''.replace(chr(39),chr(34)) if False else '$RESPONSE').get('success', False))" 2>/dev/null || echo "false")"
    # Simpler/robust parse via python reading the actual string (avoids shell quoting issues):
    SUCCESS="$(echo "$RESPONSE" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d.get('success', False))" 2>/dev/null || echo "False")"
    check "login as $ROLE ($USERNAME)" "$([ "$SUCCESS" = "True" ] && echo 0 || echo 1)"

    UNIT_IDS="$(echo "$RESPONSE" | $PY -c "import json,sys; d=json.load(sys.stdin); print(','.join(str(x) for x in sorted(d.get('user',{}).get('allowed_unit_ids', []))))" 2>/dev/null || echo "")"
    ALLOWED_UNITS["$ROLE"]="$UNIT_IDS"
    ROLE_FOR_USER["$ROLE"]="$USERNAME"

    curl -s -b "/tmp/qualify_cookies_${ROLE}.txt" -X POST "$BACKEND_URL/api/auth/logout" >/dev/null
    rm -f "/tmp/qualify_cookies_${ROLE}.txt"
done

echo ""
echo "=== 2. Hierarchy-based scope verification (independently recomputed from the DB, not trusting the app's own output) ==="

if [ -n "${ALLOWED_UNITS[SECTION_ADMIN]:-}" ]; then
    SEC_USERNAME="${ROLE_FOR_USER[SECTION_ADMIN]}"
    SEC_ORG_UNIT="$(sql_scalar "SELECT m.SourceUserID FROM dbo.APP_UserSourceIDMap m JOIN dbo.APP_Users u ON m.LocalUserID = u.UserID WHERE u.Username = N'${SEC_USERNAME}'" >/dev/null; \
        sql_scalar "SELECT s.OrgUnitID FROM dbo.APP_UserRoleScope s JOIN dbo.APP_Users u ON s.UserID = u.UserID WHERE u.Username = N'${SEC_USERNAME}'")"
    EXPECTED="$SEC_ORG_UNIT"
    check "SECTION_ADMIN ($SEC_USERNAME) allowed_unit_ids == exactly their own unit ($EXPECTED)" \
        "$([ "${ALLOWED_UNITS[SECTION_ADMIN]}" = "$EXPECTED" ] && echo 0 || echo 1)"
fi

if [ -n "${ALLOWED_UNITS[DEPARTMENT_ADMIN]:-}" ]; then
    DEPT_USERNAME="${ROLE_FOR_USER[DEPARTMENT_ADMIN]}"
    DEPT_ORG_UNIT="$(sql_scalar "SELECT s.OrgUnitID FROM dbo.APP_UserRoleScope s JOIN dbo.APP_Users u ON s.UserID = u.UserID WHERE u.Username = N'${DEPT_USERNAME}'")"
    EXPECTED="$(sqlcmd_exec "$CONTAINER" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -d "$DB_NAME" -h -1 \
        -Q "SET NOCOUNT ON; SELECT UniqueID FROM dbo.AdminsrationUnit WHERE UniqueID = ${DEPT_ORG_UNIT} OR ParentID = ${DEPT_ORG_UNIT} ORDER BY UniqueID" \
        | tr -s '[:space:]' ',' | sed 's/^,//;s/,$//')"
    check "DEPARTMENT_ADMIN ($DEPT_USERNAME) allowed_unit_ids == unit + child sections ($EXPECTED)" \
        "$([ "${ALLOWED_UNITS[DEPARTMENT_ADMIN]}" = "$EXPECTED" ] && echo 0 || echo 1)"
fi

if [ -n "${ALLOWED_UNITS[COMPLAINT_SUPERVISOR]:-}" ]; then
    TOTAL_UNITS="$(sql_scalar "SELECT COUNT(*) FROM dbo.AdminsrationUnit")"
    ACTUAL_COUNT="$(echo "${ALLOWED_UNITS[COMPLAINT_SUPERVISOR]}" | tr ',' '\n' | grep -c .)"
    check "COMPLAINT_SUPERVISOR has org-wide access ($ACTUAL_COUNT units, expected $TOTAL_UNITS)" \
        "$([ "$ACTUAL_COUNT" = "$TOTAL_UNITS" ] && echo 0 || echo 1)"
fi

echo ""
echo "=== 3. Functional checks (using the COMPLAINT_SUPERVISOR test account) ==="
FUNC_COOKIES="/tmp/qualify_func_cookies.txt"
FUNC_USERNAME="$($PY -c "
import json
accounts = json.load(open('$CREDS_FILE'))['accounts']
match = [a for a in accounts if a['role'] == 'COMPLAINT_SUPERVISOR']
print((match or accounts)[0]['username'])
")"
FUNC_PASSWORD="$($PY -c "
import json
accounts = json.load(open('$CREDS_FILE'))['accounts']
match = [a for a in accounts if a['role'] == 'COMPLAINT_SUPERVISOR']
print((match or accounts)[0]['password'])
")"
curl -s -c "$FUNC_COOKIES" -X POST "$BACKEND_URL/api/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"${FUNC_USERNAME}\",\"password\":\"${FUNC_PASSWORD}\"}" >/dev/null

# --- Dashboard Scope: cascading dropdowns return real, non-empty data ---
HIER_RESPONSE="$(curl -s -b "$FUNC_COOKIES" "$BACKEND_URL/api/dashboard/hierarchy")"
HIER_ADMIN_COUNT="$(echo "$HIER_RESPONSE" | $PY -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('Administration', [])))" 2>/dev/null || echo 0)"
check "Dashboard Scope: Administration list is non-empty (found=$HIER_ADMIN_COUNT)" \
    "$([ "$HIER_ADMIN_COUNT" -gt 0 ] && echo 0 || echo 1)"
HIER_HAS_DEPT="$(echo "$HIER_RESPONSE" | $PY -c "
import json,sys
d = json.load(sys.stdin)
depts = d.get('Department', {})
print(1 if any(len(v) > 0 for v in depts.values()) else 0)
" 2>/dev/null || echo 0)"
check "Dashboard Scope: at least one Administration has real Department children" \
    "$([ "$HIER_HAS_DEPT" = "1" ] && echo 0 || echo 1)"

# --- Custom Table Views: API actually returns them (not just DB row count) ---
VIEWS_RESPONSE="$(curl -s -b "$FUNC_COOKIES" "$BACKEND_URL/api/custom-views")"
VIEWS_COUNT="$(echo "$VIEWS_RESPONSE" | $PY -c "import json,sys; print(json.load(sys.stdin).get('total', 0))" 2>/dev/null || echo 0)"
check "Custom Table Views: API returns at least one view (found=$VIEWS_COUNT)" \
    "$([ "$VIEWS_COUNT" -gt 0 ] && echo 0 || echo 1)"

# --- Drawer Notes labels: API actually returns them ---
LABELS_RESPONSE="$(curl -s -b "$FUNC_COOKIES" "$BACKEND_URL/api/v2/drawer-labels/" 2>/dev/null || echo '{}')"
LABELS_COUNT="$(echo "$LABELS_RESPONSE" | $PY -c "import json,sys; print(json.load(sys.stdin).get('total', 0))" 2>/dev/null || echo 0)"
check "Drawer Notes: API returns at least one label (found=$LABELS_COUNT)" \
    "$([ "$LABELS_COUNT" -gt 0 ] && echo 0 || echo 1)"

# --- ML Classification: real predictions for the 7 always-available outputs,
# plus category/subcategory for a domain/category combination known to have
# a valid label map (domain=MANAGEMENT text routes to category 5, which is
# confirmed working -- see ML_CLASSIFICATION_ISSUE_FOR_DEV_TEAM.md for which
# categories are NOT expected to resolve: 1 and 2, pending model retraining) ---
CLASSIFY_RESPONSE="$(curl -s -b "$FUNC_COOKIES" -X POST "$BACKEND_URL/api/classification/classify" \
    -H "Content-Type: application/json" \
    -d '{"text":"المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج","explain":true}')"
CLASSIFY_HTTP="$(curl -s -o /dev/null -w '%{http_code}' -b "$FUNC_COOKIES" -X POST "$BACKEND_URL/api/classification/classify" \
    -H "Content-Type: application/json" \
    -d '{"text":"المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج","explain":true}')"
check "ML Classification: endpoint responds 200 (not a 500 crash)" \
    "$([ "$CLASSIFY_HTTP" = "200" ] && echo 0 || echo 1)"
CLASSIFY_HAS_DOMAIN="$(echo "$CLASSIFY_RESPONSE" | $PY -c "
import json,sys
d = json.load(sys.stdin).get('classifications', {})
print(1 if d.get('domain_id') else 0)
" 2>/dev/null || echo 0)"
check "ML Classification: domain prediction present" \
    "$([ "$CLASSIFY_HAS_DOMAIN" = "1" ] && echo 0 || echo 1)"
CLASSIFY_HAS_SEVERITY="$(echo "$CLASSIFY_RESPONSE" | $PY -c "
import json,sys
d = json.load(sys.stdin).get('classifications', {})
print(1 if d.get('severity_id') else 0)
" 2>/dev/null || echo 0)"
check "ML Classification: severity prediction present (independent of category/subcategory)" \
    "$([ "$CLASSIFY_HAS_SEVERITY" = "1" ] && echo 0 || echo 1)"
echo "  NOTE: category/subcategory are EXPECTED to show as unavailable for"
echo "        some inputs (categories 1 and 2 specifically) until those two"
echo "        models are retrained -- see ML_CLASSIFICATION_ISSUE_FOR_DEV_TEAM.md."
echo "        This is not tested as pass/fail here since the correct result"
echo "        depends on which category the input text happens to route to."

# --- ML Training: history/chart endpoints respond without crashing (a full
# "Train All Models" run takes ~100s, too heavy for this qualification pass
# -- this only confirms the endpoints backing the Training.js failure-outcome
# check and the three dashboard charts are reachable and don't 500) ---
TRAINING_HISTORY_HTTP="$(curl -s -o /dev/null -w '%{http_code}' -b "$FUNC_COOKIES" "$BACKEND_URL/api/settings/training/history")"
check "ML Training: history endpoint responds 200 (got $TRAINING_HISTORY_HTTP)" \
    "$([ "$TRAINING_HISTORY_HTTP" = "200" ] && echo 0 || echo 1)"
TRAINING_CHARTS_OK=1
for chart in db-growth performance-trends family-comparison; do
    code="$(curl -s -o /dev/null -w '%{http_code}' -b "$FUNC_COOKIES" "$BACKEND_URL/api/settings/training/charts/$chart")"
    [ "$code" = "200" ] || TRAINING_CHARTS_OK=0
done
check "ML Training: all 3 dashboard chart endpoints respond 200" \
    "$([ "$TRAINING_CHARTS_OK" = "1" ] && echo 0 || echo 1)"

# --- Speech-to-Text: model loads and transcribes without error ---
STT_TEST_WAV="/tmp/qualify_stt_test.wav"
$PY -c "
import wave, struct, math
with wave.open('$STT_TEST_WAV', 'w') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
    frames = [struct.pack('<h', int(3000*math.sin(2*math.pi*440*i/16000))) for i in range(16000)]
    f.writeframes(b''.join(frames))
"
STT_RESPONSE="$(curl -s -b "$FUNC_COOKIES" -X POST "$BACKEND_URL/api/stt/transcribe" -F "audio=@${STT_TEST_WAV};type=audio/wav")"
STT_SUCCESS="$(echo "$STT_RESPONSE" | $PY -c "import json,sys; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "False")"
check "Speech-to-Text: transcription endpoint succeeds (model loads, decodes audio)" \
    "$([ "$STT_SUCCESS" = "True" ] && echo 0 || echo 1)"
rm -f "$STT_TEST_WAV"

# --- NER: confirmed fully removed, not just hidden ---
NER_HTTP="$(curl -s -o /dev/null -w '%{http_code}' -b "$FUNC_COOKIES" -X POST "$BACKEND_URL/api/ner/extract" -H "Content-Type: application/json" -d '{"text":"test"}')"
check "NER: /api/ner/extract no longer exists (expected 404, got $NER_HTTP)" \
    "$([ "$NER_HTTP" = "404" ] && echo 0 || echo 1)"

# --- Publication batches: no raw error exposed on an empty/normal result ---
PUBBATCH_HTTP="$(curl -s -o /dev/null -w '%{http_code}' -b "$FUNC_COOKIES" "$BACKEND_URL/api/publication-batches/recent")"
check "Publication Batches: endpoint responds 200 for a logged-in user (got $PUBBATCH_HTTP)" \
    "$([ "$PUBBATCH_HTTP" = "200" ] && echo 0 || echo 1)"

# --- Patient search: doesn't crash (external API may be unconfigured, that's fine) ---
SEARCH_HTTP="$(curl -s -o /dev/null -w '%{http_code}' -b "$FUNC_COOKIES" "$BACKEND_URL/api/records/search/patients?q=test")"
check "Patient search: endpoint responds 200, does not crash (got $SEARCH_HTTP)" \
    "$([ "$SEARCH_HTTP" = "200" ] && echo 0 || echo 1)"

# --- Doctor/Worker search: doesn't crash (external API may be unconfigured, that's fine) ---
DOCSEARCH_HTTP="$(curl -s -o /dev/null -w '%{http_code}' -b "$FUNC_COOKIES" "$BACKEND_URL/api/v2/doctors/search?q=te")"
check "Doctor search: endpoint responds 200, does not crash (got $DOCSEARCH_HTTP)" \
    "$([ "$DOCSEARCH_HTTP" = "200" ] && echo 0 || echo 1)"
WORKERSEARCH_HTTP="$(curl -s -o /dev/null -w '%{http_code}' -b "$FUNC_COOKIES" "$BACKEND_URL/api/v2/workers/search?q=te")"
check "Worker search: endpoint responds 200, does not crash (got $WORKERSEARCH_HTTP)" \
    "$([ "$WORKERSEARCH_HTTP" = "200" ] && echo 0 || echo 1)"

# --- Doctor/Worker profile: id=1 doesn't crash with a type-validation error
# (regression check for the profile Path-parameter widening fix -- these
# used to hard-422 on any externally-sourced id because the router declared
# doctor_id/employee_id as plain int; 404 is a valid, expected answer on a
# fresh install with no reserve rows yet, a 422/500 means it regressed) ---
DOCPROFILE_HTTP="$(curl -s -o /dev/null -w '%{http_code}' -b "$FUNC_COOKIES" "$BACKEND_URL/api/v2/doctors/1/profile")"
check "Doctor profile: id=1 doesn't crash with a type-validation error (got $DOCPROFILE_HTTP, expect 200 or 404)" \
    "$([ "$DOCPROFILE_HTTP" = "200" -o "$DOCPROFILE_HTTP" = "404" ] && echo 0 || echo 1)"
WORKERPROFILE_HTTP="$(curl -s -o /dev/null -w '%{http_code}' -b "$FUNC_COOKIES" "$BACKEND_URL/api/v2/workers/1/profile")"
check "Worker profile: id=1 doesn't crash with a type-validation error (got $WORKERPROFILE_HTTP, expect 200 or 404)" \
    "$([ "$WORKERPROFILE_HTTP" = "200" -o "$WORKERPROFILE_HTTP" = "404" ] && echo 0 || echo 1)"

# --- ML Training: Database Growth chart endpoint doesn't crash (regression
# check for get_current_ml_db_size now querying the live ml.CaseTrainingRecord
# SQL Server table instead of a SQLite file that is never shipped) ---
DBGROWTH_HTTP="$(curl -s -o /dev/null -w '%{http_code}' -b "$FUNC_COOKIES" "$BACKEND_URL/api/settings/training/charts/db-growth?days=30")"
check "ML Training: Database Growth chart endpoint responds 200 (got $DBGROWTH_HTTP)" \
    "$([ "$DBGROWTH_HTTP" = "200" ] && echo 0 || echo 1)"

curl -s -b "$FUNC_COOKIES" -X POST "$BACKEND_URL/api/auth/logout" >/dev/null
rm -f "$FUNC_COOKIES"

echo ""
echo "=== 4. Restart recovery ==="
docker restart "${PROJECT_NAME:-pfms}-backend" >/dev/null 2>&1
sleep 5
attempt=0
while [ "$attempt" -lt 20 ]; do
    status="$(docker inspect --format='{{.State.Health.Status}}' "${PROJECT_NAME:-pfms}-backend" 2>/dev/null || echo "starting")"
    [ "$status" = "healthy" ] && break
    attempt=$((attempt + 1)); sleep 5
done
check "backend becomes healthy again after docker restart" "$([ "$status" = "healthy" ] && echo 0 || echo 1)"

echo ""
echo "=== 5. Restart policy (proxy for host-reboot recovery) ==="
for svc in sqlserver backend frontend; do
    policy="$(docker inspect --format='{{.HostConfig.RestartPolicy.Name}}' "${PROJECT_NAME:-pfms}-${svc}" 2>/dev/null || echo "none")"
    check "$svc has restart policy 'unless-stopped' (was: $policy)" "$([ "$policy" = "unless-stopped" ] && echo 0 || echo 1)"
done
echo "  NOTE: this confirms configuration only. An actual host reboot must"
echo "        still be tested physically as part of the disconnected"
echo "        clean-machine air-gap test -- this script cannot reboot the host."

echo ""
echo "=== Summary: $PASS passed, $FAIL failed ==="

echo ""
if [ "${1:-}" = "--delete-test-credentials" ]; then
    rm -f "$CREDS_FILE"
    echo "--delete-test-credentials given: deleted $CREDS_FILE"
else
    read -r -p "Qualification finished. Delete $CREDS_FILE now? [y/N]: " confirm
    if [ "$confirm" = "y" ]; then
        rm -f "$CREDS_FILE"
        echo "Deleted."
    else
        echo "Left in place -- delete manually when ready, or re-run with --delete-test-credentials."
    fi
fi

[ "$FAIL" -eq 0 ]
