"""
Centralized Hospital Directory API client.

⚠️ CANONICAL SOURCE — DO NOT DUPLICATE ⚠️
Any code that needs to call the external Hospital Directory API must go
through this module. Do not scatter ad-hoc requests/httpx calls to it
elsewhere in the codebase.

Scope: loading the runtime integration config, a /health check (used by the
/config "Hospital Directory API" tab), and PATIENT resource calls
(search_patients / get_patient — Session C1). Doctor/worker resource
calls are NOT implemented here yet — those are separate, later sessions
(C2/C3).

Configuration is stored in SQL Server (APP_ExternalApiSettings, read fresh
on every call — no restart required to apply changes), with the API key
encrypted via core/settings_encryption.py. This is intentionally separate
from the bootstrap database connection settings in db_settings.json.

Patient functions never raise for expected failure modes (disabled
integration, bad URL, timeout, connection refused, 401, 404, TLS failure,
malformed body) — they always return a structured dict with a `status` key,
the same contract check_health() already established. This lets callers
distinguish "external search unavailable" from "zero results found" (see
patients_service / patient_directory_service), instead of collapsing every
failure into an empty list.
"""

import time
import logging
from typing import Optional
from urllib.parse import urlparse

import requests

from api.db_layer import external_api_settings_db
from core.settings_encryption import decrypt_value

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10

# Prefix used to encode an external (Hospital Directory API) patient identity
# into a single opaque string so it can flow through the same "patient_id"
# path/param slots that used to hold a plain APP_RESERVE_PATIENT.PatientAdmissionID
# int. Never stored in the PatientAdmissionID SQL column (see patients_db.py) —
# only used at the service layer to route a lookup to the API instead of SQL
# Server.
#
# NOTE: the Hospital Directory API used to model identity as patient+visit
# (PatientVisit, requiring a visit_id alongside patient_id). That concept was
# dropped from the API — patients are identified by patient_id alone now
# (see the current vendor OpenAPI spec's Patient schema and GET
# /patients/{patient_id}). encode/decode below reflect that current contract.
EXTERNAL_ID_PREFIX = "ext"
EXTERNAL_ID_SEP = "__"


class IntegrationNotConfiguredError(Exception):
    """Raised when the APP_ExternalApiSettings row hasn't been seeded (migration not run)."""


def encode_external_patient_id(patient_id: str) -> str:
    """Pack patient_id into the opaque id HCAT passes around internally."""
    return f"{EXTERNAL_ID_PREFIX}{EXTERNAL_ID_SEP}{patient_id}"


def decode_external_patient_id(value: str) -> Optional[str]:
    """
    Return the external patient_id if `value` is an encoded external id,
    else None (meaning: treat it as a plain reserve PatientAdmissionID).
    """
    if not value or not isinstance(value, str):
        return None
    prefix = f"{EXTERNAL_ID_PREFIX}{EXTERNAL_ID_SEP}"
    if not value.startswith(prefix):
        return None
    rest = value[len(prefix):]
    if not rest:
        return None
    return rest


def normalize_base_url(raw: str):
    """
    Normalize a candidate API base URL.

    Returns (normalized_url, error_message) — exactly one of the two is
    non-empty/non-None. An empty input normalizes to ("", None), meaning
    "not configured" rather than an error.
    """
    value = (raw or "").strip()
    if not value:
        return "", None

    while value.endswith("/"):
        value = value[:-1]

    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        return None, "Base URL must start with http:// or https://."
    if not parsed.netloc:
        return None, "Base URL is missing a host."

    return value, None


def get_active_config() -> dict:
    """
    Load the current Hospital Directory API settings from SQL Server, with
    the API key decrypted server-side only. Never pass the returned dict's
    api_key back to the browser.
    """
    row = external_api_settings_db.get_settings_row()
    if row is None:
        raise IntegrationNotConfiguredError(
            "APP_ExternalApiSettings has no row for 'hospital_directory' — run "
            "sql_scripts/create_external_api_settings_table.sql."
        )
    api_key = decrypt_value(row["api_key_encrypted"]) if row["api_key_encrypted"] else ""
    return {
        "base_url": row["base_url"] or "",
        "api_key": api_key,
        "timeout_seconds": row["timeout_seconds"] or DEFAULT_TIMEOUT_SECONDS,
        "verify_tls": bool(row["verify_tls"]),
        "enabled": bool(row["enabled"]),
    }


def build_auth_headers(api_key: str) -> dict:
    """
    Headers for PROTECTED Hospital Directory API endpoints. NOT needed for
    /health, which the current contract says is unauthenticated — callers
    should not send this on the health check.
    """
    return {"X-API-Key": api_key} if api_key else {}


def check_health(
    base_url: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    verify_tls: Optional[bool] = None,
) -> dict:
    """
    Call GET {base_url}/health.

    If base_url is not given, uses the currently saved config (and its
    saved timeout/verify_tls, unless explicitly overridden). This lets the
    /config page test either an unsaved candidate URL or "whatever is
    currently saved" with the same function.

    Never raises for expected failure modes (bad URL, timeout, connection
    refused, TLS failure, non-200, unreadable body) — always returns a
    structured result, since this directly backs the Test Connection button
    and must never crash the config page. Never includes a stack trace in
    the returned message — only the exception's own message text, bounded.
    """
    start = time.time()

    if base_url is None:
        try:
            cfg = get_active_config()
        except IntegrationNotConfiguredError as e:
            return {"success": False, "status_code": None, "message": str(e), "duration_ms": 0.0}
        base_url = cfg["base_url"]
        if timeout_seconds is None:
            timeout_seconds = cfg["timeout_seconds"]
        if verify_tls is None:
            verify_tls = cfg["verify_tls"]

    timeout_seconds = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
    verify_tls = True if verify_tls is None else verify_tls

    if not base_url:
        return {"success": False, "status_code": None, "message": "No API base URL is configured.", "duration_ms": 0.0}

    normalized, error = normalize_base_url(base_url)
    if error:
        return {"success": False, "status_code": None, "message": error, "duration_ms": 0.0}

    url = f"{normalized}/health"

    try:
        resp = requests.get(url, timeout=timeout_seconds, verify=verify_tls)
        duration_ms = round((time.time() - start) * 1000, 1)

        if resp.status_code == 200:
            return {"success": True, "status_code": 200, "message": "Hospital Directory API is reachable.", "duration_ms": duration_ms}

        return {
            "success": False,
            "status_code": resp.status_code,
            "message": f"Health check returned HTTP {resp.status_code}.",
            "duration_ms": duration_ms,
        }

    except requests.exceptions.SSLError:
        return {
            "success": False, "status_code": None,
            "message": "TLS certificate verification failed. If this is a self-signed/internal certificate, "
                        "disable 'Verify TLS certificate' for this integration.",
            "duration_ms": round((time.time() - start) * 1000, 1),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False, "status_code": None,
            "message": f"Connection timed out after {timeout_seconds}s.",
            "duration_ms": round((time.time() - start) * 1000, 1),
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False, "status_code": None,
            "message": f"Could not reach {normalized} — connection refused or host unreachable.",
            "duration_ms": round((time.time() - start) * 1000, 1),
        }
    except Exception as e:
        logger.error(f"Unexpected error checking Hospital Directory API health: {type(e).__name__}: {e}")
        return {
            "success": False, "status_code": None,
            "message": f"Unexpected error: {str(e)[:300]}",
            "duration_ms": round((time.time() - start) * 1000, 1),
        }


def verify_api_key(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    verify_tls: Optional[bool] = None,
) -> dict:
    """
    Verify the API key actually authenticates, by calling a real protected
    endpoint (GET /doctors, limit=1) — NOT /health, which per the API's own
    contract requires no authentication and therefore can never validate a
    key. A "healthy" /health response says nothing about whether the
    configured key works; this is the only way to actually know.

    Same override-vs-saved-config behavior as check_health() (lets /config
    test an unsaved candidate key before saving), and the same
    never-raises, always-structured-result contract.
    """
    start = time.time()

    if base_url is None or api_key is None:
        try:
            cfg = get_active_config()
        except IntegrationNotConfiguredError as e:
            return {"success": False, "status_code": None, "message": str(e), "duration_ms": 0.0}
        if base_url is None:
            base_url = cfg["base_url"]
        if api_key is None:
            api_key = cfg["api_key"]
        if timeout_seconds is None:
            timeout_seconds = cfg["timeout_seconds"]
        if verify_tls is None:
            verify_tls = cfg["verify_tls"]

    timeout_seconds = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
    verify_tls = True if verify_tls is None else verify_tls

    if not base_url:
        return {"success": False, "status_code": None, "message": "No API base URL is configured.", "duration_ms": 0.0}

    normalized, error = normalize_base_url(base_url)
    if error:
        return {"success": False, "status_code": None, "message": error, "duration_ms": 0.0}

    if not api_key:
        return {
            "success": False, "status_code": None,
            "message": "No API key is configured — cannot verify authentication.",
            "duration_ms": 0.0,
        }

    url = f"{normalized}/doctors"
    headers = build_auth_headers(api_key)

    try:
        resp = requests.get(url, params={"limit": 1}, headers=headers, timeout=timeout_seconds, verify=verify_tls)
        duration_ms = round((time.time() - start) * 1000, 1)

        if resp.status_code == 200:
            return {"success": True, "status_code": 200, "message": "API key verified — authenticated successfully.", "duration_ms": duration_ms}
        if resp.status_code == 401:
            return {
                "success": False, "status_code": 401,
                "message": "API key rejected (401 Unauthorized) — this key does not authenticate.",
                "duration_ms": duration_ms,
            }

        return {
            "success": False, "status_code": resp.status_code,
            "message": f"Unexpected response verifying the key: HTTP {resp.status_code}.",
            "duration_ms": duration_ms,
        }

    except requests.exceptions.SSLError:
        return {
            "success": False, "status_code": None,
            "message": "TLS certificate verification failed. If this is a self-signed/internal certificate, "
                        "disable 'Verify TLS certificate' for this integration.",
            "duration_ms": round((time.time() - start) * 1000, 1),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False, "status_code": None,
            "message": f"Connection timed out after {timeout_seconds}s.",
            "duration_ms": round((time.time() - start) * 1000, 1),
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False, "status_code": None,
            "message": f"Could not reach {normalized} — connection refused or host unreachable.",
            "duration_ms": round((time.time() - start) * 1000, 1),
        }
    except Exception as e:
        logger.error(f"Unexpected error verifying Hospital Directory API key: {type(e).__name__}: {e}")
        return {
            "success": False, "status_code": None,
            "message": f"Unexpected error: {str(e)[:300]}",
            "duration_ms": round((time.time() - start) * 1000, 1),
        }


def verify_integration(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    verify_tls: Optional[bool] = None,
) -> dict:
    """
    The real "Test Connection" check: is the server reachable AND does the
    API key actually authenticate. Replaces relying on check_health() alone,
    which can only ever prove the first half — /health needs no auth, so a
    healthy response is not evidence the key works.

    Skips the auth check entirely if the health check already failed (no
    point authenticating against a server that isn't there).
    """
    health_result = check_health(base_url, timeout_seconds, verify_tls)

    if not health_result["success"]:
        return {
            "success": False,
            "health": health_result,
            "auth": {"success": False, "status_code": None, "message": "Skipped — server unreachable.", "duration_ms": 0.0},
            "message": f"Server unreachable: {health_result['message']}",
        }

    auth_result = verify_api_key(base_url, api_key, timeout_seconds, verify_tls)
    overall_success = health_result["success"] and auth_result["success"]

    message = (
        "Server reachable and API key verified."
        if overall_success else
        f"Server reachable, but {auth_result['message']}"
    )

    return {
        "success": overall_success,
        "health": health_result,
        "auth": auth_result,
        "message": message,
    }


# =============================================================================
# PATIENT RESOURCE CALLS (Session C1)
# =============================================================================
#
# Both functions below share one status vocabulary so callers (patients_db /
# patient_directory_service) can render a distinct, honest message for each
# failure mode instead of collapsing everything into "not found":
#
#   "ok"                — call succeeded, data attached
#   "disabled"           — integration not configured or Enabled=0 (not an error)
#   "unauthorized"        — 401 from the API (bad/expired key)
#   "not_found"           — 404 (get_patient only)
#   "bad_request"          — 400 (e.g. no search criteria given)
#   "timeout" / "connection_error" / "ssl_error" — network-level failure
#   "http_error"           — any other non-2xx status
#   "invalid_response"      — 2xx but body wasn't parseable JSON in the expected shape

def _get_client_config():
    """
    Load + validate the active integration config for a resource call.
    Returns (config_dict_or_None, disabled_result_or_None).

    There is no separate "enabled" on/off switch — the Hospital Directory
    API isn't an optional feature the app can run without once patient/
    doctor/worker data comes from it, so a manual disable toggle isn't a
    valid operating state. "Configured" (a Base URL is set) is the only
    signal: if it's missing, that's a genuine "not set up yet" state, not a
    deliberate disable.
    """
    try:
        cfg = get_active_config()
    except IntegrationNotConfiguredError as e:
        return None, {"status": "disabled", "message": str(e)}

    normalized, error = normalize_base_url(cfg["base_url"])
    if error or not normalized:
        return None, {"status": "disabled", "message": error or "No Hospital Directory API base URL is configured."}

    cfg["base_url"] = normalized
    return cfg, None


def _get(path: str, params: dict) -> dict:
    """
    Shared GET helper for authenticated patient resource endpoints. Never
    raises — always returns a dict with at least {"status", "message"}.
    """
    cfg, disabled = _get_client_config()
    if disabled:
        return disabled

    url = f"{cfg['base_url']}{path}"
    headers = build_auth_headers(cfg["api_key"])
    timeout_seconds = cfg["timeout_seconds"]

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout_seconds, verify=cfg["verify_tls"])
    except requests.exceptions.SSLError:
        return {
            "status": "ssl_error",
            "message": "TLS certificate verification failed. If this is a self-signed/internal certificate, "
                        "disable 'Verify TLS certificate' for this integration.",
        }
    except requests.exceptions.Timeout:
        return {"status": "timeout", "message": f"Hospital Directory API request timed out after {timeout_seconds}s."}
    except requests.exceptions.ConnectionError:
        return {"status": "connection_error", "message": f"Could not reach the Hospital Directory API — connection refused or host unreachable."}
    except Exception as e:
        logger.error(f"Unexpected error calling Hospital Directory API {path}: {type(e).__name__}: {e}")
        return {"status": "http_error", "message": f"Unexpected error: {str(e)[:300]}"}

    if resp.status_code == 401:
        return {"status": "unauthorized", "message": "Hospital Directory API rejected the configured API key (401 Unauthorized)."}
    if resp.status_code == 404:
        return {"status": "not_found", "message": "Not found."}
    if resp.status_code == 400:
        try:
            detail = resp.json().get("message", resp.text[:300])
        except Exception:
            detail = resp.text[:300]
        return {"status": "bad_request", "message": detail}
    if resp.status_code >= 400:
        return {"status": "http_error", "message": f"Hospital Directory API returned HTTP {resp.status_code}."}

    try:
        data = resp.json()
    except Exception:
        return {"status": "invalid_response", "message": "Hospital Directory API returned a non-JSON response."}

    return {"status": "ok", "message": None, "data": data}


def search_patients(
    q: Optional[str] = None,
    patient_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """
    GET {base_url}/patients?q=&patient_id=&limit=&offset=

    At least one of q/patient_id must be given (matches the API's own
    MISSING_SEARCH_CRITERIA validation) — callers should check before
    calling to avoid an avoidable "bad_request" result.

    Returns {"status", "message", "items": [...], "total": int} on "ok";
    "items"/"total" are omitted for non-"ok" statuses.
    """
    params = {k: v for k, v in {
        "q": q, "patient_id": patient_id,
        "limit": max(1, min(limit, 500)), "offset": max(0, offset),
    }.items() if v is not None}

    result = _get("/patients", params)
    if result["status"] != "ok":
        return result

    data = result["data"]
    if not isinstance(data, dict) or "items" not in data:
        return {"status": "invalid_response", "message": "Hospital Directory API /patients response was missing 'items'."}

    return {
        "status": "ok", "message": None,
        "items": data.get("items", []),
        "total": data.get("total", len(data.get("items", []))),
    }


def get_patient(patient_id: str) -> dict:
    """
    GET {base_url}/patients/{patient_id}

    Returns {"status", "message", "patient": {...}} on "ok".
    """
    from urllib.parse import quote

    path = f"/patients/{quote(str(patient_id), safe='')}"
    result = _get(path, params={})
    if result["status"] != "ok":
        return result

    return {"status": "ok", "message": None, "patient": result["data"]}
