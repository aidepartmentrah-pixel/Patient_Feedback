# Hospital Directory API Integration — Middle-Name Search & General Usage Reference

**Purpose of this document**: how this app (internally called **HCAT** in its own source comments) talks to the hospital's external patient/doctor/worker directory system, how to tell its two environments apart ("the malin thing"), and specifically how the middle-name search-assist feature uses it. Every claim is backed by a specific file in this repo — paths given throughout — so this can be lifted into a new project accurately, not from memory.

---

## 1. The two names for the same thing — clearing up "malin"

There are **two completely separate systems** in play, both called "the Hospital Directory API" in code, and it's easy to conflate them:

| | Real production system | Local dev mock |
|---|---|---|
| **What it is** | The hospital's own internal patient/doctor/staff directory server — nicknamed **"malin"** by hostname | A separate small repo (`RESTful-API-Integration`) that re-implements the same HTTP contract for offline development |
| **Address** | `http://malin:8080/his-general/api/directory/v1` — `malin` resolves to a fixed internal IP, **`150.50.1.34`**, on the hospital's own network | `http://localhost:6000` (or the container name `restful-api-integration-api-1` from inside another container), published on host port **6000** |
| **How it's reached from this app's Docker container** | The container's own network namespace has **no route** to the hospital's internal DNS, so `malin` would not resolve at all without help — this repo's [`docker-compose.yml:110-111`](../docker-compose.yml) adds an explicit `extra_hosts: ["malin:150.50.1.34"]` entry. This was confirmed necessary via live testing: the **host machine** could reach `malin` once its own `/etc/hosts` had the entry, but the backend **container** — a separate network namespace — still could not, until the same override was added to the container's own hosts file via Compose. | Just a normal container/localhost port; no DNS trickery needed. |
| **X-API-Key value** | `dev-secret-key` (dev/test key for the real server) | `change_me` (baked into the mock's own `API_KEY` env var — confirmed via `docker exec restful-api-integration-api-1 env`) |
| **These two keys are NOT interchangeable.** Using the mock's key against the real server (or vice versa) produces a silent `401 Unauthorized` from every call — which surfaces in this app as `external_status: not_searched` / zero external results, easily mistaken for a frontend bug. | | |

**How to tell which one you're pointed at, right now**: don't guess from the key or the hostname alone — read the actual configured value. See §3 for where it's stored and §4 for how to check/test it live.

---

## 2. What this integration is for

HCAT keeps its **own local patient records** ("reserve" patients, table `APP_RESERVE_PATIENT`) but also needs to search/display patients, doctors, and workers that live only in the hospital's own directory system. Rather than mirroring that data into SQL Server, HCAT calls the Hospital Directory API live, on every search, and **merges** the two result sets into one normalized shape the rest of the app doesn't need to know the origin of. See [`backend/api/services/patient_directory_service.py`](../backend/api/services/patient_directory_service.py) module docstring.

Two records that look identical to a user (reserve vs. external) are told apart internally by an **opaque id encoding**, never by a separate flag threaded through every function:
- A plain reserve id (e.g. `"482"`) → resolves against `APP_RESERVE_PATIENT`.
- An external id is packed as `"ext__<patient_id>"` (`encode_external_patient_id()` / `decode_external_patient_id()` in [`backend/core/hospital_directory_client.py`](../backend/core/hospital_directory_client.py)) → resolves against the Hospital Directory API's `GET /patients/{id}`.

This lets both kinds of "patient_id" flow through the exact same function signatures, URL path params, and frontend selection code without a parallel set of "external" variants everywhere.

**A hard product requirement baked into the design**: if the external API is down/misconfigured, **local (reserve) results must still work** — a network failure never makes local data disappear. Every merge function returns reserve results unconditionally and reports the external half's outcome separately via an `external_status`/`external_message` pair (see §5's status vocabulary), so the UI can show "external search unavailable" distinctly from "zero results found" instead of collapsing both into the same empty list.

---

## 3. Where the connection is configured

**Not** an environment variable and **not** `db_settings.json` (that file is only for the app's own bootstrap database connection). Instead:

- **`APP_ExternalApiSettings`** — one SQL Server table, one row (`IntegrationName = 'hospital_directory'`), created by [`backend/sql_scripts/create_external_api_settings_table.sql`](../backend/sql_scripts/create_external_api_settings_table.sql):

  | Column | Meaning |
  |---|---|
  | `BaseUrl` | e.g. `http://malin:8080/his-general/api/directory/v1` or the mock's URL |
  | `ApiKeyEncrypted` | the `X-API-Key` value, Fernet-encrypted at rest |
  | `TimeoutSeconds` | default 10 |
  | `VerifyTls` | default `1` (on) |
  | `Enabled` | historical column — see note below |
  | `LastTestStatus` / `LastTestMessage` / `LastTestAt` | result of the most recent "Test Connection" click, persisted so it survives a page reload |

- **Read fresh from the database on every single call** — no in-process caching, no restart required. Changing the base URL or key in Settings takes effect on the very next search.
- **The API key is encrypted with [Fernet](https://cryptography.io/en/latest/fernet/)**, via [`backend/core/settings_encryption.py`](../backend/core/settings_encryption.py). The encryption key itself lives **only** in the `SETTINGS_ENCRYPTION_KEY` process environment variable — never in the database, never in this table. It's set per-environment in `docker-compose.yml` (`SETTINGS_ENCRYPTION_KEY: ${SETTINGS_ENCRYPTION_KEY}`) and generated fresh per deployment (`.env.example`/`.env.offline.template` both ship a placeholder that must be replaced with a real generated Fernet key before first use — never reuse the example value across environments).
- **Important practical consequence**: an encrypted `ApiKeyEncrypted` blob is only decryptable by the same `SETTINGS_ENCRYPTION_KEY` that encrypted it. Copying a database backup (or just this one row) between two environments **without** also matching their `SETTINGS_ENCRYPTION_KEY` values leaves the key undecryptable — `GET /api/config/external-api` handles this explicitly by returning `api_key_error` instead of a 500, rather than crashing the settings page.
- Despite the `Enabled` column existing, there is **no runtime on/off toggle that makes sense to use** — per [`hospital_directory_client.py`](../backend/core/hospital_directory_client.py) `_get_client_config()`'s own comment, this isn't an optional feature the app can run without once patient/doctor/worker data depends on it; "configured" (a base URL is set) is treated as the only meaningful signal, not a deliberate disable switch.

---

## 4. Configuring and testing it (admin flow)

All of this lives behind `/api/config/external-api/*` in [`backend/api/routers/config_router.py`](../backend/api/routers/config_router.py), gated by an **`X-Config-Password` header** (a separate password from normal user login — checked in `_require_config_password()`, with IP-based lockout after repeated failed attempts). This is the same gate used for the database-connection settings and the middle-name candidate-set admin CRUD (§7).

| Endpoint | Purpose |
|---|---|
| `GET /api/config/external-api` | Current settings, API key **masked** (e.g. `••••••1234`) |
| `POST /api/config/external-api/save` | Update base URL / key / timeout / TLS verification / enabled flag. Applies immediately, no restart. A masked placeholder submitted unchanged is detected and left alone (`_is_masked_password`) so re-saving the form without retyping the key doesn't overwrite it with literal asterisks. |
| `POST /api/config/external-api/reveal-key` | Returns the real decrypted key — an explicit, audit-logged action (`logger.warning("AUDIT: Hospital Directory API key revealed via /config ...")`), never returned by the plain `GET`. |
| `POST /api/config/external-api/test-connection` | The real health+auth test — see below. Persists its result as the row's `LastTestStatus`/`LastTestMessage`. |

**Why "Test Connection" is two checks, not one**: `GET /health` on this API is (by contract) **unauthenticated** — a healthy response only proves the server is reachable, it proves nothing about whether the configured key actually works. So `verify_integration()` in `hospital_directory_client.py` runs `check_health()` first, and **only if that succeeds**, separately calls `GET /doctors?limit=1` *with* the configured `X-API-Key` header to prove the key itself authenticates (a `401` there means "reachable, but the key is wrong" — a distinct, correctly-labeled outcome from "server unreachable"). Skips the auth call entirely if the server isn't even reachable, so a down server never gets reported confusingly as a bad key.

**Practical troubleshooting** (this happened for real on 2026-09-10, see project memory): if guess-mode/structured patient search suddenly returns zero external results with **no visible frontend error**, that's the signature of a `401` from every candidate call being interpreted as "not_searched" rather than a crash. Before assuming the search code is broken:
1. Check the actual saved row (`external_api_settings_db.get_settings_row()`) — compare `base_url`/`enabled` against a known-good environment.
2. Confirm the `api_key` matches the **target** environment (§1's table) — not copied by analogy from a different one.
3. Remember an encrypted key copied from one container will not decrypt in another with a different `SETTINGS_ENCRYPTION_KEY` — re-encrypt with `core.settings_encryption.encrypt_value()` from *inside* the target container if restoring a row manually.

---

## 5. The canonical client module — don't duplicate it

**[`backend/core/hospital_directory_client.py`](../backend/core/hospital_directory_client.py)** is explicitly marked as the **only** place allowed to make HTTP calls to this API — its own docstring: *"Any code that needs to call the external Hospital Directory API must go through this module. Do not scatter ad-hoc requests/httpx calls to it elsewhere in the codebase."* This is the single most important pattern to copy into a new project.

**Contract every function in this module follows**: never raise for an expected failure mode. Every call — health check, key verification, patient/doctor/worker search — returns a **structured dict with a `status` key**, using one shared vocabulary so every caller can render a distinct, honest message instead of collapsing everything into "no results":

| `status` | Meaning |
|---|---|
| `ok` | Call succeeded, data attached |
| `disabled` | Integration not configured (no base URL saved yet) — not treated as an error |
| `unauthorized` | `401` — bad/expired API key |
| `not_found` | `404` (single-resource lookups only) |
| `bad_request` | `400`/`422` — e.g. a search with no valid criteria |
| `timeout` / `connection_error` / `ssl_error` | Network-level failure, each reported distinctly |
| `http_error` | Any other non-2xx |
| `invalid_response` | `2xx` but the body wasn't parseable/didn't have the expected shape |

Every non-2xx response is assumed to carry `{"error": "<CODE>", "message": "<text>"}` (the vendor's own v1.1 contract guarantees this on every non-2xx status), and `_parse_error_message()` extracts `message` for display.

---

## 6. The external API's actual resource contract (v1.1)

All paths below are relative to the configured `base_url` (e.g. `.../his-general/api/directory/v1`). Auth: header `X-API-Key: <key>` on every endpoint **except** `/health`.

| Method & path | Auth | Notes |
|---|---|---|
| `GET /health` | No | Used only to prove reachability — never proves the key works |
| `GET /patients` | Yes | **Exactly one search mode per call**: either `patient_id` alone, **or** `first_name`+`father_name`+`last_name` together as three separate params (each matched case-insensitively as a substring against only its own column, Arabic or English) — any other combination gets a `422 VALIDATION_ERROR` from the API itself. A free-text `q` param is also accepted (the one mode confirmed working against the real production server historically) but is **not** part of the documented v1.1 structured contract — never combine `q` with the structured fields in one call. `limit`/`offset` supported, `limit` capped server-side-friendly at 500 by this client. |
| `GET /patients/{patient_id}` | Yes | Single patient by id. `father_name` is **search-only** — it is never returned in a patient's result object. |
| `GET /doctors` | Yes | `q`, `active_only`, `limit`, `offset` — all optional; returns a default paginated "active doctors" list with no criteria at all. |
| `GET /doctors/{doctor_id}` | Yes | |
| `GET /workers` | Yes | `limit`/`offset` only — confirmed via live testing that `q`/`active_only` are **not** accepted here (unlike `/doctors`); it always returns active-only workers. Callers needing filtering must fetch-all and filter client-side. |
| `GET /workers/{employee_id}` | Yes | |

**Naming mismatch to know about**: this app's own UI/database column is called **"middle name"**; the vendor's API calls the identical concept **`father_name`**. The translation happens at exactly one boundary — `patient_directory_service.search_patients_structured()` — and nowhere else; don't let `father_name` leak into HCAT's own UI/DB terminology, and don't let `middle_name` leak into a raw API call.

**Identity model — patients are visit-scoped, not person-scoped.** The vendor's schema used to model identity as patient+visit; that concept was dropped from the current API (patients are identified by `patient_id` alone now), **but the underlying data still isn't strictly one-record-per-real-person**: the same real person can appear more than once under separate `patient_id`s (e.g. repeat admissions, each entered slightly differently). Two different dedup strategies exist in this codebase for two different situations, and conflating them was an actual shipped bug (fixed 2026-09-10):
- **`_dedupe_external_items()`** (used by the structured 3-field search, all fields known) — dedupes by **`(full_name, birth_date)`** pair, because the vendor's own v1.1 test fixtures deliberately require two *different* real people who share a name but differ in birth date to both be returned, never merged.
- **The missing-middle-name guess loop** (§7) — dedupes by **`full_name` alone**, because the frontend's selection handler only ever commits the clicked row's name string (never `patient_admission_id`/`birth_date`), so multiple rows sharing an identical full name are, from the user's point of view, one indistinguishable choice — showing 20 near-identical rows for one real person is noise, not options.

**Lesson for a new project**: never assume a "unique id" field from a partner API actually identifies a person the way your own product needs it to — check what your UI actually lets a human distinguish before picking a dedup key.

---

## 7. The middle-name search-assist feature specifically

### 7.1 Two paths, one field
The Insert-Record patient-name UI ([`PatientNameSearch.jsx`](../../Front_End_Feedback_Analysis/src/components/incident/PatientNameSearch.jsx) in the frontend repo) has three separate inputs — First / Middle (optional) / Last — because the real API 422s on anything other than all three structured fields together.

- **Middle name typed** → a normal debounced structured search, `GET /api/records/search/patients?first_name=&middle_name=&last_name=` → `patient_directory_service.search_patients_structured()`.
- **Middle name left blank** → a single explicit **"Find possible matches"** button click triggers the **guess loop**: try every name in a curated candidate list as if it were the real middle name, and combine every hit.

### 7.2 The candidate list — file-based, hot-swappable, no database
[`backend/api/services/middle_name_sets_service.py`](../backend/api/services/middle_name_sets_service.py) manages named, switchable sets of candidate names as plain JSON files:
```
backend/config/middle_name_sets/
    thirty_names.json   {"display_name": "...", "names": [...]}
    _active.json         {"active_set": "thirty_names"}
```
- Lives inside the same persistent config volume as `db_settings.json` — survives container replacement, editable without a rebuild.
- Set ids are slugified display names (`[a-z0-9_]` only) used directly as filenames — validated before ever touching the filesystem, specifically to close off path traversal from a crafted identifier.
- The default seeded set (`thirty_names`, "Starting set (32 names)") is a curated list of common Lebanese Shia male given names (محمد، علي، حسين... — 32 entries despite the id's "thirty"), ordered most-common-first — order matters because it's tried in that order.
- Admin CRUD (`/api/config/middle-name-sets/*`, password-gated) lets creating/editing/switching sets without touching code; the search-time read (`GET /api/patient-search/middle-name-candidates`, no password — same trust level as any other search-support endpoint) always reads the active set fresh off disk, so an admin's change is live on the very next search.

### 7.3 The guess loop itself
`_iter_missing_middle_name_candidates()` in `patient_directory_service.py`:
1. Loads the active candidate set (§7.2).
2. Dispatches **all** candidates at once to a `ThreadPoolExecutor(max_workers=8)` — a deliberately conservative concurrency cap since the vendor's real tolerance for concurrent requests has never been load-tested.
3. **Never stops early.** An earlier version stopped at the first candidate with any hit, as a latency optimization — real testing proved that's wrong: a first+last name pair can legitimately match multiple different real people who differ only in father's name, and stopping early silently hid the second person rather than just costing an extra search. Every candidate is always tried.
4. Combines every candidate's hits, deduped by `full_name` alone (§6's second bullet).
5. Yields progress events as each candidate's real network call completes (not in list order — whatever finishes first), so both a plain blocking endpoint and a streaming endpoint can share exactly one implementation.

**This is real load against the live external API on one explicit user action** (a single button click), never fired automatically per keystroke — deliberately backend-owned so the concurrency cap/dedup logic is enforced in one place regardless of what UI calls it.

### 7.4 Two ways to consume the loop
- **`GET /api/records/search/patients/missing-middle-name`** — blocking, returns the final combined list plus `tried` (how many candidates were actually attempted).
- **`GET /api/records/search/patients/missing-middle-name/stream`** — Server-Sent Events, same loop, reported live:
  ```
  event: match           {index, total, patient}
  event: candidate_done  {index, total, candidate, new_match_count, external_status, external_message}
  event: done            {count, tried, external_status, external_message}   -- always last
  event: error           {message}                                          -- only on an unexpected exception
  ```
  Auth is the same session-cookie dependency as every other endpoint — a browser's native `EventSource` sends cookies automatically same-origin, no extra handling. The frontend consumes this via a small wrapper (`streamPatientsMissingMiddleName()` in `src/api/insertRecord.js`) that opens an `EventSource` and forwards named events to callbacks.

  **⚠️ One inconsistency worth knowing if you extend this**: the frontend wrapper currently listens for an event named `"trying"`, but the current backend implementation only ever emits `match` / `candidate_done` / `done` / `error` (the `"trying"` event was explicitly removed when the loop became concurrent — see the router's own docstring: *"There is no more 'trying' event"*). In practice this doesn't break the feature (matches still arrive via `match`, completion still arrives via `done`), but the per-candidate progress bar callback (`onTrying`) is dead code as currently wired. **If porting this SSE pattern, keep the event-name contract in one shared place (or a shared constants file) so frontend and backend can't drift like this.**

### 7.5 Why the progress bar shows no candidate names
Deliberate: the candidate-guessing mechanism (trying a curated list of common names behind the scenes) is treated as an internal implementation detail, not something to surface to the person filling out the form — the UI shows only a bare fraction-complete fill, never "trying محمد... trying علي...".

---

## 8. Full internal endpoint reference (HCAT's own API, not the vendor's)

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /api/records/search/patients?q=` | session login | Free-text merged search (reserve + external) |
| `GET /api/records/search/patients?first_name=&middle_name=&last_name=` | session login | Structured merged search, all fields known |
| `GET /api/records/search/patients/missing-middle-name?first_name=&last_name=` | session login | Guess-loop, blocking |
| `GET /api/records/search/patients/missing-middle-name/stream?first_name=&last_name=` | session login (cookie) | Guess-loop, SSE |
| `GET /api/patient-search/middle-name-candidates` | session login (no config password) | Active candidate set, for the chip/guess UI |
| `GET/POST/PUT/DELETE /api/config/middle-name-sets*` | `X-Config-Password` | Admin CRUD over candidate name sets |
| `GET /api/config/external-api` | `X-Config-Password` | Read current Hospital Directory API connection settings (key masked) |
| `POST /api/config/external-api/save` | `X-Config-Password` | Update connection settings, effective immediately |
| `POST /api/config/external-api/reveal-key` | `X-Config-Password` | Return the real decrypted key (audit-logged) |
| `POST /api/config/external-api/test-connection` | `X-Config-Password` | Health + auth test, persists last-test result |

---

## 9. Technology / patterns used

- **HTTP client**: plain `requests` (synchronous), not `httpx`/`aiohttp` — calls happen inside a thread pool (`ThreadPoolExecutor`) for the concurrent guess loop rather than via native async I/O.
- **Secrets**: `cryptography`'s `Fernet` symmetric encryption, key supplied only via process environment (`SETTINGS_ENCRYPTION_KEY`), never persisted alongside the ciphertext.
- **Runtime config**: a single-row SQL Server table, read fresh per-request — the same "no restart to apply a config change" convention this app also uses for the middle-name candidate sets (files) and the database settings.
- **Streaming**: FastAPI `StreamingResponse` with `media_type="text/event-stream"`, hand-rolled `event: ...\ndata: ...\n\n` framing (no SSE library) — headers `Cache-Control: no-cache`, `X-Accel-Buffering: no`, `Connection: keep-alive` to defeat proxy buffering.
- **Concurrency**: `concurrent.futures.ThreadPoolExecutor` + `as_completed`, capped worker count — appropriate here because the work is I/O-bound (waiting on HTTP calls to a partner server), not CPU-bound.
- **DNS pinning for a partner host only reachable inside a private network**: Docker Compose `extra_hosts`, not a custom resolver or hardcoded IP inline in application code — keeps the IP address as infrastructure config, not source code.

---

## 10. File map

| Concern | File |
|---|---|
| Canonical external API client (never duplicate) | `backend/core/hospital_directory_client.py` |
| Runtime config DB layer | `backend/api/db_layer/external_api_settings_db.py` |
| Secret encryption | `backend/core/settings_encryption.py` |
| Config table DDL/seed | `backend/sql_scripts/create_external_api_settings_table.sql` |
| Admin settings + test-connection endpoints | `backend/api/routers/config_router.py` |
| Merge logic (reserve + external → one shape) | `backend/api/services/patient_directory_service.py` |
| Reserve-side (local DB) patient search | `backend/api/db_layer/patients_db.py` |
| Candidate middle-name sets (file-based) | `backend/api/services/middle_name_sets_service.py`, `backend/api/routers/middle_name_sets_router.py` |
| SSE streaming endpoint | `backend/api/routers/insert_router.py` (search around `/search/patients/missing-middle-name`) |
| Docker DNS override for the real "malin" host | `docker-compose.yml` (`extra_hosts`) |
| Frontend 3-field search UI | `Front_End_Feedback_Analysis/src/components/incident/PatientNameSearch.jsx` |
| Frontend API client (incl. SSE wrapper) | `Front_End_Feedback_Analysis/src/api/insertRecord.js` |

---

## 11. Design lessons worth carrying into a new project

1. **One canonical client module per external API, enforced by convention (and ideally by lint/review), never scattered ad-hoc HTTP calls.**
2. **A shared status vocabulary that never raises for expected failures** — lets every caller give an honest, specific message ("server unreachable" vs "key rejected" vs "not configured") instead of a generic error or a silently-empty result.
3. **Store partner-API config (URL, encrypted key, timeout, TLS flag) in your own database, read fresh per request — not in a `.env` file** — so an admin can fix a bad key without a deployment/restart.
4. **Health checks and auth checks are different questions.** If a health endpoint doesn't require auth, it can never prove a configured key is valid — test both, explicitly, and label failures accordingly.
5. **A partner API's "unique id" may not mean what your product needs it to mean.** Confirm what your own UI actually lets a user distinguish before choosing a dedup/identity key — don't assume person-scoped just because there's an id field.
6. **When bridging naming differences between your product and a vendor's contract** (here: "middle name" vs `father_name`), translate at exactly one boundary function, and never let the vendor's term leak into your own UI/DB, or vice versa.
7. **A partner host only reachable on a private/internal network belongs in infrastructure config (Docker `extra_hosts`, DNS), not hardcoded into application code.**
8. **Never let an external integration's failure take down local functionality that doesn't need it** — always merge/degrade gracefully and say so explicitly, rather than propagating a partner outage into a blank screen.
