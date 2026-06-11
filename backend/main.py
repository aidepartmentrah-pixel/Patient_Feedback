import os
from fastapi import FastAPI
from api.routers.dashboard_router import router as dashboard_router
from api.routers.trend_router import router as trend_router
from api.routers.investigation_router import router as investigation_router
from api.routers.table_view_router import router as table_view_router
from api.routers.classification_router import router as classification_router
from api.routers.custom_views_router import router as custom_views_router
from api.routers.ner_router import router as ner_router
from api.routers.stt_router import router as stt_router
from api.routers.red_flags_router import router as red_flags_router
from api.routers.never_events_router import router as never_events_router
from api.routers.insert_router import router as insert_router
from api.routers.incident_router import router as incident_router
from api.routers.reference_router import router as reference_router
from api.routers.training_router import router as training_router
from api.routers.patients_router import router as patients_router
from api.routers.reports_router import router as reports_router
from api.routers.settings_router import router as settings_router
from api.routers.doctors_router import router as doctors_router
from api.routers.worker_reporting_router import router as worker_reporting_router
from api.routers.seasonal_export_router import router as seasonal_export_router
from api.routers.person_seasonal_report_router import router as person_seasonal_report_router
from api.routers.follow_up_router import router as follow_up_router
from api.routers.action_items import router as action_items_router
# UPDATED: Using refactored three-type explanation system
from api.routers.explanation_routes_refactored import router as explanation_router
from api.routers.seasonal_comparison_routes import router as seasonal_comparison_router
# Phase 2 RBAC: Session-based authentication router
from api.routers.auth_router import router as auth_router
# Phase 2 RBAC: Example protected endpoints (demonstrates get_current_user dependency)
from api.routers.example_protected_router import router as example_router
# Phase 2 RBAC: Example guarded endpoints (demonstrates authorization guards)
from api.routers.example_guarded_router import router as guarded_router
# NOTE: system_settings_router has a bug (uses 'any' instead of 'Any' in pydantic model)
# from api.routers.system_settings_router import router as system_settings_router
from api.routers.operators_router import router as operators_router
# Phase 3.5: API v2 Workflow Router (unified workflow surface)
from api_v2.routers.workflow_router import router as workflow_router
# Phase 4B: API v2 Insight Router (analytics and KPI endpoints)
from api_v2.routers.insight_router import router as insight_router
# Phase F: API v2 Action Log Router (action item export for follow up page)
from api_v2.routers.action_log_router import router as action_log_router
# Phase B — B-B1: API v2 Doctors Router (doctors endpoints under v2 namespace)
from api_v2.routers.doctors_router import router as doctors_v2_router
# Phase B — B-B2: API v2 Patients Router (patients endpoints under v2 namespace)
from api_v2.routers.patients_router import router as patients_v2_router
# Phase B — B-B3: API v2 Workers Router (worker search endpoint)
from api_v2.routers.workers_router import router as workers_v2_router
# Phase G — G-B7: API v2 Drawer Notes Router (drawer notes CRUD endpoints)
from api_v2.routers.drawer_notes_router import router as drawer_notes_router
# Phase G — G-B8: API v2 Drawer Labels Router (drawer labels management endpoints)
from api_v2.routers.drawer_labels_router import router as drawer_labels_router
# Satisfaction Router (patient satisfaction on cases)
from api_v2.routers.satisfaction_router import router as satisfaction_router
# Phase 5: User Inventory Router (admin-only user management queries)
from api.routers.user_inventory_router import router as user_inventory_router
# Phase 5: Admin Section Router (create sections with admin users)
from api.routers.admin_section_router import router as admin_section_router
# Phase 5: Admin User Credentials Router (TEST ONLY - list users with passwords)
from api.routers.admin_user_credentials_router import router as admin_user_credentials_router
# Phase 5: Admin User Markdown Export Router (TEST ONLY - export credentials as markdown)
from api.routers.admin_user_markdown_router import router as admin_user_markdown_router
# Phase 5: Admin User Management Router (delete users with safety checks)
from api.routers.admin_user_management_router import router as admin_user_management_router
# Phase 5: Admin Section Admin Recreate Router (recreate section admin users)
from api.routers.admin_section_admin_recreate_router import router as admin_section_admin_recreate_router
# Phase B: Settings Users Router (user management CRUD operations)
from api.routers.settings_users_router import router as settings_users_router
# Classification Management Router (Settings — add/rename/freeze classifications)
from api.routers.classification_mgmt_router import router as classification_mgmt_router
# Report Config Router (institutional header/footer/report-code metadata)
from api.routers.report_config_router import router as report_config_router
# Phase K: Migration Router (legacy case migration endpoints)
from api.routers.migration_router import router as migration_router
# Organization Unit Router (specialized organization unit selection endpoints)
from api.routers.org_unit_router import router as org_unit_router
# Bootstrap Configuration Router (password-protected, no DB auth needed)
from api.routers.config_router import router as config_router
# Import Pipeline Router (template download + Excel upload)
from api.routers.import_router import router as import_router
# Policy Metrics Refactor: 4-card org-unit policy management + classification-aware evaluator
from api.routers.policy_router import router as policy_router
# RCA Settings Router (manage categories and suggestions)
from api.routers.rca_settings_router import router as rca_settings_router
# RCA Inbox Router (section inbox RCA read/save)
from api_v2.routers.rca_inbox_router import router as rca_inbox_router
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import core.bootstrap as bootstrap_module
from core.bootstrap import run_bootstrap_check

app = FastAPI(title="Incident Manager API")

# ==================== SESSION MIDDLEWARE (Phase 2: RBAC) ====================
# Add session middleware for authentication
# This enables server-side session storage for user authentication
# SECRET_KEY should be changed in production and stored securely
app.add_middleware(
    SessionMiddleware,
    secret_key="CHANGE_ME_IN_PRODUCTION_USE_SECURE_RANDOM_KEY",  # TODO: Move to environment variable
    session_cookie="incident_manager_session",
    max_age=86400,  # 24 hours session lifetime
    same_site="lax",
    https_only=False  # Set to True in production with HTTPS
)

# ==================== CORS MIDDLEWARE ====================
# CORS is needed when frontend and backend are on different origins (different ports count as different origins)
# CORS origins are now auto-detected from the VM's IP address via deployment_port.py
# This allows the system to work automatically when the VM IP changes.

from core.deployment_port import CORS_ORIGINS, AUTO_DETECTED_IP

# Log the auto-detected configuration at startup
import logging
_cors_logger = logging.getLogger(__name__)
_cors_logger.info(f"Auto-detected VM IP: {AUTO_DETECTED_IP}")
_cors_logger.info(f"CORS origins configured: {CORS_ORIGINS}")

# Allow override via environment variable (optional)
origins_override = os.getenv("ALLOWED_ORIGINS", "")
if origins_override:
    origins = [o.strip() for o in origins_override.split(",") if o.strip()]
else:
    # Use auto-detected CORS origins from deployment_port.py
    origins = CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Auto-detected + configured origins
    allow_credentials=True,  # Enable credentials for session cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(dashboard_router)
app.include_router(trend_router)
app.include_router(investigation_router)
app.include_router(table_view_router)
app.include_router(classification_router)
app.include_router(custom_views_router)
app.include_router(ner_router)
app.include_router(stt_router)
app.include_router(red_flags_router)
app.include_router(never_events_router)
app.include_router(insert_router)
app.include_router(incident_router)
app.include_router(reference_router)
app.include_router(training_router)
app.include_router(patients_router)
app.include_router(reports_router)
app.include_router(settings_router)
app.include_router(doctors_router)
app.include_router(worker_reporting_router)
app.include_router(seasonal_export_router)
app.include_router(person_seasonal_report_router)
app.include_router(follow_up_router)
app.include_router(action_items_router)
app.include_router(explanation_router)
app.include_router(seasonal_comparison_router)
# Phase 2 RBAC: Authentication endpoints
app.include_router(auth_router)
# Phase 2 RBAC: Example protected endpoints
app.include_router(example_router)
# Phase 2 RBAC: Example guarded endpoints
app.include_router(guarded_router)
# NOTE: system_settings_router commented out due to pydantic validation error
# app.include_router(system_settings_router)
app.include_router(operators_router)
# Phase 3.5: API v2 Workflow Router (unified workflow endpoints)
app.include_router(workflow_router)
# Phase 4B: API v2 Insight Router (analytics and KPI endpoints)
app.include_router(insight_router)
# Phase F: API v2 Action Log Router (action item export endpoint)
app.include_router(action_log_router)
# Phase B — B-B1: API v2 Doctors Router (doctors endpoints under v2 namespace)
app.include_router(doctors_v2_router)
# Phase B — B-B2: API v2 Patients Router (patients endpoints under v2 namespace)
app.include_router(patients_v2_router)
# Phase B — B-B3: API v2 Workers Router (worker search endpoint)
app.include_router(workers_v2_router)
# Phase G — G-B7: API v2 Drawer Notes Router (drawer notes CRUD endpoints)
app.include_router(drawer_notes_router)
# Phase G — G-B8: API v2 Drawer Labels Router (drawer label management endpoints)
app.include_router(drawer_labels_router)
# Satisfaction Router (patient satisfaction on cases)
app.include_router(satisfaction_router)
# Phase 5: User Inventory Router (admin-only user management queries)
app.include_router(user_inventory_router)
# Phase 5: Admin Section Router (create sections with admin users)
app.include_router(admin_section_router)
# Phase 5: Admin User Credentials Router (TEST ONLY - list users with passwords)
app.include_router(admin_user_credentials_router)
# Phase 5: Admin User Markdown Export Router (TEST ONLY - export credentials as markdown)
app.include_router(admin_user_markdown_router)
# Phase 5: Admin User Management Router (delete users with safety checks)
app.include_router(admin_user_management_router)
# Phase 5: Admin Section Admin Recreate Router (recreate section admin users)
app.include_router(admin_section_admin_recreate_router)
# Phase B: Settings Users Router (user management CRUD operations)
app.include_router(settings_users_router)
# Classification Management Router (Settings)
app.include_router(classification_mgmt_router)
app.include_router(report_config_router)
# Phase K: Migration Router (legacy case migration endpoints)
app.include_router(migration_router)
# Organization Unit Router (specialized organization unit selection endpoints)
app.include_router(org_unit_router)
# Bootstrap Configuration Router (password-protected, no DB auth needed)
app.include_router(config_router)
app.include_router(import_router)
app.include_router(policy_router)
app.include_router(rca_settings_router)
app.include_router(rca_inbox_router)


# ==================== BOOTSTRAP MIDDLEWARE ====================
# When BOOTSTRAP_MODE is True (DB unreachable), only config and status
# endpoints are allowed. All other routes return 503.

_BOOTSTRAP_ALLOWED_PREFIXES = (
    "/api/config",
    "/api/status",
    "/docs",
    "/redoc",
    "/openapi.json",
)


@app.middleware("http")
async def bootstrap_gate_middleware(request: Request, call_next):
    """Block non-config routes when in bootstrap mode."""
    if bootstrap_module.BOOTSTRAP_MODE:
        path = request.url.path
        # Allow health check, config endpoints, status, and docs
        if path == "/" or path.startswith(_BOOTSTRAP_ALLOWED_PREFIXES):
            return await call_next(request)
        # Block everything else with 503
        return JSONResponse(
            status_code=503,
            content={
                "error": "database_not_configured",
                "message": "Database is not configured or unreachable. Please configure database settings.",
                "config_url": "/config",
            },
        )
    return await call_next(request)


# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def startup_bootstrap_check():
    """Check database connection on startup and set bootstrap mode."""
    import logging
    logger = logging.getLogger("bootstrap")
    logger.info("Running bootstrap database connection check...")
    db_ok = run_bootstrap_check()
    if db_ok:
        logger.info("Database connection OK — running in NORMAL mode")
    else:
        logger.warning("Database connection FAILED — running in BOOTSTRAP mode")
        logger.warning("Only /api/config/* and /api/status endpoints are available")
        logger.warning("Configure database via /config page or /api/config/save endpoint")


@app.on_event("startup")
async def startup_ml_warmup():
    """Pre-load ML models at startup to eliminate cold-start latency."""
    import time
    import logging
    logger = logging.getLogger("ml_warmup")
    
    # Skip warmup if in bootstrap mode (DB not configured)
    if bootstrap_module.BOOTSTRAP_MODE:
        logger.info("[ML WARMUP] Skipped - running in BOOTSTRAP mode")
        return
    
    logger.info("[ML WARMUP] Starting model initialization...")
    start = time.time()
    
    try:
        # Import and run a dummy classification to force all lazy-loaded models into RAM
        from models_directory.Classification_Models.package_models import classify_feedback
        
        # Dummy call - forces loading of:
        # - Tokenizer/Transformer (embedding)
        # - All hierarchical predictors (domain/category/subcategory)
        # - Severity, Harm, Stage models
        # - Classification EN model
        classify_feedback(
            patient_text="Warmup initialization text.",
            text_2="",
            text_3="",
            Print=False
        )
        
        elapsed = time.time() - start
        logger.info(f"[ML WARMUP] Completed in {elapsed:.3f} seconds")
        
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[ML WARMUP] Failed after {elapsed:.3f} seconds: {e}")
        logger.warning("[ML WARMUP] Server will continue - first classification may be slow")


@app.on_event("startup")
async def startup_stt_warmup():
    """Pre-load Whisper STT model at startup to eliminate cold-start latency."""
    import time
    import logging
    logger = logging.getLogger("stt_warmup")
    
    # Skip warmup if in bootstrap mode (DB not configured)
    if bootstrap_module.BOOTSTRAP_MODE:
        logger.info("[STT WARMUP] Skipped - running in BOOTSTRAP mode")
        return
    
    logger.info("[STT WARMUP] Starting Whisper model initialization...")
    start = time.time()
    
    try:
        from models_directory.STT_Models.Faster_Wisper import get_whisper_model
        get_whisper_model()
        
        elapsed = time.time() - start
        logger.info(f"[STT WARMUP] Whisper model loaded successfully in {elapsed:.3f} seconds")
        
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[STT WARMUP] Failed after {elapsed:.3f} seconds: {e}")
        logger.warning("[STT WARMUP] Server will continue - first transcription may be slow")


@app.on_event("startup")
async def startup_db_reconnect_watcher():
    """
    Background task: if the backend started in bootstrap mode (SQL Server was
    unreachable at startup), keep retrying the DB connection every 15 seconds.
    When the connection succeeds, exit bootstrap mode automatically — no manual
    restart required.
    """
    import asyncio
    import logging
    logger = logging.getLogger("db_reconnect")

    if not bootstrap_module.BOOTSTRAP_MODE:
        logger.info("[DB WATCHER] DB connected at startup — watcher idle")
        return

    logger.warning("[DB WATCHER] Started in BOOTSTRAP mode — will retry DB every 15 s")

    async def _reconnect_loop():
        attempt = 0
        while bootstrap_module.BOOTSTRAP_MODE:
            await asyncio.sleep(15)
            attempt += 1
            logger.info(f"[DB WATCHER] Attempt {attempt}: testing DB connection...")
            try:
                result = run_bootstrap_check()
                if result:
                    logger.info("[DB WATCHER] DB connection restored — exiting BOOTSTRAP mode")
                    # Trigger ML warmup now that DB is available
                    try:
                        from models_directory.Classification_Models.package_models import classify_feedback
                        classify_feedback(patient_text="Warmup.", text_2="", text_3="", Print=False)
                        logger.info("[DB WATCHER] ML warmup completed after reconnect")
                    except Exception as ml_e:
                        logger.warning(f"[DB WATCHER] ML warmup skipped: {ml_e}")
                    break
                else:
                    logger.warning(f"[DB WATCHER] Attempt {attempt}: DB still unreachable")
            except Exception as e:
                logger.warning(f"[DB WATCHER] Attempt {attempt}: error during check: {e}")

    asyncio.create_task(_reconnect_loop())


@app.get("/")
def health_check():
    return {"status": "ok"}



