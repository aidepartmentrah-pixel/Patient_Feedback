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
# Phase K: Migration Router (legacy case migration endpoints)
from api.routers.migration_router import router as migration_router
# Organization Unit Router (specialized organization unit selection endpoints)
from api.routers.org_unit_router import router as org_unit_router
# Bootstrap Configuration Router (password-protected, no DB auth needed)
from api.routers.config_router import router as config_router
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
# Production: Frontend on IIS (port 80) at http://170.70.32.52, Backend on port 8000
# Development: Frontend on Vite/React dev server, Backend on port 8000

# Production origins
production_origins = [
    "http://170.70.32.52",  # IIS frontend (production)
]

# Development origins
development_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

# Allow override via environment variable
origins_override = os.getenv("ALLOWED_ORIGINS", "")
if origins_override:
    origins = [o.strip() for o in origins_override.split(",") if o.strip()]
else:
    # Combine production and development origins
    origins = production_origins + development_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # All allowed origins
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
# Phase K: Migration Router (legacy case migration endpoints)
app.include_router(migration_router)
# Organization Unit Router (specialized organization unit selection endpoints)
app.include_router(org_unit_router)
# Bootstrap Configuration Router (password-protected, no DB auth needed)
app.include_router(config_router)


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


@app.get("/")
def health_check():
    return {"status": "ok"}



