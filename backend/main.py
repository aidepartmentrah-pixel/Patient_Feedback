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
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Incident Manager API")

# Add CORS middleware - Allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins including file:// for local HTML testing
    allow_credentials=False,  # Must be False when allow_origins is "*"
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

@app.get("/")
def health_check():
    return {"status": "ok"}



