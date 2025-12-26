from fastapi import FastAPI
from api.routers.dashboard_router import router as dashboard_router
from api.routers.trend_router import router as trend_router
from api.routers.investigation_router import router as investigation_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Incident Manager API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers

app.include_router(dashboard_router)
app.include_router(trend_router)
app.include_router(investigation_router)

@app.get("/")
def health_check():
    return {"status": "ok"}

from fastapi import FastAPI



