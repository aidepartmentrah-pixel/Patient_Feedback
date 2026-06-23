"""
Report Configuration Router
GET /api/settings/report-config  — retrieve institutional report metadata
PUT /api/settings/report-config  — update one or more config keys
"""

from typing import Dict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..db_layer.report_config_db import get_report_config, set_report_config

router = APIRouter(prefix="/api/settings/report-config", tags=["Report Config"])


class ReportConfigUpdateRequest(BaseModel):
    header_title:             str | None = None
    header_subtitle:          str | None = None
    footer_text:              str | None = None
    report_code:              str | None = None
    seasonal_header_title:    str | None = None
    seasonal_header_subtitle: str | None = None
    seasonal_footer_text:     str | None = None
    seasonal_report_code:     str | None = None
    monthly_report_format:    str | None = None   # "classical" | "stylish"


@router.get("")
def read_report_config(current_user: CurrentUser = Depends(get_current_user)):
    """Return all institutional report config values."""
    try:
        return get_report_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("")
def write_report_config(
    body: ReportConfigUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Update any subset of institutional report config values."""
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields provided")
    try:
        set_report_config(updates, user_id=getattr(current_user, "user_id", None))
        return get_report_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
