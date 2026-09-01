"""
Middle-Name Candidate Sets Router

Two trust levels, deliberately different:
  - GET /api/patient-search/middle-name-candidates: no config password --
    this is what the patient-search chip UI calls on every search, for
    every normal app user, same trust level as any other search-support
    endpoint (see patients_router.py, which has no password gate either).
  - Everything under /api/config/middle-name-sets/*: CRUD/admin surface,
    gated by the same X-Config-Password check as the rest of
    config_router.py's /api/config/* endpoints (Hospital Directory API
    settings, database settings) -- same trust level, same header.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from api.routers.config_router import _require_config_password
from api.services import middle_name_sets_service as svc

router = APIRouter(tags=["Middle Name Assist"])


class CreateSetRequest(BaseModel):
    display_name: str
    names: Optional[List[str]] = None


class UpdateSetRequest(BaseModel):
    display_name: Optional[str] = None
    names: Optional[List[str]] = None


class ActivateSetRequest(BaseModel):
    set_id: str


class AddNameRequest(BaseModel):
    name: str


class UpdateNameRequest(BaseModel):
    new_name: str


def _error(e: svc.MiddleNameSetsError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(e))


# ==================== SEARCH-TIME (no password) ====================

@router.get("/api/patient-search/middle-name-candidates")
async def get_middle_name_candidates():
    """
    Active set's candidate names, for the patient-search chip UI. Read
    fresh from disk on every call, so a Settings-tab change is live on the
    next search immediately -- no restart required.
    """
    try:
        active = svc.get_active_set()
    except svc.MiddleNameSetsError as e:
        raise _error(e)
    return {"display_name": active["display_name"], "names": active["names"]}


# ==================== ADMIN CRUD (X-Config-Password required) ====================

@router.get("/api/config/middle-name-sets")
async def list_middle_name_sets(request: Request):
    _require_config_password(request)
    return {"sets": svc.list_sets()}


@router.get("/api/config/middle-name-sets/{set_id}")
async def get_middle_name_set(set_id: str, request: Request):
    _require_config_password(request)
    try:
        return svc.get_set(set_id)
    except svc.MiddleNameSetsError as e:
        raise _error(e)


@router.post("/api/config/middle-name-sets")
async def create_middle_name_set(body: CreateSetRequest, request: Request):
    _require_config_password(request)
    try:
        return svc.create_set(body.display_name, body.names)
    except svc.MiddleNameSetsError as e:
        raise _error(e)


@router.put("/api/config/middle-name-sets/{set_id}")
async def update_middle_name_set(set_id: str, body: UpdateSetRequest, request: Request):
    _require_config_password(request)
    try:
        return svc.update_set(set_id, body.display_name, body.names)
    except svc.MiddleNameSetsError as e:
        raise _error(e)


@router.delete("/api/config/middle-name-sets/{set_id}")
async def delete_middle_name_set(set_id: str, request: Request):
    _require_config_password(request)
    try:
        svc.delete_set(set_id)
    except svc.MiddleNameSetsError as e:
        raise _error(e)
    return {"deleted": True}


@router.post("/api/config/middle-name-sets/active")
async def activate_middle_name_set(body: ActivateSetRequest, request: Request):
    _require_config_password(request)
    try:
        svc.set_active(body.set_id)
    except svc.MiddleNameSetsError as e:
        raise _error(e)
    return {"active_set": body.set_id}


@router.post("/api/config/middle-name-sets/{set_id}/names")
async def add_middle_name(set_id: str, body: AddNameRequest, request: Request):
    _require_config_password(request)
    try:
        return svc.add_name(set_id, body.name)
    except svc.MiddleNameSetsError as e:
        raise _error(e)


@router.put("/api/config/middle-name-sets/{set_id}/names/{name}")
async def update_middle_name(set_id: str, name: str, body: UpdateNameRequest, request: Request):
    _require_config_password(request)
    try:
        return svc.update_name(set_id, name, body.new_name)
    except svc.MiddleNameSetsError as e:
        raise _error(e)


@router.delete("/api/config/middle-name-sets/{set_id}/names/{name}")
async def delete_middle_name(set_id: str, name: str, request: Request):
    _require_config_password(request)
    try:
        return svc.delete_name(set_id, name)
    except svc.MiddleNameSetsError as e:
        raise _error(e)
