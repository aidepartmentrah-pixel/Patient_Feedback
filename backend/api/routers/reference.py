"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Insert Record Page (Reference Data)
Iteration: 1
Status: API skeleton only – no implementation
"""

from typing import List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/reference", tags=["Reference Data"])


# -------------------------
# Shared Models
# -------------------------

class IdNameArEn(BaseModel):
    id: int
    name_en: str
    name_ar: str


class IdName(BaseModel):
    id: int
    name: str


# -------------------------
# Response Models
# -------------------------

class DepartmentsResponse(BaseModel):
    departments: List[IdNameArEn]


class SourcesResponse(BaseModel):
    sources: List[IdName]


class DomainsResponse(BaseModel):
    domains: List[IdNameArEn]


class CategoriesResponse(BaseModel):
    categories: List[IdNameArEn]


class SubcategoriesResponse(BaseModel):
    subcategories: List[IdNameArEn]


class ClassificationsResponse(BaseModel):
    classifications: List[IdNameArEn]


class SeverityLevelsResponse(BaseModel):
    severity_levels: List[IdNameArEn]


class StagesResponse(BaseModel):
    stages: List[IdNameArEn]


class HarmLevelsResponse(BaseModel):
    harm_levels: List[IdNameArEn]


class ReferenceAllResponse(BaseModel):
    departments: List[IdNameArEn]
    sources: List[IdName]
    domains: List[IdNameArEn]
    categories: List[IdNameArEn]
    subcategories: List[IdNameArEn]
    classifications: List[IdNameArEn]
    severity_levels: List[IdNameArEn]
    stages: List[IdNameArEn]
    harm_levels: List[IdNameArEn]


# -------------------------
# Routes
# -------------------------

@router.get("/departments", response_model=DepartmentsResponse)
def get_departments():
    raise NotImplementedError


@router.get("/sources", response_model=SourcesResponse)
def get_sources():
    raise NotImplementedError


@router.get("/domains", response_model=DomainsResponse)
def get_domains():
    raise NotImplementedError


@router.get("/categories", response_model=CategoriesResponse)
def get_categories(domain_id: int = Query(...)):
    raise NotImplementedError


@router.get("/subcategories", response_model=SubcategoriesResponse)
def get_subcategories(category_id: int = Query(...)):
    raise NotImplementedError


@router.get("/classifications", response_model=ClassificationsResponse)
def get_classifications(subcategory_id: int = Query(...)):
    raise NotImplementedError


@router.get("/severity-levels", response_model=SeverityLevelsResponse)
def get_severity_levels():
    raise NotImplementedError


@router.get("/stages", response_model=StagesResponse)
def get_stages():
    raise NotImplementedError


@router.get("/harm-levels", response_model=HarmLevelsResponse)
def get_harm_levels():
    raise NotImplementedError


@router.get("/all", response_model=ReferenceAllResponse)
def get_all_reference_data():
    raise NotImplementedError
