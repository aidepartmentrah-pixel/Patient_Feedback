"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Investigation Page
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import date
from typing import List, Optional, Dict, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/investigation", tags=["Investigation"])


# =====================================================
# Enums / Type Aliases
# =====================================================

TreeType = Literal[
    "incident_count",
    "domain_distribution_numbers",
    "domain_distribution_percentage",
    "severity_distribution_numbers",
    "severity_distribution_percentage",
    "red_flag_incidents",
    "never_event_incidents",
]

NodeType = Literal["administration", "department", "section"]


# =====================================================
# Season Models
# =====================================================

class InvestigationSeason(BaseModel):
    season_id: str
    season_label: str
    start_date: date
    end_date: date
    is_current: bool


class InvestigationSeasonsResponse(BaseModel):
    seasons: List[InvestigationSeason]
    current_season: str


# =====================================================
# Hierarchy Models (Optional Endpoint)
# =====================================================

class AdministrationNode(BaseModel):
    id: str
    name_en: str
    name_ar: str


class DepartmentNode(BaseModel):
    id: str
    administration_id: str
    name_en: str
    name_ar: str


class SectionNode(BaseModel):
    id: str
    department_id: str
    name_en: str
    name_ar: str


class InvestigationHierarchyResponse(BaseModel):
    administrations: List[AdministrationNode]
    departments: List[DepartmentNode]
    sections: List[SectionNode]


# =====================================================
# Tree Node Models
# =====================================================

class BaseTreeNode(BaseModel):
    node_id: str
    node_name: str
    node_name_ar: Optional[str] = None
    node_type: NodeType
    parent_id: Optional[str] = None
    level: int
    value: float | int
    children: List["BaseTreeNode"] = []


class DomainBreakdownNode(BaseTreeNode):
    total_incidents: Optional[int] = None
    domain_breakdown: Optional[Dict[str, float | int]] = None


class SeverityBreakdownNode(BaseTreeNode):
    total_incidents: Optional[int] = None
    severity_breakdown: Optional[Dict[str, float | int]] = None


class FlaggedIncidentNode(BaseTreeNode):
    total_incidents: Optional[int] = None
    red_flag_percentage: Optional[float] = None


BaseTreeNode.model_rebuild()


# =====================================================
# Response Models
# =====================================================

class InvestigationScope(BaseModel):
    level: Literal["hospital", "administration", "department", "section"]
    administration_id: Optional[str] = None
    department_id: Optional[str] = None
    section_id: Optional[str] = None


class InvestigationSummary(BaseModel):
    total_incidents: Optional[int] = None
    administration_count: Optional[int] = None
    department_count: Optional[int] = None
    section_count: Optional[int] = None

    total_red_flags: Optional[int] = None
    overall_red_flag_percentage: Optional[float] = None

    overall_domain_breakdown: Optional[Dict[str, float]] = None


class InvestigationTreeResponse(BaseModel):
    season: str
    season_label: Optional[str] = None
    tree_type: TreeType
    scope: InvestigationScope
    tree: List[BaseTreeNode]
    summary: Optional[InvestigationSummary] = None


# =====================================================
# Routes
# =====================================================

@router.get(
    "/tree",
    response_model=InvestigationTreeResponse,
)
def get_investigation_tree(
    season: str = Query(...),
    tree_type: TreeType = Query(...),
    administration_id: Optional[str] = Query(None),
    department_id: Optional[str] = Query(None),
    section_id: Optional[str] = Query(None),
):
    """
    Fetch hierarchical aggregated investigation tree data.
    """
    raise NotImplementedError


@router.get(
    "/seasons",
    response_model=InvestigationSeasonsResponse,
)
def get_investigation_seasons():
    """
    Fetch available investigation seasons/periods.
    """
    raise NotImplementedError


@router.get(
    "/hierarchy",
    response_model=InvestigationHierarchyResponse,
)
def get_investigation_hierarchy():
    """
    Fetch organizational hierarchy for investigation filters.
    """
    raise NotImplementedError
