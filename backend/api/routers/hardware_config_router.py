"""
Hardware Configuration Router
FastAPI endpoints for hardware/deployment configuration management.

⚠️ RESTRICTED ACCESS - SOFTWARE_ADMIN ONLY ⚠️

This router provides endpoints for managing:
- Database connection settings
- External system view names
- Network/API configuration
- SMTP email server settings
- Deployment mode

All endpoints require SOFTWARE_ADMIN role.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List
from pydantic import BaseModel

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_logged_in, require_software_admin
from ..services.hardware_config_service import HardwareConfigService


router = APIRouter(prefix="/api/hardware-config", tags=["Hardware Configuration"])


# ==================== REQUEST/RESPONSE MODELS ====================

class ConfigUpdateRequest(BaseModel):
    """Request model for updating a single configuration."""
    value: str


class ConfigBatchUpdateRequest(BaseModel):
    """Request model for updating multiple configurations."""
    updates: List[dict]  # List of {key: str, value: str}


class DatabaseTestRequest(BaseModel):
    """Request model for testing database connection."""
    server: str
    database: str
    driver: str = "ODBC Driver 17 for SQL Server"
    use_windows_auth: bool = True
    username: Optional[str] = None
    password: Optional[str] = None


class SmtpTestRequest(BaseModel):
    """Request model for testing SMTP connection."""
    host: str
    port: int = 25
    use_tls: bool = False
    use_ssl: bool = False
    username: Optional[str] = None
    password: Optional[str] = None


# ==================== GET ENDPOINTS ====================

@router.get("")
async def get_all_configurations(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get all hardware configurations organized by group.
    
    **Access:** SOFTWARE_ADMIN only
    
    **Returns:**
    - groups: Dictionary of configuration groups with their settings
    - total_configs: Total number of configuration entries
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    result = HardwareConfigService.get_all_configurations()
    return result


@router.get("/summary")
async def get_deployment_summary(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get a quick summary of current deployment configuration.
    
    **Access:** SOFTWARE_ADMIN only
    
    **Returns:**
    Summary of deployment mode, database, network, and email settings.
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    return HardwareConfigService.get_deployment_summary()


@router.get("/groups")
async def get_configuration_groups(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get list of configuration groups with metadata.
    
    **Access:** SOFTWARE_ADMIN only
    
    **Returns:**
    List of available configuration groups with names and descriptions.
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    return {
        "groups": HardwareConfigService.GROUPS
    }


@router.get("/group/{group_name}")
async def get_group_configurations(
    group_name: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get configurations for a specific group.
    
    **Access:** SOFTWARE_ADMIN only
    
    **Groups:**
    - database: SQL Server connection settings
    - views: External system view names
    - network: Backend API and CORS settings
    - email: SMTP server configuration
    - system: Deployment mode settings
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    if group_name not in HardwareConfigService.GROUPS:
        raise HTTPException(
            status_code=404,
            detail=f"Configuration group '{group_name}' not found"
        )
    
    return HardwareConfigService.get_group_configurations(group_name)


@router.get("/key/{config_key}")
async def get_configuration(
    config_key: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get a single configuration by key.
    
    **Access:** SOFTWARE_ADMIN only
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    config = HardwareConfigService.get_configuration(config_key)
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Configuration '{config_key}' not found"
        )
    
    return config


# ==================== UPDATE ENDPOINTS ====================

@router.put("/key/{config_key}")
async def update_configuration(
    config_key: str,
    request: ConfigUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Update a single configuration value.
    
    **Access:** SOFTWARE_ADMIN only
    
    **Note:** Some configurations are read-only and cannot be updated.
    Password fields can be updated but will show as masked.
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    result = HardwareConfigService.update_configuration(
        key=config_key,
        value=request.value,
        user_id=current_user.user_id
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Update failed")
        )
    
    return result


@router.put("/batch")
async def update_configurations_batch(
    request: ConfigBatchUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Update multiple configurations at once.
    
    **Access:** SOFTWARE_ADMIN only
    
    **Request Format:**
    ```json
    {
        "updates": [
            {"key": "db_server", "value": "192.168.1.100"},
            {"key": "backend_port", "value": "8000"}
        ]
    }
    ```
    
    **Returns:**
    - success_count: Number of successful updates
    - error_count: Number of failed updates
    - errors: List of error details
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    return HardwareConfigService.update_configurations_batch(
        updates=request.updates,
        user_id=current_user.user_id
    )


# ==================== TEST ENDPOINTS ====================

@router.post("/test/database")
async def test_database_connection(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Test database connection with current configuration.
    
    **Access:** SOFTWARE_ADMIN only
    
    **Returns:**
    - success: True if connection successful
    - message: Connection result message
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    return HardwareConfigService.test_database_connection()


@router.post("/test/database/custom")
async def test_custom_database_connection(
    request: DatabaseTestRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Test database connection with custom parameters.
    
    **Access:** SOFTWARE_ADMIN only
    
    Use this to test new database settings before saving them.
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    return HardwareConfigService.test_custom_database_connection(
        server=request.server,
        database=request.database,
        driver=request.driver,
        use_windows_auth=request.use_windows_auth,
        username=request.username,
        password=request.password
    )


@router.post("/test/smtp")
async def test_smtp_connection(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Test SMTP connection with current configuration.
    
    **Access:** SOFTWARE_ADMIN only
    
    **Returns:**
    - success: True if connection successful
    - message: Connection result message
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    return HardwareConfigService.test_smtp_connection()


@router.post("/test/smtp/custom")
async def test_custom_smtp_connection(
    request: SmtpTestRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Test SMTP connection with custom parameters.
    
    **Access:** SOFTWARE_ADMIN only
    
    Use this to test new SMTP settings before saving them.
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    return HardwareConfigService.test_custom_smtp_connection(
        host=request.host,
        port=request.port,
        use_tls=request.use_tls,
        use_ssl=request.use_ssl,
        username=request.username,
        password=request.password
    )
