"""
PHASE K — Migration Orchestrator Service

Service layer for coordinating legacy case migration.

This service orchestrates:
- Pre-check for existing mappings (idempotent)
- Migrated case insert via create_record_migrated
- Mapping row insert
- Duplicate handling and retry safety
"""

from typing import Dict, Any
from api.services.migration_insert_service import create_record_migrated
from api.db_layer.migration_map_db import (
    get_migration_map_by_legacy_id,
    insert_migration_mapping,
    find_case_by_complaint_text
)



def migrate_legacy_case(
    legacy_case_id: int,
    payload: Dict[str, Any],
    migrated_by_user_id: int
) -> Dict[str, Any]:
    """
    Migrate a single legacy case into APP_IncidentCase with mapping.
    
    This function is idempotent and retry-safe:
    - Checks for existing mapping before attempting migration
    - Handles duplicate mapping errors gracefully
    - Returns consistent result structure
    - Never crashes on duplicate attempts
    
    Args:
        legacy_case_id: ID from legacy IncidentRequestCase table
        payload: Case data for create_record_migrated
        migrated_by_user_id: User ID performing the migration
        
    Returns:
        dict: One of:
        
        Success (new migration):
        {
            "success": True,
            "status": "MIGRATED",
            "legacy_case_id": int,
            "new_case_id": int
        }
        
        Success (already migrated):
        {
            "success": True,
            "status": "ALREADY_MIGRATED",
            "legacy_case_id": int,
            "new_case_id": int
        }
        
        Failure (case insert failed):
        {
            "success": False,
            "error": "...",
            "message": "..."
        }
        
        Failure (mapping write failed):
        {
            "success": False,
            "error": "MAPPING_WRITE_FAILED",
            "message": "..."
        }
    """
    
    # STEP 1: Pre-check mapping table (idempotent protection)
    existing_mapping = get_migration_map_by_legacy_id(legacy_case_id)
    
    if existing_mapping.get("exists"):
        return {
            "success": True,
            "status": "ALREADY_MIGRATED",
            "legacy_case_id": legacy_case_id,
            "new_case_id": existing_mapping.get("new_case_id")
        }
    
    # STEP 1.5: Fallback check for orphaned cases (partial failure recovery)
    # If mapping doesn't exist but case was created, find it by complaint text
    complaint_text = payload.get("complaint_text", "")
    if complaint_text:
        existing_case = find_case_by_complaint_text(complaint_text)
        
        if existing_case.get("exists"):
            # Found orphaned case - create mapping for it
            orphaned_case_id = existing_case.get("case_id")
            
            try:
                insert_migration_mapping(
                    legacy_case_id,
                    orphaned_case_id,
                    migrated_by_user_id
                )
                
                # Successfully recovered from partial failure
                return {
                    "success": True,
                    "status": "MIGRATED",
                    "legacy_case_id": legacy_case_id,
                    "new_case_id": orphaned_case_id,
                    "note": "Recovered orphaned case from partial failure"
                }
            
            except ValueError as ve:
                # Unexpected - mapping should not exist at this point
                error_msg = str(ve).lower()
                
                if "already migrated" in error_msg:
                    # Race condition - another process just created mapping
                    return {
                        "success": True,
                        "status": "ALREADY_MIGRATED",
                        "legacy_case_id": legacy_case_id,
                        "new_case_id": orphaned_case_id
                    }
                else:
                    return {
                        "success": False,
                        "error": "MAPPING_WRITE_FAILED",
                        "message": str(ve),
                        "new_case_id": orphaned_case_id
                    }
            
            except Exception as e:
                # Generic error during recovery mapping
                return {
                    "success": False,
                    "error": "MAPPING_WRITE_FAILED",
                    "message": f"Recovery failed: {str(e)}",
                    "new_case_id": orphaned_case_id
                }
    
    # STEP 2: Create migrated case
    # Note: create_record_migrated returns {"success": bool, "id": int, ...}
    # The key is "id" not "incident_id"
    insert_result = create_record_migrated(payload, legacy_case_id, migrated_by_user_id)
    
    if not insert_result.get("success"):
        # Case insert failed - return error directly
        return insert_result
    
    new_case_id = insert_result.get("id")
    
    # STEP 3: Write mapping row
    try:
        insert_migration_mapping(
            legacy_case_id,
            new_case_id,
            migrated_by_user_id
        )
        
        # Success - both case and mapping created
        return {
            "success": True,
            "status": "MIGRATED",
            "legacy_case_id": legacy_case_id,
            "new_case_id": new_case_id
        }
        
    except ValueError as ve:
        # Duplicate mapping detected by proactive check
        # This means another process migrated between our pre-check and now
        error_msg = str(ve).lower()
        
        if "already migrated" in error_msg:
            return {
                "success": True,
                "status": "ALREADY_MIGRATED",
                "legacy_case_id": legacy_case_id,
                "new_case_id": new_case_id
            }
        else:
            # Unexpected ValueError
            return {
                "success": False,
                "error": "MAPPING_WRITE_FAILED",
                "message": str(ve),
                "new_case_id": new_case_id  # Case was created but mapping failed
            }
    
    except Exception as e:
        # Generic database error during mapping insert
        error_msg = str(e).lower()
        
        # Check if it's specifically a unique constraint/duplicate key violation
        # These keywords indicate the SAME legacy_case_id was already mapped
        is_duplicate_error = any(keyword in error_msg for keyword in [
            "unique",
            "duplicate key", 
            "uq_app_datamigration_map_legacycase"  # Specific unique constraint name
        ])
        
        if is_duplicate_error:
            # Database caught duplicate - treat as already migrated
            return {
                "success": True,
                "status": "ALREADY_MIGRATED",
                "legacy_case_id": legacy_case_id,
                "new_case_id": new_case_id
            }
        else:
            # Genuine error (FK violation, connection error, etc) - mapping write failed
            return {
                "success": False,
                "error": "MAPPING_WRITE_FAILED",
                "message": str(e),
                "new_case_id": new_case_id  # Case was created but mapping failed
            }
