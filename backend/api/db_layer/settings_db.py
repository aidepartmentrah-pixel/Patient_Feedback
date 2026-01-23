"""
Database layer for Settings page operations.
Handles departments, variable attributes, and policies.
"""

import pyodbc
from datetime import datetime
from typing import Dict, List, Any, Optional

def get_connection():
    """Get database connection."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


# =============================================
# DEPARTMENTS
# =============================================

def get_all_departments(
    mapping_mode: Optional[str] = None,
    is_active: bool = True,
    flat: bool = False
) -> List[Dict[str, Any]]:
    """Fetch all departments with optional filtering."""
    conn = get_connection()
    cursor = conn.cursor()
    
    where_parts = []
    if mapping_mode:
        where_parts.append(f"MappingMode = '{mapping_mode}'")
    if is_active:
        where_parts.append("IsActive = 1")
    
    where_clause = " AND ".join(where_parts) if where_parts else "1=1"
    
    query = f"""
    SELECT 
        OrgUnitID as id,
        OrgUnitName as name,
        OrgUnitNameAr as name_ar,
        Code as code,
        ParentOrgUnitID as parent_id,
        [Level] as level,
        MappingMode as mapping_mode,
        IsActive as is_active,
        DisplayOrder as display_order,
        CreatedAt as created_at,
        UpdatedAt as updated_at,
        CreatedByUserID as created_by_user_id,
        UpdatedByUserID as updated_by_user_id
    FROM dbo.APP_OrgUnit
    WHERE {where_clause}
    ORDER BY DisplayOrder ASC, OrgUnitName ASC
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    departments = [dict(zip(columns, row)) for row in rows]
    
    conn.close()
    
    if not flat:
        return _build_department_tree(departments)
    return departments


def _build_department_tree(departments: List[Dict]) -> List[Dict]:
    """Build hierarchical tree structure from flat department list."""
    # Create lookup for quick access
    dept_map = {d['id']: d for d in departments}
    
    # Add children and computed fields
    for dept in departments:
        dept['children'] = []
        dept['has_children'] = False
        dept['depth'] = 0
        dept['path'] = f"/{dept['id']}"
    
    # Build parent-child relationships
    roots = []
    for dept in departments:
        if dept['parent_id'] is None:
            roots.append(dept)
        else:
            parent = dept_map.get(dept['parent_id'])
            if parent:
                parent['children'].append(dept)
                parent['has_children'] = True
                dept['depth'] = parent['depth'] + 1
                dept['path'] = parent['path'] + f"/{dept['id']}"
    
    return roots


def get_department_by_id(department_id: int) -> Optional[Dict[str, Any]]:
    """Fetch single department by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT 
        OrgUnitID as id,
        OrgUnitName as name,
        OrgUnitNameAr as name_ar,
        Code as code,
        ParentOrgUnitID as parent_id,
        [Level] as level,
        MappingMode as mapping_mode,
        IsActive as is_active,
        DisplayOrder as display_order,
        CreatedAt as created_at,
        UpdatedAt as updated_at,
        CreatedByUserID as created_by_user_id,
        UpdatedByUserID as updated_by_user_id
    FROM dbo.APP_OrgUnit
    WHERE OrgUnitID = ?
    """, department_id)
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    return None


def create_department(data: dict, created_by_user_id: int) -> int:
    """Create new department."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO dbo.APP_OrgUnit (
        OrgUnitName,
        OrgUnitNameAr,
        Code,
        ParentOrgUnitID,
        [Level],
        MappingMode,
        IsActive,
        DisplayOrder,
        CreatedByUserID,
        UpdatedByUserID,
        CreatedAt,
        UpdatedAt
    )
    OUTPUT INSERTED.OrgUnitID
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE(), GETDATE())
    """,
    data.get('name'),
    data.get('name_ar'),
    data.get('code'),
    data.get('parent_id'),
    data.get('level'),
    data.get('mapping_mode', 'internal'),
    1,
    data.get('display_order', 0),
    created_by_user_id,
    created_by_user_id
    )
    
    dept_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return dept_id


def update_department(department_id: int, data: dict, updated_by_user_id: int) -> None:
    """Update existing department."""
    conn = get_connection()
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if 'name' in data:
        updates.append("OrgUnitName = ?")
        params.append(data['name'])
    if 'name_ar' in data:
        updates.append("OrgUnitNameAr = ?")
        params.append(data['name_ar'])
    if 'code' in data:
        updates.append("Code = ?")
        params.append(data['code'])
    if 'parent_id' in data:
        updates.append("ParentOrgUnitID = ?")
        params.append(data['parent_id'])
    if 'display_order' in data:
        updates.append("DisplayOrder = ?")
        params.append(data['display_order'])
    if 'is_active' in data:
        updates.append("IsActive = ?")
        params.append(1 if data['is_active'] else 0)
    
    updates.append("UpdatedByUserID = ?")
    params.append(updated_by_user_id)
    updates.append("UpdatedAt = GETDATE()")
    
    params.append(department_id)
    
    if updates:
        query = f"UPDATE dbo.APP_OrgUnit SET {', '.join(updates)} WHERE OrgUnitID = ?"
        cursor.execute(query, params)
        conn.commit()
    
    conn.close()


def deactivate_department(department_id: int, updated_by_user_id: int) -> None:
    """Soft delete (deactivate) a department."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    UPDATE dbo.APP_OrgUnit
    SET IsActive = 0, UpdatedByUserID = ?, UpdatedAt = GETDATE()
    WHERE OrgUnitID = ?
    """, updated_by_user_id, department_id)
    
    conn.commit()
    conn.close()


def count_incidents_by_department(department_id: int) -> int:
    """Count incidents associated with a department."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT COUNT(*) FROM dbo.APP_IncidentCase
    WHERE IssuingOrgUnitID = ?
    """, department_id)
    
    count = cursor.fetchone()[0]
    conn.close()
    return count


# =============================================
# VARIABLE ATTRIBUTES
# =============================================

def get_all_attributes(
    attribute_type: Optional[str] = None,
    is_active: bool = True
) -> List[Dict[str, Any]]:
    """Fetch all variable attributes."""
    conn = get_connection()
    cursor = conn.cursor()
    
    where_parts = []
    if attribute_type:
        where_parts.append(f"AT.AttributeType = '{attribute_type}'")
    
    where_clause = " AND ".join(where_parts) if where_parts else "1=1"
    
    query = f"""
    SELECT 
        AT.AttributeType as attribute_type,
        AT.AttributeTypeLabel as attribute_type_label,
        AT.AttributeTypeLabelAr as attribute_type_label_ar,
        AV.AttributeValueID as id,
        AV.AttributeValue as value,
        AV.AttributeValueAr as value_ar,
        AV.Code as code,
        AV.Color as color,
        AV.DisplayOrder as display_order,
        AV.IsActive as is_active
    FROM dbo.APP_AttributeType AT
    LEFT JOIN dbo.APP_AttributeValue AV ON AT.AttributeTypeID = AV.AttributeTypeID
    WHERE {where_clause}
    ORDER BY AT.AttributeType, AV.DisplayOrder
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    
    # Group by attribute type
    attributes_dict = {}
    for row in rows:
        row_dict = dict(zip(columns, row))
        attr_type = row_dict['attribute_type']
        
        if attr_type not in attributes_dict:
            attributes_dict[attr_type] = {
                'attribute_type': attr_type,
                'attribute_type_label': row_dict['attribute_type_label'],
                'attribute_type_label_ar': row_dict['attribute_type_label_ar'],
                'values': []
            }
        
        if row_dict['id']:  # Only add if value exists
            attributes_dict[attr_type]['values'].append({
                'id': row_dict['id'],
                'value': row_dict['value'],
                'value_ar': row_dict['value_ar'],
                'code': row_dict['code'],
                'color': row_dict['color'],
                'display_order': row_dict['display_order'],
                'is_active': row_dict['is_active']
            })
    
    conn.close()
    return list(attributes_dict.values())


def update_attribute_values(attribute_type: str, values: List[Dict]) -> Dict[str, Any]:
    """Update attribute values for a specific attribute type."""
    conn = get_connection()
    cursor = conn.cursor()
    
    added_count = 0
    updated_count = 0
    
    for value_data in values:
        value_id = value_data.get('id')
        
        if value_id:
            # Update existing
            cursor.execute("""
            UPDATE dbo.APP_AttributeValue
            SET AttributeValue = ?,
                AttributeValueAr = ?,
                Code = ?,
                Color = ?,
                DisplayOrder = ?,
                IsActive = ?
            WHERE AttributeValueID = ?
            """,
            value_data.get('value'),
            value_data.get('value_ar'),
            value_data.get('code'),
            value_data.get('color'),
            value_data.get('display_order'),
            1 if value_data.get('is_active') else 0,
            value_id
            )
            updated_count += 1
        else:
            # Add new - get attribute type ID first
            cursor.execute("""
            SELECT AttributeTypeID FROM dbo.APP_AttributeType
            WHERE AttributeType = ?
            """, attribute_type)
            
            attr_type_id = cursor.fetchone()
            if attr_type_id:
                cursor.execute("""
                INSERT INTO dbo.APP_AttributeValue (
                    AttributeTypeID,
                    AttributeValue,
                    AttributeValueAr,
                    Code,
                    Color,
                    DisplayOrder,
                    IsActive
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                attr_type_id[0],
                value_data.get('value'),
                value_data.get('value_ar'),
                value_data.get('code'),
                value_data.get('color'),
                value_data.get('display_order'),
                1 if value_data.get('is_active') else 0
                )
                added_count += 1
    
    conn.commit()
    conn.close()
    
    return {
        'added_count': added_count,
        'updated_count': updated_count
    }


# =============================================
# POLICIES
# =============================================

def get_all_policies(
    category: Optional[str] = None,
    scope: Optional[str] = None,
    department_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fetch all policies with optional filtering."""
    conn = get_connection()
    cursor = conn.cursor()
    
    where_parts = []
    if category:
        where_parts.append(f"Category = '{category}'")
    if scope:
        where_parts.append(f"Scope = '{scope}'")
    if department_id:
        where_parts.append(f"DepartmentID = {department_id}")
    
    where_clause = " AND ".join(where_parts) if where_parts else "1=1"
    
    query = f"""
    SELECT 
        PolicyKey as policy_key,
        PolicyName as policy_name,
        PolicyNameAr as policy_name_ar,
        PolicyValue as policy_value,
        PolicyType as policy_type,
        Category as category,
        [Description] as description,
        DescriptionAr as description_ar,
        CASE WHEN DepartmentID IS NULL THEN 1 ELSE 0 END as is_global,
        Scope as scope,
        DepartmentID as department_id,
        UpdatedAt as updated_at
    FROM dbo.APP_Policy
    WHERE {where_clause}
    ORDER BY Category, PolicyKey
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    policies = [dict(zip(columns, row)) for row in rows]
    
    conn.close()
    return policies


def update_policies(policy_updates: List[Dict]) -> List[Dict]:
    """Update multiple policy values."""
    conn = get_connection()
    cursor = conn.cursor()
    
    updated_policies = []
    
    for policy in policy_updates:
        cursor.execute("""
        UPDATE dbo.APP_Policy
        SET PolicyValue = ?, UpdatedAt = GETDATE()
        WHERE PolicyKey = ?
        """,
        str(policy.get('policy_value')),
        policy.get('policy_key')
        )
        
        updated_policies.append({
            'policy_key': policy.get('policy_key'),
            'policy_value': policy.get('policy_value'),
            'updated_at': datetime.now().isoformat()
        })
    
    conn.commit()
    conn.close()
    
    return updated_policies


# =============================================
# EXPORT/IMPORT/SNAPSHOT
# =============================================

def export_full_configuration(
    include_inactive: bool = False
) -> Dict[str, Any]:
    """Export entire system configuration."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Departments
    where = "1=1" if include_inactive else "IsActive = 1"
    cursor.execute(f"""
    SELECT OrgUnitID, OrgUnitName, OrgUnitNameAr, Code, ParentOrgUnitID,
           [Level], MappingMode, IsActive
    FROM dbo.APP_OrgUnit
    WHERE {where}
    ORDER BY DisplayOrder
    """)
    dept_rows = cursor.fetchall()
    departments = []
    for row in dept_rows:
        departments.append({
            'id': row[0],
            'name': row[1],
            'name_ar': row[2],
            'code': row[3],
            'parent_id': row[4],
            'level': row[5],
            'mapping_mode': row[6],
            'is_active': row[7]
        })
    
    # Attributes
    cursor.execute("""
    SELECT AT.AttributeType, AT.AttributeTypeLabel, AT.AttributeTypeLabelAr,
           AV.AttributeValueID, AV.AttributeValue, AV.AttributeValueAr,
           AV.Code, AV.Color, AV.DisplayOrder, AV.IsActive
    FROM dbo.APP_AttributeType AT
    LEFT JOIN dbo.APP_AttributeValue AV ON AT.AttributeTypeID = AV.AttributeTypeID
    ORDER BY AT.AttributeType, AV.DisplayOrder
    """)
    
    attr_rows = cursor.fetchall()
    attributes_dict = {}
    for row in attr_rows:
        attr_type = row[0]
        if attr_type not in attributes_dict:
            attributes_dict[attr_type] = {
                'attribute_type': attr_type,
                'attribute_type_label': row[1],
                'attribute_type_label_ar': row[2],
                'values': []
            }
        
        if row[3]:  # If value exists
            attributes_dict[attr_type]['values'].append({
                'id': row[3],
                'value': row[4],
                'value_ar': row[5],
                'code': row[6],
                'color': row[7],
                'display_order': row[8],
                'is_active': row[9]
            })
    
    # Policies
    cursor.execute("""
    SELECT PolicyKey, PolicyName, PolicyNameAr, PolicyValue, PolicyType,
           Category, [Description], DescriptionAr, Scope, DepartmentID
    FROM dbo.APP_Policy
    ORDER BY Category, PolicyKey
    """)
    
    policy_rows = cursor.fetchall()
    policies = []
    for row in policy_rows:
        policies.append({
            'policy_key': row[0],
            'policy_name': row[1],
            'policy_name_ar': row[2],
            'policy_value': row[3],
            'policy_type': row[4],
            'category': row[5],
            'description': row[6],
            'description_ar': row[7],
            'scope': row[8],
            'department_id': row[9]
        })
    
    conn.close()
    
    return {
        'departments': departments,
        'attributes': list(attributes_dict.values()),
        'policies': policies
    }


def save_configuration_snapshot(
    snapshot_name: str,
    snapshot_name_ar: str,
    description: str,
    created_by_user_id: int,
    config_data: Dict
) -> str:
    """Save configuration snapshot for rollback."""
    conn = get_connection()
    cursor = conn.cursor()
    
    import json
    snapshot_id = f"snap-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config_json = json.dumps(config_data)
    
    cursor.execute("""
    INSERT INTO dbo.APP_ConfigSnapshot (
        SnapshotID,
        SnapshotName,
        SnapshotNameAr,
        [Description],
        ConfigData,
        CreatedAt,
        CreatedByUserID
    )
    VALUES (?, ?, ?, ?, ?, GETDATE(), ?)
    """,
    snapshot_id,
    snapshot_name,
    snapshot_name_ar,
    description,
    config_json,
    created_by_user_id
    )
    
    conn.commit()
    conn.close()
    
    return snapshot_id


def get_configuration_snapshots() -> List[Dict]:
    """Get list of saved configuration snapshots."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT SnapshotID, SnapshotName, SnapshotNameAr, [Description],
           CreatedAt, CreatedByUserID
    FROM dbo.APP_ConfigSnapshot
    ORDER BY CreatedAt DESC
    """)
    
    rows = cursor.fetchall()
    snapshots = []
    for row in rows:
        snapshots.append({
            'snapshot_id': row[0],
            'snapshot_name': row[1],
            'snapshot_name_ar': row[2],
            'description': row[3],
            'created_at': row[4],
            'created_by_user_id': row[5]
        })
    
    conn.close()
    return snapshots


# =============================================
# SYSTEM SETTINGS
# =============================================

def get_all_system_settings(is_active: bool = True) -> List[Dict[str, Any]]:
    """Get all system settings."""
    conn = get_connection()
    cursor = conn.cursor()
    
    where_clause = "WHERE IsActive = 1" if is_active else ""
    
    query = f"""
    SELECT 
        SettingID as id,
        SettingKey as setting_key,
        SettingValue as setting_value,
        SettingLabel as label,
        SettingLabelAr as label_ar,
        SettingType as setting_type,
        Description as description,
        DescriptionAr as description_ar,
        IsActive as is_active,
        CreatedAt as created_at,
        UpdatedAt as updated_at,
        UpdatedBy as updated_by
    FROM dbo.APP_SystemSettings
    {where_clause}
    ORDER BY SettingKey
    """
    
    try:
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            setting = dict(zip(columns, row))
            # Format datetimes
            if setting.get('created_at'):
                setting['created_at'] = setting['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            if setting.get('updated_at'):
                setting['updated_at'] = setting['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
            results.append(setting)
        
        return results
    finally:
        cursor.close()
        conn.close()


def get_setting_by_key(setting_key: str) -> Optional[Dict[str, Any]]:
    """Get a specific setting by key."""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT 
        SettingID as id,
        SettingKey as setting_key,
        SettingValue as setting_value,
        SettingLabel as label,
        SettingLabelAr as label_ar,
        SettingType as setting_type,
        Description as description,
        DescriptionAr as description_ar,
        IsActive as is_active,
        CreatedAt as created_at,
        UpdatedAt as updated_at,
        UpdatedBy as updated_by
    FROM dbo.APP_SystemSettings
    WHERE SettingKey = ? AND IsActive = 1
    """
    
    try:
        cursor.execute(query, (setting_key,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        columns = [col[0] for col in cursor.description]
        setting = dict(zip(columns, row))
        
        # Format datetimes
        if setting.get('created_at'):
            setting['created_at'] = setting['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        if setting.get('updated_at'):
            setting['updated_at'] = setting['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return setting
    finally:
        cursor.close()
        conn.close()


def update_setting(setting_key: str, setting_value: str, updated_by: Optional[int] = None) -> bool:
    """Update a setting value."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            UPDATE dbo.APP_SystemSettings
            SET SettingValue = ?,
                UpdatedAt = GETDATE(),
                UpdatedBy = ?
            WHERE SettingKey = ?
        """, (setting_value, updated_by, setting_key))
        
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


def create_setting(
    setting_key: str,
    setting_value: str,
    label: Optional[str] = None,
    label_ar: Optional[str] = None,
    setting_type: str = 'text',
    description: Optional[str] = None,
    description_ar: Optional[str] = None,
    created_by: Optional[int] = None
) -> int:
    """Create a new system setting."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO dbo.APP_SystemSettings 
            (SettingKey, SettingValue, SettingLabel, SettingLabelAr, SettingType, 
             Description, DescriptionAr, UpdatedBy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            
            SELECT SCOPE_IDENTITY()
        """, (setting_key, setting_value, label, label_ar, setting_type, 
              description, description_ar, created_by))
        
        setting_id = cursor.fetchone()[0]
        conn.commit()
        return int(setting_id)
    finally:
        cursor.close()
        conn.close()
