"""
Settings Service Layer
Business logic for departments, attributes, and policies.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from ..db_layer import settings_db


class SettingsService:
    """Service class for settings operations."""
    
    @staticmethod
    def get_departments(
        mapping_mode: Optional[str] = None,
        is_active: bool = True,
        include_children: bool = True,
        flat: bool = False
    ) -> Dict[str, Any]:
        """Fetch departments with optional filtering."""
        departments = settings_db.get_all_departments(
            mapping_mode=mapping_mode,
            is_active=is_active,
            flat=flat
        )
        
        # Count active/total
        all_depts = settings_db.get_all_departments(is_active=False, flat=True)
        total_count = len(all_depts)
        active_count = len([d for d in all_depts if d['is_active']])
        
        return {
            'mapping_mode': mapping_mode,
            'departments': departments,
            'total_count': total_count,
            'active_count': active_count
        }
    
    @staticmethod
    def create_department(
        name: str,
        name_ar: str,
        code: str,
        parent_id: Optional[int],
        mapping_mode: str,
        is_active: bool,
        display_order: int,
        created_by_user_id: int
    ) -> Dict[str, Any]:
        """Create new department."""
        # Check for duplicate code
        all_depts = settings_db.get_all_departments(is_active=False, flat=True)
        if any(d['code'] == code and d['mapping_mode'] == mapping_mode for d in all_depts):
            raise ValueError(f"Department code '{code}' already exists in {mapping_mode} mode")
        
        # Determine level based on parent
        level = 1
        if parent_id:
            parent = settings_db.get_department_by_id(parent_id)
            if not parent:
                raise ValueError(f"Parent department {parent_id} not found")
            level = parent['level'] + 1
        
        dept_data = {
            'name': name,
            'name_ar': name_ar,
            'code': code,
            'parent_id': parent_id,
            'level': level,
            'mapping_mode': mapping_mode,
            'display_order': display_order
        }
        
        dept_id = settings_db.create_department(dept_data, created_by_user_id)
        dept = settings_db.get_department_by_id(dept_id)
        
        return {
            **dept,
            'message': 'Department created successfully',
            'message_ar': 'تم إنشاء القسم بنجاح'
        }
    
    @staticmethod
    def update_department(
        department_id: int,
        name: Optional[str],
        name_ar: Optional[str],
        code: Optional[str],
        parent_id: Optional[int],
        is_active: Optional[bool],
        display_order: Optional[int],
        updated_by_user_id: int
    ) -> Dict[str, Any]:
        """Update existing department."""
        dept = settings_db.get_department_by_id(department_id)
        if not dept:
            raise ValueError(f"Department {department_id} not found")
        
        # Check for circular parent reference
        if parent_id and parent_id != dept['parent_id']:
            if parent_id == department_id:
                raise ValueError("Cannot set department as its own parent")
            
            # Check if parent_id is a descendant
            all_depts = settings_db.get_all_departments(is_active=False, flat=True)
            parent_descendants = _get_descendants(department_id, all_depts)
            if parent_id in parent_descendants:
                raise ValueError("Cannot set parent_id to a descendant department")
        
        update_data = {}
        if name is not None:
            update_data['name'] = name
        if name_ar is not None:
            update_data['name_ar'] = name_ar
        if code is not None:
            update_data['code'] = code
        if parent_id is not None:
            update_data['parent_id'] = parent_id
        if is_active is not None:
            update_data['is_active'] = is_active
        if display_order is not None:
            update_data['display_order'] = display_order
        
        settings_db.update_department(department_id, update_data, updated_by_user_id)
        updated_dept = settings_db.get_department_by_id(department_id)
        
        return {
            **updated_dept,
            'message': 'Department updated successfully',
            'message_ar': 'تم تحديث القسم بنجاح'
        }
    
    @staticmethod
    def delete_department(
        department_id: int,
        force: bool = False,
        updated_by_user_id: int = None
    ) -> Dict[str, Any]:
        """Delete or deactivate a department."""
        dept = settings_db.get_department_by_id(department_id)
        if not dept:
            raise ValueError(f"Department {department_id} not found")
        
        # Check for associated incidents
        incident_count = settings_db.count_incidents_by_department(department_id)
        
        if incident_count > 0 and not force:
            raise ValueError(
                f"Cannot delete department with {incident_count} associated incidents. "
                f"Use force=true to deactivate instead."
            )
        
        # Always soft delete (deactivate)
        settings_db.deactivate_department(department_id, updated_by_user_id)
        
        return {
            'id': department_id,
            'is_active': False,
            'deleted_at': datetime.now().isoformat(),
            'message': 'Department deactivated successfully',
            'message_ar': 'تم إلغاء تنشيط القسم بنجاح'
        }
    
    @staticmethod
    def get_attributes(
        attribute_type: Optional[str] = None,
        is_active: bool = True
    ) -> Dict[str, Any]:
        """Fetch variable attributes."""
        attributes = settings_db.get_all_attributes(
            attribute_type=attribute_type,
            is_active=is_active
        )
        
        return {
            'attributes': attributes,
            'total_attribute_types': len(attributes),
            'last_updated_at': datetime.now().isoformat()
        }
    
    @staticmethod
    def update_attributes(
        attribute_type: str,
        values: List[Dict]
    ) -> Dict[str, Any]:
        """Update attribute values."""
        result = settings_db.update_attribute_values(attribute_type, values)
        
        # Fetch updated values
        updated_attrs = settings_db.get_all_attributes(attribute_type=attribute_type)
        
        return {
            'attribute_type': attribute_type,
            'updated_count': result['updated_count'],
            'added_count': result['added_count'],
            'deactivated_count': 0,
            'values': updated_attrs[0]['values'] if updated_attrs else [],
            'message': 'Attribute values updated successfully',
            'message_ar': 'تم تحديث قيم السمة بنجاح'
        }
    
    @staticmethod
    def get_policies(
        category: Optional[str] = None,
        scope: Optional[str] = None,
        department_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch policies."""
        policies = settings_db.get_all_policies(
            category=category,
            scope=scope,
            department_id=department_id
        )
        
        global_count = len([p for p in policies if p['is_global']])
        dept_count = len([p for p in policies if not p['is_global']])
        
        return {
            'policies': policies,
            'total_policies': len(policies),
            'global_policies': global_count,
            'department_policies': dept_count
        }
    
    @staticmethod
    def update_policies(policy_updates: List[Dict]) -> Dict[str, Any]:
        """Update multiple policies."""
        updated = settings_db.update_policies(policy_updates)
        
        return {
            'updated_count': len(updated),
            'policies': updated,
            'message': 'Policy configuration updated successfully',
            'message_ar': 'تم تحديث تكوين السياسة بنجاح'
        }
    
    @staticmethod
    def export_configuration(
        include_inactive: bool = False,
        format: str = 'json'
    ) -> Dict[str, Any]:
        """Export full system configuration."""
        data = settings_db.export_full_configuration(include_inactive=include_inactive)
        
        export_id = f"cfg-exp-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            'export_id': export_id,
            'config_version': 'v1.2.3',
            'exported_at': datetime.now().isoformat(),
            'exported_by': 'admin@hospital.local',
            'data': data,
            'metadata': {
                'total_departments': len(data['departments']),
                'total_attributes': len(data['attributes']),
                'total_policies': len(data['policies'])
            }
        }
    
    @staticmethod
    def save_snapshot(
        snapshot_name: str,
        snapshot_name_ar: str,
        description: str,
        created_by_user_id: int
    ) -> Dict[str, Any]:
        """Save configuration snapshot."""
        # Export current configuration
        config_data = settings_db.export_full_configuration(include_inactive=True)
        
        snapshot_id = settings_db.save_configuration_snapshot(
            snapshot_name,
            snapshot_name_ar,
            description,
            created_by_user_id,
            config_data
        )
        
        return {
            'snapshot_id': snapshot_id,
            'snapshot_name': snapshot_name,
            'config_version': 'v1.2.3',
            'created_at': datetime.now().isoformat(),
            'created_by_user_id': created_by_user_id,
            'message': 'Configuration snapshot saved successfully',
            'message_ar': 'تم حفظ لقطة التكوين بنجاح'
        }
    
    @staticmethod
    def get_snapshots() -> Dict[str, Any]:
        """Get list of saved snapshots."""
        snapshots = settings_db.get_configuration_snapshots()
        
        return {
            'snapshots': snapshots,
            'total_snapshots': len(snapshots)
        }
    
    @staticmethod
    def get_system_settings(is_active: bool = True) -> Dict[str, Any]:
        """Get all system settings."""
        try:
            settings = settings_db.get_all_system_settings(is_active=is_active)
            
            return {
                'settings': settings,
                'total': len(settings),
                'message': f'Retrieved {len(settings)} system setting(s)',
                'message_ar': f'تم جلب {len(settings)} إعداد(ات) النظام'
            }
        except Exception as e:
            raise Exception(f"Failed to fetch system settings: {str(e)}")
    
    @staticmethod
    def get_setting(setting_key: str) -> Dict[str, Any]:
        """Get a specific setting by key."""
        try:
            setting = settings_db.get_setting_by_key(setting_key)
            
            if not setting:
                raise ValueError(f"Setting '{setting_key}' not found")
            
            return setting
        except Exception as e:
            raise Exception(f"Failed to fetch setting: {str(e)}")
    
    @staticmethod
    def update_setting(
        setting_key: str,
        setting_value: str,
        updated_by: Optional[int] = None
    ) -> Dict[str, Any]:
        """Update a system setting."""
        try:
            # Check if setting exists
            setting = settings_db.get_setting_by_key(setting_key)
            if not setting:
                raise ValueError(f"Setting '{setting_key}' not found")
            
            # Update the setting
            success = settings_db.update_setting(setting_key, setting_value, updated_by)
            
            if not success:
                raise Exception("Failed to update setting")
            
            # Get updated setting
            updated_setting = settings_db.get_setting_by_key(setting_key)
            
            return {
                'success': True,
                'setting': updated_setting,
                'message': f"Setting '{setting_key}' updated successfully",
                'message_ar': f"تم تحديث الإعداد '{setting_key}' بنجاح"
            }
        except ValueError as ve:
            raise ValueError(str(ve))
        except Exception as e:
            raise Exception(f"Failed to update setting: {str(e)}")
    
    @staticmethod
    def create_setting(
        setting_key: str,
        setting_value: str,
        label: Optional[str] = None,
        label_ar: Optional[str] = None,
        setting_type: str = 'text',
        description: Optional[str] = None,
        description_ar: Optional[str] = None,
        created_by: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new system setting."""
        try:
            # Validate setting_key format (alphanumeric and underscores only)
            if not re.match(r'^[a-zA-Z0-9_]+$', setting_key):
                raise ValueError("Setting key must contain only letters, numbers, and underscores")
            
            # Check if setting already exists
            existing = settings_db.get_setting_by_key(setting_key)
            if existing:
                raise ValueError(f"Setting '{setting_key}' already exists")
            
            # Create setting
            setting_id = settings_db.create_setting(
                setting_key=setting_key,
                setting_value=setting_value,
                label=label,
                label_ar=label_ar,
                setting_type=setting_type,
                description=description,
                description_ar=description_ar,
                created_by=created_by
            )
            
            # Get created setting
            created_setting = settings_db.get_setting_by_key(setting_key)
            
            return {
                'success': True,
                'setting': created_setting,
                'message': f"Setting '{setting_key}' created successfully",
                'message_ar': f"تم إنشاء الإعداد '{setting_key}' بنجاح"
            }
        except ValueError as ve:
            raise ValueError(str(ve))
        except Exception as e:
            raise Exception(f"Failed to create setting: {str(e)}")


def _get_descendants(department_id: int, all_depts: List[Dict]) -> List[int]:
    """Get all descendant IDs for a department."""
    descendants = []
    for dept in all_depts:
        if dept['parent_id'] == department_id:
            descendants.append(dept['id'])
            descendants.extend(_get_descendants(dept['id'], all_depts))
    return descendants
