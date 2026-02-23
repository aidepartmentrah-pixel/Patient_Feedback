"""
Hardware Configuration Service Layer
Business logic for hardware/deployment configuration management.

This service provides controlled access to hardware configuration settings
that can only be modified by SOFTWARE_ADMIN users.
"""

from typing import Dict, List, Any, Optional
from ..db_layer import hardware_config_db


class HardwareConfigService:
    """
    Service for managing hardware and deployment configuration.
    
    Configuration Groups:
    - database: SQL Server connection settings
    - views: External system view/table names
    - network: Backend API URL and CORS settings
    - email: SMTP server configuration
    - system: Deployment mode and system-wide settings
    """
    
    # Configuration groups metadata
    GROUPS = {
        "database": {
            "name": "Database",
            "name_ar": "قاعدة البيانات",
            "description": "SQL Server connection settings",
            "icon": "database",
            "order": 1
        },
        "views": {
            "name": "External Views",
            "name_ar": "العروض الخارجية",
            "description": "Hospital system view names (HR, HIS)",
            "icon": "table",
            "order": 2
        },
        "network": {
            "name": "Network",
            "name_ar": "الشبكة",
            "description": "Backend API and CORS settings",
            "icon": "globe",
            "order": 3
        },
        "email": {
            "name": "Email (SMTP)",
            "name_ar": "البريد الإلكتروني",
            "description": "SMTP server for notifications",
            "icon": "mail",
            "order": 4
        },
        "system": {
            "name": "System",
            "name_ar": "النظام",
            "description": "Deployment mode and system settings",
            "icon": "settings",
            "order": 5
        }
    }
    
    @staticmethod
    def get_all_configurations() -> Dict[str, Any]:
        """
        Get all configurations organized by group.
        
        Returns:
            Dictionary with groups and their configurations
        """
        all_configs = hardware_config_db.get_all_configs()
        
        # Organize by group
        grouped = {}
        for config in all_configs:
            group = config['ConfigGroup']
            if group not in grouped:
                grouped[group] = {
                    "metadata": HardwareConfigService.GROUPS.get(group, {
                        "name": group,
                        "name_ar": group,
                        "description": "",
                        "icon": "settings",
                        "order": 99
                    }),
                    "configs": []
                }
            grouped[group]["configs"].append({
                "key": config['ConfigKey'],
                "value": config['ConfigValue'],
                "type": config['ConfigType'],
                "display_name": config['DisplayName'],
                "display_name_ar": config['DisplayNameAr'],
                "description": config['Description'],
                "is_encrypted": config['IsEncrypted'],
                "is_editable": config['IsEditable']
            })
        
        # Sort groups by order
        sorted_groups = dict(sorted(
            grouped.items(),
            key=lambda x: x[1]["metadata"].get("order", 99)
        ))
        
        return {
            "groups": sorted_groups,
            "total_configs": len(all_configs)
        }
    
    @staticmethod
    def get_group_configurations(group: str) -> Dict[str, Any]:
        """
        Get configurations for a specific group.
        
        Args:
            group: Configuration group name
        
        Returns:
            Dictionary with group metadata and configurations
        """
        configs = hardware_config_db.get_configs_by_group(group)
        
        return {
            "group": group,
            "metadata": HardwareConfigService.GROUPS.get(group, {
                "name": group,
                "name_ar": group,
                "description": "",
                "icon": "settings",
                "order": 99
            }),
            "configs": [{
                "key": c['ConfigKey'],
                "value": c['ConfigValue'],
                "type": c['ConfigType'],
                "display_name": c['DisplayName'],
                "display_name_ar": c['DisplayNameAr'],
                "description": c['Description'],
                "is_encrypted": c['IsEncrypted'],
                "is_editable": c['IsEditable']
            } for c in configs]
        }
    
    @staticmethod
    def get_configuration(key: str) -> Optional[Dict[str, Any]]:
        """
        Get a single configuration by key.
        
        Args:
            key: Configuration key
        
        Returns:
            Configuration dictionary or None
        """
        config = hardware_config_db.get_config_with_metadata(key)
        if not config:
            return None
        
        return {
            "key": config['ConfigKey'],
            "value": config['ConfigValue'],
            "type": config['ConfigType'],
            "group": config['ConfigGroup'],
            "display_name": config['DisplayName'],
            "display_name_ar": config['DisplayNameAr'],
            "description": config['Description'],
            "is_encrypted": config['IsEncrypted'],
            "is_editable": config['IsEditable'],
            "updated_at": config['UpdatedAt'].isoformat() if config['UpdatedAt'] else None
        }
    
    @staticmethod
    def update_configuration(key: str, value: str, user_id: int) -> Dict[str, Any]:
        """
        Update a single configuration value.
        
        Args:
            key: Configuration key
            value: New value
            user_id: User making the change
        
        Returns:
            Result dictionary with success status
        """
        # Validate the value based on type
        config = hardware_config_db.get_config_with_metadata(key)
        if not config:
            return {
                "success": False,
                "error": f"Configuration '{key}' not found"
            }
        
        if not config['IsEditable']:
            return {
                "success": False,
                "error": f"Configuration '{key}' is not editable"
            }
        
        # Type validation
        error = HardwareConfigService._validate_value(config['ConfigType'], value)
        if error:
            return {
                "success": False,
                "error": error
            }
        
        success = hardware_config_db.update_config(key, value, user_id)
        
        return {
            "success": success,
            "message": f"Configuration '{key}' updated successfully" if success else "Update failed"
        }
    
    @staticmethod
    def update_configurations_batch(updates: List[Dict[str, str]], user_id: int) -> Dict[str, Any]:
        """
        Update multiple configurations at once.
        
        Args:
            updates: List of {key, value} dictionaries
            user_id: User making the changes
        
        Returns:
            Result dictionary with success/error counts
        """
        # Validate all values first
        validated_updates = []
        errors = []
        
        for update in updates:
            key = update.get('key')
            value = update.get('value', '')
            
            config = hardware_config_db.get_config_with_metadata(key)
            if not config:
                errors.append({"key": key, "error": f"Configuration not found"})
                continue
            
            if not config['IsEditable']:
                errors.append({"key": key, "error": "Configuration is not editable"})
                continue
            
            error = HardwareConfigService._validate_value(config['ConfigType'], value)
            if error:
                errors.append({"key": key, "error": error})
                continue
            
            validated_updates.append(update)
        
        # Process validated updates
        if validated_updates:
            result = hardware_config_db.update_configs_batch(validated_updates, user_id)
            result["validation_errors"] = errors
            result["error_count"] += len(errors)
            return result
        
        return {
            "success_count": 0,
            "error_count": len(errors),
            "errors": errors,
            "validation_errors": errors
        }
    
    @staticmethod
    def _validate_value(config_type: str, value: str) -> Optional[str]:
        """
        Validate a configuration value based on its type.
        
        Returns:
            Error message if invalid, None if valid
        """
        if config_type == 'int':
            try:
                int(value)
            except ValueError:
                return f"Value must be an integer, got '{value}'"
        
        elif config_type == 'bool':
            if value.lower() not in ('true', 'false', '1', '0', 'yes', 'no'):
                return f"Value must be true/false, got '{value}'"
        
        elif config_type == 'password':
            # Passwords can be any string, but don't allow updating with masked value
            if value == '********':
                return "Cannot set password to masked value"
        
        return None
    
    @staticmethod
    def test_database_connection() -> Dict[str, Any]:
        """
        Test database connection with current configuration.
        
        Returns:
            Test result dictionary
        """
        # Get current config values (need actual values, not masked)
        from core.deployment_port import (
            DB_SERVER, DB_DATABASE, DB_DRIVER, USE_WINDOWS_AUTH
        )
        
        return hardware_config_db.test_database_connection(
            server=DB_SERVER,
            database=DB_DATABASE,
            driver=DB_DRIVER,
            use_windows_auth=USE_WINDOWS_AUTH
        )
    
    @staticmethod
    def test_smtp_connection() -> Dict[str, Any]:
        """
        Test SMTP connection with current configuration.
        
        Returns:
            Test result dictionary
        """
        from core.deployment_port import (
            SMTP_HOST, SMTP_PORT, SMTP_USE_TLS, SMTP_USE_SSL,
            SMTP_USERNAME, SMTP_PASSWORD
        )
        
        return hardware_config_db.test_smtp_connection(
            host=SMTP_HOST,
            port=SMTP_PORT,
            use_tls=SMTP_USE_TLS,
            use_ssl=SMTP_USE_SSL,
            username=SMTP_USERNAME,
            password=SMTP_PASSWORD
        )
    
    @staticmethod
    def test_custom_database_connection(server: str, database: str, driver: str,
                                         use_windows_auth: bool, username: str = None,
                                         password: str = None) -> Dict[str, Any]:
        """
        Test database connection with custom parameters.
        
        Returns:
            Test result dictionary
        """
        return hardware_config_db.test_database_connection(
            server=server,
            database=database,
            driver=driver,
            use_windows_auth=use_windows_auth,
            username=username,
            password=password
        )
    
    @staticmethod
    def test_custom_smtp_connection(host: str, port: int, use_tls: bool = False,
                                     use_ssl: bool = False, username: str = None,
                                     password: str = None) -> Dict[str, Any]:
        """
        Test SMTP connection with custom parameters.
        
        Returns:
            Test result dictionary
        """
        return hardware_config_db.test_smtp_connection(
            host=host,
            port=port,
            use_tls=use_tls,
            use_ssl=use_ssl,
            username=username,
            password=password
        )
    
    @staticmethod
    def get_deployment_summary() -> Dict[str, Any]:
        """
        Get a summary of current deployment configuration.
        
        Returns:
            Summary dictionary for quick overview
        """
        from core.deployment_port import (
            DEPLOYMENT_MODE, DB_SERVER, DB_DATABASE,
            BACKEND_API_URL, SMTP_HOST, NOTIFICATION_MODE
        )
        
        return {
            "deployment_mode": DEPLOYMENT_MODE,
            "database": {
                "server": DB_SERVER,
                "database": DB_DATABASE
            },
            "network": {
                "api_url": BACKEND_API_URL
            },
            "email": {
                "smtp_host": SMTP_HOST,
                "notification_mode": NOTIFICATION_MODE
            }
        }
