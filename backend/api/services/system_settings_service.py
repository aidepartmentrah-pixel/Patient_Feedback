"""
System Settings Service Layer
Business logic for system-wide configuration settings.
"""

import json
from typing import Dict, List, Any, Optional
from ..db_layer import system_settings_db


class SystemSettingsService:
    """Service class for system settings operations."""
    
    @staticmethod
    def parse_setting_value(setting_type: str, raw_value: str) -> Any:
        """
        Parse a setting value based on its type.
        
        Args:
            setting_type: The type of the setting ('int', 'bool', 'string', 'json')
            raw_value: The raw string value from the database
            
        Returns:
            Parsed value in the appropriate Python type
            
        Raises:
            ValueError: If the value cannot be parsed as the specified type
        """
        if setting_type == 'int':
            try:
                return int(raw_value)
            except ValueError:
                raise ValueError(f"Cannot parse '{raw_value}' as integer")
        
        elif setting_type == 'bool':
            lower_value = raw_value.lower()
            if lower_value in ('true', '1', 'yes'):
                return True
            elif lower_value in ('false', '0', 'no'):
                return False
            else:
                raise ValueError(f"Cannot parse '{raw_value}' as boolean. Use 'true' or 'false'")
        
        elif setting_type == 'json':
            try:
                return json.loads(raw_value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {str(e)}")
        
        elif setting_type == 'string':
            return raw_value
        
        else:
            raise ValueError(f"Unknown setting type: {setting_type}")
    
    @staticmethod
    def validate_setting_value(setting_type: str, value: str) -> None:
        """
        Validate a setting value without parsing it.
        
        Args:
            setting_type: The type of the setting
            value: The value to validate
            
        Raises:
            ValueError: If the value is invalid for the given type
        """
        try:
            SystemSettingsService.parse_setting_value(setting_type, value)
        except ValueError as e:
            raise ValueError(f"Validation failed: {str(e)}")
    
    @staticmethod
    def get_all_settings() -> List[Dict[str, Any]]:
        """
        Fetch all system settings.
        
        Returns:
            List of settings with parsed values
        """
        settings = system_settings_db.get_all_system_settings()
        
        # Parse values for each setting
        for setting in settings:
            try:
                setting['parsed_value'] = SystemSettingsService.parse_setting_value(
                    setting['type'],
                    setting['value']
                )
            except ValueError as e:
                # If parsing fails, keep raw value and add error
                setting['parsed_value'] = None
                setting['parse_error'] = str(e)
        
        return settings
    
    @staticmethod
    def get_setting(key: str) -> Dict[str, Any]:
        """
        Fetch a single system setting by key.
        
        Args:
            key: The setting key
            
        Returns:
            Setting dictionary with parsed value
            
        Raises:
            ValueError: If setting not found
        """
        setting = system_settings_db.get_system_setting(key)
        
        if not setting:
            raise ValueError(f"Setting '{key}' not found")
        
        # Parse the value
        try:
            setting['parsed_value'] = SystemSettingsService.parse_setting_value(
                setting['type'],
                setting['value']
            )
        except ValueError as e:
            setting['parsed_value'] = None
            setting['parse_error'] = str(e)
        
        return setting
    
    @staticmethod
    def get_setting_value(key: str) -> Any:
        """
        Get the parsed value of a setting directly.
        
        Args:
            key: The setting key
            
        Returns:
            Parsed value (int, bool, str, dict, etc.)
            
        Raises:
            ValueError: If setting not found or cannot be parsed
        """
        setting = SystemSettingsService.get_setting(key)
        
        if 'parse_error' in setting:
            raise ValueError(f"Cannot parse setting '{key}': {setting['parse_error']}")
        
        return setting['parsed_value']
    
    @staticmethod
    def update_setting(
        key: str,
        value: str,
        updated_by_user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update a system setting value.
        
        Args:
            key: The setting key to update
            value: The new value (as string)
            updated_by_user_id: ID of the user making the update
            
        Returns:
            Updated setting dictionary
            
        Raises:
            ValueError: If setting not found or validation fails
        """
        # First get the setting to know its type
        current_setting = system_settings_db.get_system_setting(key)
        
        if not current_setting:
            raise ValueError(f"Setting '{key}' not found")
        
        # Validate the new value
        SystemSettingsService.validate_setting_value(
            current_setting['type'],
            value
        )
        
        # Update in database
        updated_setting = system_settings_db.update_system_setting(
            key=key,
            value=value,
            updated_by_user_id=updated_by_user_id
        )
        
        # Parse and return
        updated_setting['parsed_value'] = SystemSettingsService.parse_setting_value(
            updated_setting['type'],
            updated_setting['value']
        )
        
        return updated_setting
    
    @staticmethod
    def create_setting(
        key: str,
        value: str,
        setting_type: str,
        description: Optional[str] = None,
        updated_by_user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a new system setting.
        
        Args:
            key: Unique setting key
            value: Setting value (as string)
            setting_type: Type ('int', 'bool', 'string', 'json')
            description: Optional description
            updated_by_user_id: ID of the user creating the setting
            
        Returns:
            Created setting dictionary
            
        Raises:
            ValueError: If key exists or validation fails
        """
        # Validate the value
        SystemSettingsService.validate_setting_value(setting_type, value)
        
        # Create in database
        new_setting = system_settings_db.create_system_setting(
            key=key,
            value=value,
            setting_type=setting_type,
            description=description,
            updated_by_user_id=updated_by_user_id
        )
        
        # Parse and return
        new_setting['parsed_value'] = SystemSettingsService.parse_setting_value(
            new_setting['type'],
            new_setting['value']
        )
        
        return new_setting
    
    @staticmethod
    def delete_setting(key: str) -> bool:
        """
        Delete a system setting.
        
        Args:
            key: The setting key to delete
            
        Returns:
            True if deleted successfully
        """
        return system_settings_db.delete_system_setting(key)
