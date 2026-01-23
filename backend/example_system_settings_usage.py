"""
Example: How to Use System Settings in Your Backend Code

This file demonstrates various ways to use the system settings infrastructure
in your application code.
"""

from api.services.system_settings_service import SystemSettingsService


# =============================================
# EXAMPLE 1: Get a setting value (type-safe)
# =============================================

def check_complaint_delay(days_since_complaint: int) -> bool:
    """
    Check if a complaint is delayed based on the system setting.
    
    Args:
        days_since_complaint: Number of days since complaint was filed
        
    Returns:
        True if complaint is delayed, False otherwise
    """
    # Get the delay threshold from settings (returns parsed int)
    delay_threshold = SystemSettingsService.get_setting_value("ComplaintDelayDays")
    
    # Now we have a proper integer, not a string!
    is_delayed = days_since_complaint > delay_threshold
    
    return is_delayed


# =============================================
# EXAMPLE 2: Get setting with error handling
# =============================================

def get_max_upload_size() -> int:
    """
    Get maximum upload size, with fallback to default.
    """
    try:
        max_size = SystemSettingsService.get_setting_value("MaxUploadSizeMB")
        return max_size
    except ValueError:
        # Setting not found or invalid, use default
        return 50  # Default: 50 MB


# =============================================
# EXAMPLE 3: Use boolean settings
# =============================================

def is_feature_enabled(feature_key: str) -> bool:
    """
    Check if a feature flag is enabled.
    
    Example usage:
        if is_feature_enabled("EnableMLPredictions"):
            # Run ML prediction
            ...
    """
    try:
        enabled = SystemSettingsService.get_setting_value(feature_key)
        return bool(enabled)
    except ValueError:
        # If setting doesn't exist or is invalid, assume disabled
        return False


# =============================================
# EXAMPLE 4: Use JSON settings for complex config
# =============================================

def get_escalation_rules() -> dict:
    """
    Get escalation rules from JSON setting.
    
    Example setting value:
    {
      "level1": 3,   # Escalate after 3 days
      "level2": 7,   # Escalate after 7 days
      "level3": 14   # Escalate after 14 days
    }
    """
    try:
        rules = SystemSettingsService.get_setting_value("EscalationRules")
        return rules  # Already parsed as dict
    except ValueError:
        # Return default rules
        return {
            "level1": 3,
            "level2": 7,
            "level3": 14
        }


# =============================================
# EXAMPLE 5: Update setting programmatically
# =============================================

def auto_adjust_delay_threshold(new_threshold: int, user_id: int) -> None:
    """
    Programmatically update the delay threshold.
    
    This could be called by an admin function or automated process.
    """
    SystemSettingsService.update_setting(
        key="ComplaintDelayDays",
        value=str(new_threshold),  # Must be string
        updated_by_user_id=user_id
    )
    print(f"Delay threshold updated to {new_threshold} days")


# =============================================
# EXAMPLE 6: Cache settings for performance
# =============================================

class SettingsCache:
    """
    Simple cache for frequently accessed settings.
    In production, consider using Redis or similar.
    """
    _cache = {}
    
    @classmethod
    def get_cached_setting(cls, key: str, ttl_seconds: int = 300):
        """
        Get a setting with simple in-memory caching.
        
        Args:
            key: Setting key
            ttl_seconds: Time to live in seconds (default 5 minutes)
        """
        import time
        
        # Check if cached and not expired
        if key in cls._cache:
            value, timestamp = cls._cache[key]
            if time.time() - timestamp < ttl_seconds:
                return value
        
        # Fetch from database
        value = SystemSettingsService.get_setting_value(key)
        cls._cache[key] = (value, time.time())
        
        return value
    
    @classmethod
    def clear_cache(cls, key: str = None):
        """Clear cache for one or all settings."""
        if key:
            cls._cache.pop(key, None)
        else:
            cls._cache.clear()


# =============================================
# EXAMPLE 7: Validate complaint in service layer
# =============================================

def process_complaint(complaint_data: dict) -> dict:
    """
    Process a complaint and check if it's delayed.
    
    This demonstrates integration in a service layer.
    """
    from datetime import datetime, timedelta
    
    # Get complaint date
    complaint_date = datetime.fromisoformat(complaint_data['complaint_date'])
    days_elapsed = (datetime.now() - complaint_date).days
    
    # Check if delayed using system setting
    delay_threshold = SystemSettingsService.get_setting_value("ComplaintDelayDays")
    
    is_delayed = days_elapsed > delay_threshold
    
    return {
        "complaint_id": complaint_data['id'],
        "days_elapsed": days_elapsed,
        "delay_threshold": delay_threshold,
        "is_delayed": is_delayed,
        "status": "DELAYED" if is_delayed else "ON_TIME"
    }


# =============================================
# EXAMPLE 8: Get all settings for admin page
# =============================================

def get_settings_for_admin_page():
    """
    Get all settings formatted for the admin/settings page.
    """
    all_settings = SystemSettingsService.get_all_settings()
    
    # Group by category (you can add categories later)
    categorized = {
        "timing": [],
        "limits": [],
        "features": []
    }
    
    for setting in all_settings:
        # Categorize based on key prefix or name
        if "Days" in setting['key'] or "Time" in setting['key']:
            categorized["timing"].append(setting)
        elif "Max" in setting['key'] or "Limit" in setting['key']:
            categorized["limits"].append(setting)
        elif "Enable" in setting['key']:
            categorized["features"].append(setting)
    
    return categorized


# =============================================
# EXAMPLE 9: Settings validation before use
# =============================================

def safe_get_setting(key: str, default_value: any, expected_type: type):
    """
    Safely get a setting with type checking and default value.
    
    Args:
        key: Setting key
        default_value: Value to return if setting not found
        expected_type: Expected Python type (int, bool, str, dict)
    """
    try:
        value = SystemSettingsService.get_setting_value(key)
        
        # Type check
        if not isinstance(value, expected_type):
            print(f"Warning: Setting '{key}' has unexpected type. Using default.")
            return default_value
        
        return value
    
    except ValueError as e:
        print(f"Warning: Could not get setting '{key}': {e}. Using default.")
        return default_value


# =============================================
# USAGE EXAMPLES
# =============================================

if __name__ == "__main__":
    # Example 1: Check if complaint is delayed
    is_delayed = check_complaint_delay(days_since_complaint=20)
    print(f"Complaint is delayed: {is_delayed}")
    
    # Example 2: Get max upload size with fallback
    max_size = get_max_upload_size()
    print(f"Max upload size: {max_size} MB")
    
    # Example 3: Check feature flag
    ml_enabled = is_feature_enabled("EnableMLPredictions")
    print(f"ML predictions enabled: {ml_enabled}")
    
    # Example 4: Get escalation rules
    rules = get_escalation_rules()
    print(f"Escalation rules: {rules}")
    
    # Example 6: Use cached settings
    cached_value = SettingsCache.get_cached_setting("ComplaintDelayDays")
    print(f"Cached delay days: {cached_value}")
    
    # Example 8: Get categorized settings
    admin_settings = get_settings_for_admin_page()
    print(f"Settings categories: {list(admin_settings.keys())}")
    
    # Example 9: Safe get with default
    safe_value = safe_get_setting("ComplaintDelayDays", 14, int)
    print(f"Safe delay days: {safe_value}")
