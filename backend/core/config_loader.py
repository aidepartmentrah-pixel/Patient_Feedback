"""
Configuration Loader Module

Loads system configuration from config/db_settings.json with environment
variable override support. This is the single source of truth for all
deployment configuration.

Usage:
    from core.config_loader import load_config, save_config, get_config

    config = get_config()              # Cached config dict
    config = load_config()             # Force reload from disk
    save_config(updated_config)        # Write back to JSON
"""

import json
import os
import logging
import copy
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Path to the JSON config file (relative to backend/)
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_CONFIG_FILE = _CONFIG_DIR / "db_settings.json"

# Cached config (loaded once, refreshable via load_config)
_cached_config: Optional[Dict[str, Any]] = None


def _get_default_config() -> Dict[str, Any]:
    """Return a minimal default configuration if JSON file is missing."""
    return {
        "deployment_mode": "offline",
        "database": {
            "server": "localhost",
            "database": "IncidentManager",
            "driver": "ODBC Driver 17 for SQL Server",
            "use_windows_auth": True,
            "username": "",
            "password": "",
            "trust_server_certificate": True,
        },
        "views": {
            "hr_employees": "VW_HrEmployeeProfileView",
            "patient_admission": "VW_PatientAdmission",
            "doctors": "VW_Doctors",
        },
        "network": {
            "backend_api_url": "http://localhost:8000",
            "backend_port": 8000,
            "backend_host": "127.0.0.1",
            "cors_origins": [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
            ],
        },
        "email": {
            "notification_mode": "mock",
            "smtp_host": "smtp.hospital.local",
            "smtp_port": 25,
            "smtp_use_tls": False,
            "smtp_use_ssl": False,
            "smtp_username": None,
            "smtp_password": None,
            "sender_email": "complaint-system@hospital.local",
            "sender_name": "Hospital Complaint System",
        },
        "config_password": "admin2024",
    }


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides on top of the JSON config.

    Environment variables take precedence over file values.
    Only override if the env var is actually set (not empty).
    """
    def _env(key: str, default: Any = None) -> Optional[str]:
        val = os.environ.get(key)
        return val if val else None

    def _env_bool(key: str, current: bool) -> bool:
        val = os.environ.get(key)
        if val is None:
            return current
        return val.lower() in ("true", "1", "yes")

    def _env_int(key: str, current: int) -> int:
        val = os.environ.get(key)
        if val is None:
            return current
        try:
            return int(val)
        except ValueError:
            return current

    db = config.get("database", {})
    db["server"] = _env("DB_SERVER") or db.get("server", "localhost")
    db["database"] = _env("DB_DATABASE") or db.get("database", "IncidentManager")
    db["driver"] = _env("DB_DRIVER") or db.get("driver", "ODBC Driver 17 for SQL Server")
    db["use_windows_auth"] = _env_bool("USE_WINDOWS_AUTH", db.get("use_windows_auth", True))
    db["username"] = _env("DB_USERNAME") or db.get("username", "")
    db["password"] = _env("DB_PASSWORD") or db.get("password", "")
    db["trust_server_certificate"] = _env_bool(
        "TRUST_SERVER_CERTIFICATE", db.get("trust_server_certificate", True)
    )

    net = config.get("network", {})
    net["backend_host"] = _env("BACKEND_HOST") or net.get("backend_host", "127.0.0.1")
    net["backend_port"] = _env_int("BACKEND_PORT", net.get("backend_port", 8000))

    # CONFIG_PASSWORD override (allows setting via env for automation)
    config["config_password"] = _env("CONFIG_PASSWORD") or config.get("config_password", "admin2024")

    return config


def load_config(force: bool = False) -> Dict[str, Any]:
    """
    Load configuration from db_settings.json with env var overrides.

    Args:
        force: If True, reload even if already cached.

    Returns:
        Configuration dictionary.
    """
    global _cached_config

    if _cached_config is not None and not force:
        return copy.deepcopy(_cached_config)

    config = _get_default_config()

    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                file_config = json.load(f)

            # Deep merge file config into defaults
            _deep_merge(config, file_config)
            logger.info(f"Loaded configuration from {_CONFIG_FILE}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {_CONFIG_FILE}: {e} — using defaults")
        except Exception as e:
            logger.error(f"Error reading {_CONFIG_FILE}: {e} — using defaults")
    else:
        logger.warning(f"Config file not found at {_CONFIG_FILE} — using defaults")
        # Create the file with defaults so it can be edited later
        try:
            _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            save_config(config)
            logger.info(f"Created default config file at {_CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Could not create default config file: {e}")

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    _cached_config = copy.deepcopy(config)
    return copy.deepcopy(config)


def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration to db_settings.json.

    Args:
        config: Full configuration dictionary to save.

    Returns:
        True if successful, False otherwise.
    """
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Remove internal comment/warning keys before saving
        save_data = copy.deepcopy(config)
        save_data.pop("_comment", None)
        save_data.pop("_warning", None)

        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)

        # Invalidate cache so next load picks up changes
        global _cached_config
        _cached_config = None

        logger.info(f"Configuration saved to {_CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration to {_CONFIG_FILE}: {e}")
        return False


def get_config() -> Dict[str, Any]:
    """Get the current configuration (cached). Use load_config(force=True) to reload."""
    return load_config()


def get_config_password() -> str:
    """Get the config access password."""
    config = get_config()
    return config.get("config_password", "admin2024")


def get_config_file_path() -> str:
    """Return the absolute path to the config file (for diagnostics)."""
    return str(_CONFIG_FILE)


def is_test_password_mode() -> bool:
    """
    Return True if the system is configured for test-mode password storage.

    In test mode, passwords are stored as TEMP_HASH_<plain> so they remain
    visible in credential exports. In production mode, bcrypt is used.

    Controlled by db_settings.json → password_mode: "test" | "production"
    Environment variable HCAT_PASSWORD_MODE overrides the file value.
    """
    env_override = os.environ.get("HCAT_PASSWORD_MODE")
    if env_override:
        return env_override.strip().lower() == "test"
    config = get_config()
    return config.get("password_mode", "production").strip().lower() == "test"


def _deep_merge(base: dict, override: dict) -> None:
    """
    Recursively merge `override` into `base` (in-place).
    Lists are replaced, not merged.
    """
    for key, value in override.items():
        if key.startswith("_"):
            continue  # Skip comment keys
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
