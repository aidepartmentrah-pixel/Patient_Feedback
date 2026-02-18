"""
Organization Unit Type Constants
Centralized constants for AdminsrationUnit.Type field values.

These constants represent the Type field in the AdminsrationUnit table:
- Type is stored as INTEGER in the database
- Used throughout the application for org unit classification
- Maps to string representations in APP_UserRoleScope.OrgUnitType

Database Schema:
    AdminsrationUnit.Type (INTEGER) -> org unit level classification
    APP_UserRoleScope.OrgUnitType (VARCHAR) -> string representation
"""

# Organization Unit Type Codes (INTEGER values in AdminsrationUnit.Type)
ORG_TYPE_ADMINISTRATION = 323  # Top-level administration unit
ORG_TYPE_DEPARTMENT = 325      # Mid-level department unit
ORG_TYPE_SECTION = 324         # Leaf-level section unit

# Mapping from Type code to string name
ORG_TYPE_NAME_MAP = {
    ORG_TYPE_ADMINISTRATION: "ADMINISTRATION",
    ORG_TYPE_DEPARTMENT: "DEPARTMENT",
    ORG_TYPE_SECTION: "SECTION",
}

# Reverse mapping from lowercase string name to Type code
ORG_TYPE_NAME_TO_CODE_MAP = {
    "administration": ORG_TYPE_ADMINISTRATION,
    "department": ORG_TYPE_DEPARTMENT,
    "section": ORG_TYPE_SECTION,
}
