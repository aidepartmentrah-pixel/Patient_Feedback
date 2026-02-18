"""
PHASE K — Migration Text Builder Utility

Pure text transformation functions for building new case text fields from legacy data.

This module contains ONLY text construction logic.
NO database access. NO I/O. Pure functions only.

These functions are deterministic: identical inputs produce identical outputs.
"""

from typing import Dict, List, Any, Optional


# ==================== HELPER FUNCTIONS ====================

def non_empty(parts: List[Optional[str]]) -> List[str]:
    """
    Filter out None and empty string values from a list of text parts.
    
    Args:
        parts: List of text parts (may contain None or empty strings)
    
    Returns:
        List of non-empty strings only
    
    Example:
        >>> non_empty(["hello", None, "", "world"])
        ["hello", "world"]
    """
    return [p for p in parts if p]


def join_single(parts: List[str]) -> str:
    """
    Join text parts with single newline separator.
    
    Args:
        parts: List of text strings to join
    
    Returns:
        Joined string with single newline between parts
    
    Example:
        >>> join_single(["Line 1", "Line 2"])
        "Line 1\\nLine 2"
    """
    return "\n".join(parts)


def join_double(parts: List[str]) -> str:
    """
    Join text parts with double newline separator.
    
    Args:
        parts: List of text strings to join
    
    Returns:
        Joined string with double newline between parts
    
    Example:
        >>> join_double(["Block 1", "Block 2"])
        "Block 1\\n\\nBlock 2"
    """
    return "\n\n".join(parts)


# ==================== FIELD BUILDERS ====================

def build_complaint_content(case_row: Dict[str, Any], request_row: Dict[str, Any]) -> str:
    """
    Build complaint_content field from legacy case and request data.
    
    Sources (in order):
        1. Case Description
        2. Requester Note
    
    Format:
        [Case Description]
        {text}
        
        [Requester Note]
        {text}
    
    Args:
        case_row: Dict with "Description" key
        request_row: Dict with "Note" key
    
    Returns:
        Formatted complaint content string (empty if no data)
    """
    blocks = []
    
    # Add case description if present
    case_desc = case_row.get("Description")
    if case_desc:
        blocks.append("[Case Description]\n" + case_desc)
    
    # Add requester note if present
    req_note = request_row.get("Note")
    if req_note:
        blocks.append("[Requester Note]\n" + req_note)
    
    return join_double(non_empty(blocks))


def build_immediate_action(actions: List[Dict[str, Any]]) -> str:
    """
    Build immediate_action field from FIRST action record only.
    
    Sources (from first action, in order):
        1. Description → [Action Description]
        2. SectionNote → [Section Note]
        3. SelectionNote → [Selection Note]
        4. ProblemReason → [Problem Reason]
    
    Format:
        [Action Description]
        {text}
        
        [Section Note]
        {text}
        
        [Selection Note]
        {text}
        
        [Problem Reason]
        {text}
    
    Args:
        actions: List of action dicts (ordered ASC by date)
    
    Returns:
        Formatted immediate action string (empty if no actions)
    """
    if not actions:
        return ""
    
    first_action = actions[0]
    blocks = []
    
    # Field order is fixed
    field_mapping = [
        ("Description", "[Action Description]"),
        ("SectionNote", "[Section Note]"),
        ("SelectionNote", "[Selection Note]"),
        ("ProblemReason", "[Problem Reason]")
    ]
    
    for field_key, label in field_mapping:
        value = first_action.get(field_key)
        if value:
            blocks.append(f"{label}\n{value}")
    
    return join_double(non_empty(blocks))


def build_actions_taken(actions: List[Dict[str, Any]]) -> str:
    """
    Build actions_taken field from ALL actions EXCEPT the first.
    
    Each action formatted as:
        [Action — YYYY-MM-DD HH:MM]
        Description: {text}
        Note: {text}
        Department Note: {text}
        Section Note: {text}
        Policies: {text}
    
    Field order (per action):
        1. Description
        2. Note
        3. DepartmentNote
        4. SectionNote
        5. GoverningPolicies
    
    Args:
        actions: List of action dicts (ordered ASC by date)
    
    Returns:
        Formatted actions taken string (empty if 0-1 actions)
    """
    if len(actions) <= 1:
        return ""
    
    # Skip first action (that's immediate_action)
    remaining_actions = actions[1:]
    
    action_blocks = []
    
    for action in remaining_actions:
        lines = []
        
        # Format header with datetime
        datetime_str = action.get("DateAndTimeCreated", "")
        if datetime_str:
            # Format: [Action — YYYY-MM-DD HH:MM]
            # DateAndTimeCreated comes as "YYYY-MM-DD HH:MM:SS" from DB
            # Extract first 16 chars to get "YYYY-MM-DD HH:MM"
            formatted_date = datetime_str[:16] if len(datetime_str) >= 16 else datetime_str
            lines.append(f"[Action — {formatted_date}]")
        else:
            lines.append("[Action — Date Unknown]")
        
        # Field order is fixed
        field_mapping = [
            ("Description", "Description"),
            ("Note", "Note"),
            ("DepartmentNote", "Department Note"),
            ("SectionNote", "Section Note"),
            ("GoverningPolicies", "Policies")
        ]
        
        for field_key, label in field_mapping:
            value = action.get(field_key)
            if value:
                lines.append(f"{label}: {value}")
        
        # Join lines within action with single newline
        action_blocks.append(join_single(non_empty(lines)))
    
    # Join actions with double newline
    return join_double(non_empty(action_blocks))


# ==================== MAIN BUILDER ====================

def build_migration_texts(
    case_row: Dict[str, Any],
    request_row: Dict[str, Any],
    actions: List[Dict[str, Any]]
) -> Dict[str, str]:
    """
    Build all three migration text fields from legacy case data.
    
    This is the main entry point for migration text transformation.
    
    Args:
        case_row: Legacy IncidentRequestCase data dict
            Required keys: Description
        
        request_row: Legacy IncidentRequest data dict
            Required keys: Note
        
        actions: List of IncidentRequestCaseAction data dicts
            Must be ordered ASC by DateAndTimeCreated, UniqueID
            Each may contain: Description, Note, SectionNote, SelectionNote,
                             DepartmentNote, ProblemReason, GoverningPolicies,
                             DateAndTimeCreated
    
    Returns:
        Dict with three string fields:
        {
            "complaint_content": str,
            "immediate_action": str,
            "actions_taken": str
        }
        
        All values are strings (never None). Empty inputs produce empty strings.
    
    Guarantees:
        - Deterministic: identical inputs always produce identical outputs
        - No database access
        - No I/O operations
        - Thread-safe
    
    Example:
        >>> case = {"Description": "Patient fell"}
        >>> request = {"Note": "Reported by nurse"}
        >>> actions = []
        >>> result = build_migration_texts(case, request, actions)
        >>> result["complaint_content"]
        "[Case Description]\\nPatient fell\\n\\n[Requester Note]\\nReported by nurse"
    """
    return {
        "complaint_content": build_complaint_content(case_row, request_row),
        "immediate_action": build_immediate_action(actions),
        "actions_taken": build_actions_taken(actions)
    }
