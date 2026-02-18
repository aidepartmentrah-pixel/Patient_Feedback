"""
Follow-Up Actions Database Layer
SQL queries for action CRUD, filtering, and derived field computation.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from core.database import get_connection


def create_follow_up_action(
    action_title: str,
    action_description: Optional[str] = None,
    incident_case_id: Optional[int] = None,
    seasonal_report_id: Optional[int] = None,
    department_id: Optional[int] = None,
    assigned_to: Optional[str] = None,
    priority: str = 'medium',
    status: str = 'pending',
    due_date: str = None,
    notes: Optional[str] = None,
    created_by_user_id: int = None
) -> Optional[Dict[str, Any]]:
    """
    Create a new follow-up action using EXISTING schema.
    Maps new parameters to existing columns: IsDone, DateSubmitted.
    Ignores: department_id, assigned_to, priority, notes (not in schema).
    
    Returns:
        Created action dict with ID
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Map status to IsDone (bit field)
        is_done = 1 if status == 'completed' else 0
        
        query = """
            INSERT INTO dbo.APP_ActionItem (
                ActionTitle,
                ActionDescription,
                IncidentRequestCaseID,
                SeasonalReportID,
                DueDate,
                IsDone,
                CreatedAt,
                CreatedByUserID
            )
            OUTPUT INSERTED.ActionItemID
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(query, (
            action_title,
            action_description,
            incident_case_id,
            seasonal_report_id,
            due_date,
            is_done,
            datetime.now(),
            created_by_user_id or 1
        ))
        
        # Get the inserted ID
        row = cursor.fetchone()
        new_id = row[0] if row else None
        
        conn.commit()
        
        if new_id:
            return get_follow_up_action_by_id(new_id)
        
        return None
    
    finally:
        cursor.close()
        conn.close()


def get_follow_up_actions(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    department: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    include_completed: bool = False
) -> Dict[str, Any]:
    """
    Fetch filtered follow-up actions using EXISTING schema.
    Maps IsDone (bit) to status field.
    Note: priority, department, assignedTo, notes not available in current schema.
    
    Returns:
        Dict with actions array and global statistics
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build WHERE clause dynamically
        where_clauses = []
        params = []
        
        # Status filter - map to IsDone
        if status and status != 'all':
            if status == 'completed':
                where_clauses.append("a.IsDone = 1")
            elif status == 'pending':
                where_clauses.append("a.IsDone = 0")
            # 'delayed' not supported in current schema, treat as pending
        elif not include_completed:
            # Default: exclude completed unless explicitly included
            where_clauses.append("a.IsDone = 0")
        
        # Priority filter - not available in schema, ignore
        # Department filter - not available in schema, ignore
        
        # Date range filter
        if from_date:
            where_clauses.append("CAST(a.DueDate AS DATE) >= ?")
            params.append(from_date)
        
        if to_date:
            where_clauses.append("CAST(a.DueDate AS DATE) <= ?")
            params.append(to_date)
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Main query - using EXISTING columns only
        query = f"""
            SELECT
                a.ActionItemID AS id,
                a.ActionTitle AS actionTitle,
                a.ActionDescription AS actionDescription,
                CASE 
                    WHEN a.IncidentRequestCaseID IS NOT NULL THEN 'incident_explanation'
                    WHEN a.SeasonalReportID IS NOT NULL THEN 'seasonal_explanation'
                    ELSE 'manual'
                END AS sourceType,
                COALESCE(CAST(a.IncidentRequestCaseID AS NVARCHAR(50)), 
                         CAST(a.SeasonalReportID AS NVARCHAR(50)), '') AS sourceId,
                CAST(a.DueDate AS DATE) AS dueDate,
                CAST(a.DateSubmitted AS DATE) AS completedDate,
                a.IsDone AS isDone,
                a.CreatedAt AS createdAt,
                a.CreatedByUserID AS createdByUserId,
                -- Derived fields
                CASE 
                    WHEN CAST(a.DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                         AND a.IsDone = 0
                    THEN 1 
                    ELSE 0 
                END AS isOverdue,
                DATEDIFF(DAY, CAST(GETDATE() AS DATE), CAST(a.DueDate AS DATE)) AS daysRemaining,
                CASE 
                    WHEN CAST(a.DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                         AND a.IsDone = 0
                    THEN DATEDIFF(DAY, CAST(a.DueDate AS DATE), CAST(GETDATE() AS DATE))
                    ELSE 0
                END AS daysOverdue
            FROM dbo.APP_ActionItem a
            WHERE {where_sql}
            ORDER BY a.DueDate ASC
        """
        
        cursor.execute(query, params)
        actions = []
        
        for row in cursor.fetchall():
            is_done = bool(row[7])
            actions.append({
                'id': row[0],
                'actionTitle': row[1],
                'actionDescription': row[2],
                'sourceType': row[3],
                'sourceId': row[4],
                'departmentId': None,  # Not in schema
                'assignedTo': None,  # Not in schema
                'priority': 'medium',  # Default, not in schema
                'status': 'completed' if is_done else 'pending',
                'dueDate': row[5].isoformat() if row[5] else None,
                'completedDate': row[6].isoformat() if row[6] else None,
                'notes': None,  # Not in schema
                'createdAt': row[8].isoformat() if row[8] else None,
                'createdByUserId': row[9],
                'lastUpdatedAt': row[8].isoformat() if row[8] else None,  # Use CreatedAt
                'lastUpdatedByUserId': row[9],  # Use CreatedByUserID
                'isOverdue': bool(row[10]),
                'daysRemaining': row[11],
                'daysOverdue': row[12]
            })
        
        # Get global statistics using IsDone
        stats_query = """
            SELECT
                COUNT(CASE WHEN IsDone = 0 THEN 1 END) AS actionsToTake,
                COUNT(CASE WHEN IsDone = 0 
                           AND CAST(DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                      THEN 1 END) AS overdue,
                COUNT(CASE WHEN IsDone = 1 THEN 1 END) AS completed
            FROM dbo.APP_ActionItem
        """
        
        cursor.execute(stats_query)
        stats_row = cursor.fetchone()
        
        statistics = {
            'actionsToTake': stats_row[0] if stats_row[0] else 0,
            'overdue': stats_row[1] if stats_row[1] else 0,
            'completed': stats_row[2] if stats_row[2] else 0
        }
        
        return {
            'actions': actions,
            'total': len(actions),
            'statistics': statistics
        }
    
    finally:
        cursor.close()
        conn.close()


def get_follow_up_action_by_id(action_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch single action by ID using EXISTING schema.
    
    Returns:
        Action dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT
                a.ActionItemID AS id,
                a.ActionTitle AS actionTitle,
                a.ActionDescription AS actionDescription,
                CASE 
                    WHEN a.IncidentRequestCaseID IS NOT NULL THEN 'incident_explanation'
                    WHEN a.SeasonalReportID IS NOT NULL THEN 'seasonal_explanation'
                    ELSE 'manual'
                END AS sourceType,
                COALESCE(CAST(a.IncidentRequestCaseID AS NVARCHAR(50)), 
                         CAST(a.SeasonalReportID AS NVARCHAR(50)), '') AS sourceId,
                CAST(a.DueDate AS DATE) AS dueDate,
                CAST(a.DateSubmitted AS DATE) AS completedDate,
                a.IsDone AS isDone,
                a.CreatedAt AS createdAt,
                a.CreatedByUserID AS createdByUserId,
                CASE 
                    WHEN CAST(a.DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                         AND a.IsDone = 0
                    THEN 1 
                    ELSE 0 
                END AS isOverdue,
                DATEDIFF(DAY, CAST(GETDATE() AS DATE), CAST(a.DueDate AS DATE)) AS daysRemaining,
                CASE 
                    WHEN CAST(a.DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                         AND a.IsDone = 0
                    THEN DATEDIFF(DAY, CAST(a.DueDate AS DATE), CAST(GETDATE() AS DATE))
                    ELSE 0
                END AS daysOverdue
            FROM dbo.APP_ActionItem a
            WHERE a.ActionItemID = ?
        """
        
        cursor.execute(query, (action_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        is_done = bool(row[7])
        
        return {
            'id': row[0],
            'actionTitle': row[1],
            'actionDescription': row[2],
            'sourceType': row[3],
            'sourceId': row[4],
            'departmentId': None,
            'assignedTo': None,
            'priority': 'medium',
            'status': 'completed' if is_done else 'pending',
            'dueDate': row[5].isoformat() if row[5] else None,
            'completedDate': row[6].isoformat() if row[6] else None,
            'notes': None,
            'createdAt': row[8].isoformat() if row[8] else None,
            'createdByUserId': row[9],
            'lastUpdatedAt': row[8].isoformat() if row[8] else None,
            'lastUpdatedByUserId': row[9],
            'isOverdue': bool(row[10]),
            'daysRemaining': row[11],
            'daysOverdue': row[12]
        }
    
    finally:
        cursor.close()
        conn.close()


def update_follow_up_action(
    action_id: int,
    due_date: Optional[str] = None,
    assigned_to: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    last_updated_by_user_id: int = None
) -> Optional[Dict[str, Any]]:
    """
    Update action fields using EXISTING schema.
    Only DueDate and status->IsDone are updateable.
    Ignores: assigned_to, priority, notes (not in schema).
    
    Returns:
        Updated action dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build UPDATE clause dynamically using EXISTING columns only
        update_fields = []
        params = []
        
        if due_date:
            update_fields.append("DueDate = ?")
            params.append(due_date)
        
        # Map status to IsDone
        if status:
            is_done = 1 if status == 'completed' else 0
            update_fields.append("IsDone = ?")
            params.append(is_done)
            
            # If completing, set DateSubmitted
            if status == 'completed':
                update_fields.append("DateSubmitted = ?")
                params.append(datetime.now().strftime('%Y-%m-%d'))
        
        if not update_fields:
            # No updates to make
            return get_follow_up_action_by_id(action_id)
        
        params.append(action_id)
        
        update_sql = ", ".join(update_fields)
        
        query = f"""
            UPDATE dbo.APP_ActionItem
            SET {update_sql}
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, params)
        conn.commit()
        
        # Fetch updated action
        return get_follow_up_action_by_id(action_id)
    
    finally:
        cursor.close()
        conn.close()


def complete_follow_up_action(
    action_id: int,
    completion_notes: Optional[str] = None,
    completed_date: Optional[str] = None,
    last_updated_by_user_id: int = None
) -> Optional[Dict[str, Any]]:
    """
    Mark action as completed using EXISTING schema.
    Sets IsDone=1 and DateSubmitted.
    
    Returns:
        Updated action dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if action exists
        current_action = get_follow_up_action_by_id(action_id)
        if not current_action:
            return None
        
        # Set completed date
        comp_date = completed_date or datetime.now().strftime('%Y-%m-%d')
        
        query = """
            UPDATE dbo.APP_ActionItem
            SET
                IsDone = 1,
                DateSubmitted = ?
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, (comp_date, action_id))
        conn.commit()
        
        # Fetch updated action
        return get_follow_up_action_by_id(action_id)
    
    finally:
        cursor.close()
        conn.close()


def reopen_follow_up_action(
    action_id: int,
    reopen_reason: str,
    new_due_date: Optional[str] = None,
    last_updated_by_user_id: int = None
) -> Optional[Dict[str, Any]]:
    """
    Reopen a completed action using EXISTING schema.
    Sets IsDone=0 and clears DateSubmitted.
    
    Returns:
        Updated action dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Get current action
        current_action = get_follow_up_action_by_id(action_id)
        if not current_action:
            return None
        
        # Determine new due date
        due_date = new_due_date or datetime.now().strftime('%Y-%m-%d')
        
        query = """
            UPDATE dbo.APP_ActionItem
            SET
                IsDone = 0,
                DueDate = ?,
                DateSubmitted = NULL
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, (due_date, action_id))
        conn.commit()
        
        # Fetch updated action
        return get_follow_up_action_by_id(action_id)
    
    finally:
        cursor.close()
        conn.close()


def get_action_history(action_id: int) -> List[Dict[str, Any]]:
    """
    Get change history for an action.
    Note: Notes field not in current schema, returns basic history.
    
    Returns:
        List of history entries
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        action = get_follow_up_action_by_id(action_id)
        if not action:
            return []
        
        # Without Notes field, return basic creation info
        history = [{
            'timestamp': action['createdAt'],
            'userId': action['createdByUserId'],
            'action': 'Created',
            'details': f"Action created: {action['actionTitle']}"
        }]
        
        if action['status'] == 'completed' and action['completedDate']:
            history.append({
                'timestamp': action['completedDate'],
                'userId': action['createdByUserId'],
                'action': 'Completed',
                'details': 'Action marked as completed'
            })
        
        return history
    
    finally:
        cursor.close()
        conn.close()


def get_calendar_actions(
    year: int,
    month: int,
    department: Optional[str] = None,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get actions grouped by date for calendar view using EXISTING schema.
    
    Returns:
        Dict with actions grouped by date
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build WHERE clause
        where_clauses = []
        params = []
        
        # Filter by year and month
        where_clauses.append("YEAR(a.DueDate) = ? AND MONTH(a.DueDate) = ?")
        params.extend([year, month])
        
        # Status filter using IsDone
        if status and status != 'all':
            if status == 'completed':
                where_clauses.append("a.IsDone = 1")
            elif status == 'pending':
                where_clauses.append("a.IsDone = 0")
        else:
            where_clauses.append("a.IsDone = 0")  # Default: exclude completed
        
        # Department filter not available in schema
        
        where_sql = " AND ".join(where_clauses)
        
        query = f"""
            SELECT
                CAST(a.DueDate AS DATE) AS dueDate,
                a.ActionItemID AS id,
                a.ActionTitle AS actionTitle,
                a.IsDone AS isDone,
                CASE 
                    WHEN CAST(a.DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                         AND a.IsDone = 0
                    THEN 1 
                    ELSE 0 
                END AS isOverdue
            FROM dbo.APP_ActionItem a
            WHERE {where_sql}
            ORDER BY a.DueDate ASC
        """
        
        cursor.execute(query, params)
        
        # Group actions by date
        calendar_data = {}
        
        for row in cursor.fetchall():
            date_str = row[0].isoformat() if row[0] else None
            
            if date_str not in calendar_data:
                calendar_data[date_str] = []
            
            is_done = bool(row[3])
            calendar_data[date_str].append({
                'id': row[1],
                'actionTitle': row[2],
                'priority': 'medium',
                'status': 'completed' if is_done else 'pending',
                'departmentId': None,
                'assignedTo': None,
                'isOverdue': bool(row[4])
            })
        
        return {
            'year': year,
            'month': month,
            'calendar': calendar_data
        }
    
    finally:
        cursor.close()
        conn.close()


def bulk_complete_actions(
    action_ids: List[int],
    completion_notes: Optional[str] = None,
    completed_date: Optional[str] = None,
    last_updated_by_user_id: int = None
) -> Dict[str, Any]:
    """
    Mark multiple actions as completed.
    
    Returns:
        Dict with success count and failed IDs
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        completed_count = 0
        failed_ids = []
        
        for action_id in action_ids:
            try:
                # Get current action
                current_action = get_follow_up_action_by_id(action_id)
                if not current_action:
                    failed_ids.append({'id': action_id, 'reason': 'Action not found'})
                    continue
                
                # Skip if already completed
                if current_action['status'] == 'completed':
                    failed_ids.append({'id': action_id, 'reason': 'Already completed'})
                    continue
                
                # Set completed date
                comp_date = completed_date or datetime.now().strftime('%Y-%m-%d')
                
                query = """
                    UPDATE dbo.APP_ActionItem
                    SET
                        IsDone = 1,
                        DateSubmitted = ?
                    WHERE ActionItemID = ?
                """
                
                cursor.execute(query, (comp_date, action_id))
                
                completed_count += 1
            
            except Exception as e:
                failed_ids.append({'id': action_id, 'reason': str(e)})
        
        conn.commit()
        
        return {
            'successCount': completed_count,
            'failedCount': len(failed_ids),
            'failedIds': failed_ids
        }
    
    finally:
        cursor.close()
        conn.close()


def bulk_delay_actions(
    action_ids: List[int],
    delay_days: int,
    reason: Optional[str] = None,
    last_updated_by_user_id: int = None
) -> Dict[str, Any]:
    """
    Delay multiple actions by specified days.
    
    Returns:
        Dict with success count and failed IDs
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        from datetime import timedelta
        
        delayed_count = 0
        failed_ids = []
        
        for action_id in action_ids:
            try:
                # Get current action
                current_action = get_follow_up_action_by_id(action_id)
                if not current_action:
                    failed_ids.append({'id': action_id, 'reason': 'Action not found'})
                    continue
                
                # Skip if completed
                if current_action['status'] == 'completed':
                    failed_ids.append({'id': action_id, 'reason': 'Cannot delay completed action'})
                    continue
                
                # Calculate new due date
                due_date = datetime.fromisoformat(current_action['dueDate'])
                new_due_date = (due_date + timedelta(days=delay_days)).strftime('%Y-%m-%d')
                
                query = """
                    UPDATE dbo.APP_ActionItem
                    SET DueDate = ?
                    WHERE ActionItemID = ?
                """
                
                cursor.execute(query, (new_due_date, action_id))
                
                delayed_count += 1
            
            except Exception as e:
                failed_ids.append({'id': action_id, 'reason': str(e)})
        
        conn.commit()
        
        return {
            'successCount': delayed_count,
            'failedCount': len(failed_ids),
            'failedIds': failed_ids
        }
    
    finally:
        cursor.close()
        conn.close()


def bulk_update_actions(
    action_ids: List[int],
    assigned_to: Optional[str] = None,
    priority: Optional[str] = None,
    department_id: Optional[int] = None,
    last_updated_by_user_id: int = None
) -> Dict[str, Any]:
    """
    Update multiple actions with same values.
    NOTE: assigned_to, priority, department_id not in current schema - operation not supported.
    
    Returns:
        Dict with success count and failed IDs
    """
    # These fields don't exist in current schema
    return {
        'successCount': 0,
        'failedCount': len(action_ids),
        'failedIds': [{'id': aid, 'reason': 'Bulk update not supported with current schema'} for aid in action_ids]
    }


def delay_follow_up_action(
    action_id: int,
    delay_days: int,
    reason: Optional[str] = None,
    last_updated_by_user_id: int = None
) -> Optional[Dict[str, Any]]:
    """
    Delay action by specified days using EXISTING schema.
    
    Returns:
        Updated action dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Get current action to calculate new due date
        current_action = get_follow_up_action_by_id(action_id)
        if not current_action:
            return None
        
        # Parse current due date
        from datetime import timedelta
        due_date = datetime.fromisoformat(current_action['dueDate'])
        new_due_date = (due_date + timedelta(days=delay_days)).strftime('%Y-%m-%d')
        
        query = """
            UPDATE dbo.APP_ActionItem
            SET DueDate = ?
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, (new_due_date, action_id))
        conn.commit()
        
        # Fetch updated action
        return get_follow_up_action_by_id(action_id)
    
    finally:
        cursor.close()
        conn.close()
