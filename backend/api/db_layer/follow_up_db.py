"""
Follow-Up Actions Database Layer
SQL queries for action CRUD, filtering, and derived field computation.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import pyodbc

def get_db_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


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
    Create a new follow-up action.
    
    Returns:
        Created action dict with ID
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Generate initial notes entry
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        user_id = created_by_user_id or 0
        initial_notes = f"[{timestamp}] (user_id={user_id}): Action created"
        
        if notes:
            initial_notes += f"\n[{timestamp}] (user_id={user_id}): {notes}"
        
        query = """
            INSERT INTO dbo.APP_ActionItem (
                ActionTitle,
                ActionDescription,
                IncidentRequestCaseID,
                SeasonalReportID,
                DepartmentID,
                AssignedTo,
                Priority,
                Status,
                DueDate,
                Notes,
                CreatedAt,
                CreatedByUserID,
                LastUpdatedAt,
                LastUpdatedByUserID
            )
            OUTPUT INSERTED.ActionItemID
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(query, (
            action_title,
            action_description,
            incident_case_id,
            seasonal_report_id,
            department_id,
            assigned_to,
            priority,
            status,
            due_date,
            initial_notes,
            datetime.now(),
            created_by_user_id,
            datetime.now(),
            created_by_user_id
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
    Fetch filtered follow-up actions with derived fields and global statistics.
    
    Returns:
        Dict with actions array and global statistics
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Build WHERE clause dynamically
        where_clauses = []
        params = []
        
        # Status filter
        if status and status != 'all':
            where_clauses.append("a.Status = ?")
            params.append(status)
        elif not include_completed:
            # Default: exclude completed unless explicitly included
            where_clauses.append("a.Status != 'completed'")
        
        # Priority filter
        if priority and priority != 'all':
            where_clauses.append("a.Priority = ?")
            params.append(priority)
        
        # Department filter
        if department and department != 'all':
            where_clauses.append("a.DepartmentID = ?")
            params.append(department)
        
        # Date range filter
        if from_date:
            where_clauses.append("CAST(a.DueDate AS DATE) >= ?")
            params.append(from_date)
        
        if to_date:
            where_clauses.append("CAST(a.DueDate AS DATE) <= ?")
            params.append(to_date)
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Main query with derived fields
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
                a.DepartmentID AS departmentId,
                a.AssignedTo AS assignedTo,
                a.Priority AS priority,
                a.Status AS status,
                CAST(a.DueDate AS DATE) AS dueDate,
                CAST(a.CompletedDate AS DATE) AS completedDate,
                a.Notes AS notes,
                a.CreatedAt AS createdAt,
                a.CreatedByUserID AS createdByUserId,
                a.LastUpdatedAt AS lastUpdatedAt,
                a.LastUpdatedByUserID AS lastUpdatedByUserId,
                -- Derived fields
                CASE 
                    WHEN CAST(a.DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                         AND a.Status != 'completed' 
                    THEN 1 
                    ELSE 0 
                END AS isOverdue,
                DATEDIFF(DAY, CAST(GETDATE() AS DATE), CAST(a.DueDate AS DATE)) AS daysRemaining,
                CASE 
                    WHEN CAST(a.DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                         AND a.Status != 'completed'
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
            actions.append({
                'id': row[0],
                'actionTitle': row[1],
                'actionDescription': row[2],
                'sourceType': row[3],
                'sourceId': row[4],
                'departmentId': row[5],
                'assignedTo': row[6],
                'priority': row[7],
                'status': row[8],
                'dueDate': row[9].isoformat() if row[9] else None,
                'completedDate': row[10].isoformat() if row[10] else None,
                'notes': row[11],
                'createdAt': row[12].isoformat() if row[12] else None,
                'createdByUserId': row[13],
                'lastUpdatedAt': row[14].isoformat() if row[14] else None,
                'lastUpdatedByUserId': row[15],
                'isOverdue': bool(row[16]),
                'daysRemaining': row[17],
                'daysOverdue': row[18]
            })
        
        # Get global statistics (not affected by filters)
        stats_query = """
            SELECT
                COUNT(CASE WHEN Status = 'pending' THEN 1 END) AS actionsToTake,
                COUNT(CASE WHEN Status != 'completed' 
                           AND CAST(DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                      THEN 1 END) AS overdue,
                COUNT(CASE WHEN Status = 'completed' THEN 1 END) AS completed
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
    Fetch single action by ID with derived fields.
    
    Returns:
        Action dict or None if not found
    """
    conn = get_db_connection()
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
                a.DepartmentID AS departmentId,
                a.AssignedTo AS assignedTo,
                a.Priority AS priority,
                a.Status AS status,
                CAST(a.DueDate AS DATE) AS dueDate,
                CAST(a.CompletedDate AS DATE) AS completedDate,
                a.Notes AS notes,
                a.CreatedAt AS createdAt,
                a.CreatedByUserID AS createdByUserId,
                a.LastUpdatedAt AS lastUpdatedAt,
                a.LastUpdatedByUserID AS lastUpdatedByUserId,
                -- Derived fields
                CASE 
                    WHEN CAST(a.DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                         AND a.Status != 'completed' 
                    THEN 1 
                    ELSE 0 
                END AS isOverdue,
                DATEDIFF(DAY, CAST(GETDATE() AS DATE), CAST(a.DueDate AS DATE)) AS daysRemaining,
                CASE 
                    WHEN CAST(a.DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                         AND a.Status != 'completed'
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
        
        return {
            'id': row[0],
            'actionTitle': row[1],
            'actionDescription': row[2],
            'sourceType': row[3],
            'sourceId': row[4],
            'departmentId': row[5],
            'assignedTo': row[6],
            'priority': row[7],
            'status': row[8],
            'dueDate': row[9].isoformat() if row[9] else None,
            'completedDate': row[10].isoformat() if row[10] else None,
            'notes': row[11],
            'createdAt': row[12].isoformat() if row[12] else None,
            'createdByUserId': row[13],
            'lastUpdatedAt': row[14].isoformat() if row[14] else None,
            'lastUpdatedByUserId': row[15],
            'isOverdue': bool(row[16]),
            'daysRemaining': row[17],
            'daysOverdue': row[18]
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
    Update action fields with audit trail.
    
    Returns:
        Updated action dict or None if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Build UPDATE clause dynamically
        update_fields = []
        params = []
        
        if due_date:
            update_fields.append("DueDate = ?")
            params.append(due_date)
        
        if assigned_to is not None:
            update_fields.append("AssignedTo = ?")
            params.append(assigned_to)
        
        if priority:
            update_fields.append("Priority = ?")
            params.append(priority)
        
        if status:
            update_fields.append("Status = ?")
            params.append(status)
        
        if notes:
            update_fields.append("Notes = ?")
            params.append(notes)
        
        # Always update LastUpdatedAt and LastUpdatedByUserID
        update_fields.append("LastUpdatedAt = ?")
        params.append(datetime.now())
        
        if last_updated_by_user_id:
            update_fields.append("LastUpdatedByUserID = ?")
            params.append(last_updated_by_user_id)
        
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
    Mark action as completed.
    
    Returns:
        Updated action dict or None if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get current notes to append
        current_action = get_follow_up_action_by_id(action_id)
        if not current_action:
            return None
        
        current_notes = current_action.get('notes', '')
        
        # Append completion notes
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        user_id = last_updated_by_user_id or 0
        new_notes = current_notes or ''
        
        if completion_notes:
            append_text = f"\n[{timestamp}] (user_id={user_id}): {completion_notes}"
            new_notes = (new_notes + append_text).strip()
        else:
            append_text = f"\n[{timestamp}] (user_id={user_id}): Action marked complete"
            new_notes = (new_notes + append_text).strip()
        
        # Set completed date
        comp_date = completed_date or datetime.now().strftime('%Y-%m-%d')
        
        query = """
            UPDATE dbo.APP_ActionItem
            SET
                Status = 'completed',
                CompletedDate = ?,
                Notes = ?,
                LastUpdatedAt = ?,
                LastUpdatedByUserID = ?
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, (
            comp_date,
            new_notes,
            datetime.now(),
            last_updated_by_user_id,
            action_id
        ))
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
    Reopen a completed action back to pending status.
    
    Returns:
        Updated action dict or None if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get current action
        current_action = get_follow_up_action_by_id(action_id)
        if not current_action:
            return None
        
        # Append reopen notes
        current_notes = current_action.get('notes', '')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        user_id = last_updated_by_user_id or 0
        
        append_text = f"\n[{timestamp}] (user_id={user_id}): Reopened - {reopen_reason}"
        new_notes = (current_notes + append_text).strip() if current_notes else append_text.strip()
        
        # Determine new due date
        due_date = new_due_date or datetime.now().strftime('%Y-%m-%d')
        
        query = """
            UPDATE dbo.APP_ActionItem
            SET
                Status = 'pending',
                DueDate = ?,
                CompletedDate = NULL,
                Notes = ?,
                LastUpdatedAt = ?,
                LastUpdatedByUserID = ?
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, (
            due_date,
            new_notes,
            datetime.now(),
            last_updated_by_user_id,
            action_id
        ))
        conn.commit()
        
        # Fetch updated action
        return get_follow_up_action_by_id(action_id)
    
    finally:
        cursor.close()
        conn.close()


def get_action_history(action_id: int) -> List[Dict[str, Any]]:
    """
    Get change history for an action from notes field.
    
    Parses notes to extract timestamped changes.
    
    Returns:
        List of history entries
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        action = get_follow_up_action_by_id(action_id)
        if not action:
            return []
        
        notes = action.get('notes', '')
        if not notes:
            return []
        
        # Parse notes for history entries
        # Format: [YYYY-MM-DD HH:MM] (user_id=X): message
        import re
        pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] \(user_id=(\d+)\): (.+)'
        
        history = []
        for match in re.finditer(pattern, notes):
            timestamp_str, user_id_str, message = match.groups()
            history.append({
                'timestamp': timestamp_str,
                'userId': int(user_id_str),
                'action': message.split(' - ')[0] if ' - ' in message else message.split(':')[0] if ':' in message else 'Updated',
                'details': message
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
    Get actions grouped by date for calendar view.
    
    Returns:
        Dict with actions grouped by date
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Build WHERE clause
        where_clauses = []
        params = []
        
        # Filter by year and month
        where_clauses.append("YEAR(a.DueDate) = ? AND MONTH(a.DueDate) = ?")
        params.extend([year, month])
        
        # Status filter (default: exclude completed)
        if status and status != 'all':
            where_clauses.append("a.Status = ?")
            params.append(status)
        else:
            where_clauses.append("a.Status != 'completed'")
        
        # Department filter
        if department and department != 'all':
            where_clauses.append("a.DepartmentID = ?")
            params.append(department)
        
        where_sql = " AND ".join(where_clauses)
        
        query = f"""
            SELECT
                CAST(a.DueDate AS DATE) AS dueDate,
                a.ActionItemID AS id,
                a.ActionTitle AS actionTitle,
                a.Priority AS priority,
                a.Status AS status,
                a.DepartmentID AS departmentId,
                a.AssignedTo AS assignedTo,
                CASE 
                    WHEN CAST(a.DueDate AS DATE) < CAST(GETDATE() AS DATE) 
                         AND a.Status != 'completed' 
                    THEN 1 
                    ELSE 0 
                END AS isOverdue
            FROM dbo.APP_ActionItem a
            WHERE {where_sql}
            ORDER BY a.DueDate ASC, a.Priority DESC
        """
        
        cursor.execute(query, params)
        
        # Group actions by date
        calendar_data = {}
        
        for row in cursor.fetchall():
            date_str = row[0].isoformat() if row[0] else None
            
            if date_str not in calendar_data:
                calendar_data[date_str] = []
            
            calendar_data[date_str].append({
                'id': row[1],
                'actionTitle': row[2],
                'priority': row[3],
                'status': row[4],
                'departmentId': row[5],
                'assignedTo': row[6],
                'isOverdue': bool(row[7])
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
    conn = get_db_connection()
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
                
                # Append completion notes
                current_notes = current_action.get('notes', '')
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                user_id = last_updated_by_user_id or 0
                
                if completion_notes:
                    append_text = f"\n[{timestamp}] (user_id={user_id}): {completion_notes}"
                else:
                    append_text = f"\n[{timestamp}] (user_id={user_id}): Bulk completed"
                
                new_notes = (current_notes + append_text).strip() if current_notes else append_text.strip()
                
                # Set completed date
                comp_date = completed_date or datetime.now().strftime('%Y-%m-%d')
                
                query = """
                    UPDATE dbo.APP_ActionItem
                    SET
                        Status = 'completed',
                        CompletedDate = ?,
                        Notes = ?,
                        LastUpdatedAt = ?,
                        LastUpdatedByUserID = ?
                    WHERE ActionItemID = ?
                """
                
                cursor.execute(query, (
                    comp_date,
                    new_notes,
                    datetime.now(),
                    last_updated_by_user_id,
                    action_id
                ))
                
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
    conn = get_db_connection()
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
                
                # Append delay notes
                current_notes = current_action.get('notes', '')
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                user_id = last_updated_by_user_id or 0
                
                delay_text = f"Bulk delayed {delay_days} days"
                if reason:
                    delay_text += f" - {reason}"
                
                append_text = f"\n[{timestamp}] (user_id={user_id}): {delay_text}"
                new_notes = (current_notes + append_text).strip() if current_notes else append_text.strip()
                
                query = """
                    UPDATE dbo.APP_ActionItem
                    SET
                        DueDate = ?,
                        Status = 'pending',
                        Notes = ?,
                        LastUpdatedAt = ?,
                        LastUpdatedByUserID = ?
                    WHERE ActionItemID = ?
                """
                
                cursor.execute(query, (
                    new_due_date,
                    new_notes,
                    datetime.now(),
                    last_updated_by_user_id,
                    action_id
                ))
                
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
    
    Returns:
        Dict with success count and failed IDs
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        updated_count = 0
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
                    failed_ids.append({'id': action_id, 'reason': 'Cannot update completed action'})
                    continue
                
                # Build update clause
                update_fields = []
                params = []
                
                if assigned_to is not None:
                    update_fields.append("AssignedTo = ?")
                    params.append(assigned_to)
                
                if priority:
                    update_fields.append("Priority = ?")
                    params.append(priority)
                
                if department_id is not None:
                    update_fields.append("DepartmentID = ?")
                    params.append(department_id)
                
                if not update_fields:
                    failed_ids.append({'id': action_id, 'reason': 'No fields to update'})
                    continue
                
                # Always update audit fields
                update_fields.append("LastUpdatedAt = ?")
                params.append(datetime.now())
                
                if last_updated_by_user_id:
                    update_fields.append("LastUpdatedByUserID = ?")
                    params.append(last_updated_by_user_id)
                
                params.append(action_id)
                
                update_sql = ", ".join(update_fields)
                
                query = f"""
                    UPDATE dbo.APP_ActionItem
                    SET {update_sql}
                    WHERE ActionItemID = ?
                """
                
                cursor.execute(query, params)
                updated_count += 1
            
            except Exception as e:
                failed_ids.append({'id': action_id, 'reason': str(e)})
        
        conn.commit()
        
        return {
            'successCount': updated_count,
            'failedCount': len(failed_ids),
            'failedIds': failed_ids
        }
    
    finally:
        cursor.close()
        conn.close()


def delay_follow_up_action(
    action_id: int,
    delay_days: int,
    reason: Optional[str] = None,
    last_updated_by_user_id: int = None
) -> Optional[Dict[str, Any]]:
    """
    Delay action by specified days.
    
    Returns:
        Updated action dict or None if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get current action to calculate new due date
        current_action = get_follow_up_action_by_id(action_id)
        if not current_action:
            return None
        
        # Parse current due date
        from datetime import datetime, timedelta
        due_date = datetime.fromisoformat(current_action['dueDate'])
        new_due_date = (due_date + timedelta(days=delay_days)).strftime('%Y-%m-%d')
        
        # Append delay notes
        current_notes = current_action.get('notes', '')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        user_id = last_updated_by_user_id or 0
        
        delay_text = f"Delayed {delay_days} days"
        if reason:
            delay_text += f" - {reason}"
        
        append_text = f"\n[{timestamp}] (user_id={user_id}): {delay_text}"
        new_notes = (current_notes + append_text).strip() if current_notes else append_text.strip()
        
        query = """
            UPDATE dbo.APP_ActionItem
            SET
                DueDate = ?,
                Status = 'pending',
                Notes = ?,
                LastUpdatedAt = ?,
                LastUpdatedByUserID = ?
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, (
            new_due_date,
            new_notes,
            datetime.now(),
            last_updated_by_user_id,
            action_id
        ))
        conn.commit()
        
        # Fetch updated action
        return get_follow_up_action_by_id(action_id)
    
    finally:
        cursor.close()
        conn.close()
