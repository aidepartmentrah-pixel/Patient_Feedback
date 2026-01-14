"""
Follow-Up Actions Service Layer
Business logic for follow-up action management with status validation.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from ..db_layer import follow_up_db


class FollowUpService:
    """Service for follow-up action operations."""
    
    # Valid status transitions
    VALID_TRANSITIONS = {
        'pending': ['delayed', 'completed'],
        'delayed': ['pending', 'completed'],
        'completed': []  # Final state
    }
    
    @staticmethod
    def get_follow_up_actions(
        status: Optional[str] = None,
        priority: Optional[str] = None,
        department: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        include_completed: bool = False
    ) -> Dict[str, Any]:
        """
        Get filtered follow-up actions with statistics.
        
        Returns:
            Dict with actions array, total count, and global statistics
        """
        try:
            # Validate date format if provided
            if from_date:
                try:
                    datetime.strptime(from_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid from_date format. Use YYYY-MM-DD")
            
            if to_date:
                try:
                    datetime.strptime(to_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid to_date format. Use YYYY-MM-DD")
            
            result = follow_up_db.get_follow_up_actions(
                status=status,
                priority=priority,
                department=department,
                from_date=from_date,
                to_date=to_date,
                include_completed=include_completed
            )
            
            return result
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to fetch follow-up actions: {str(e)}")
    
    @staticmethod
    def get_follow_up_action_by_id(action_id: int) -> Dict[str, Any]:
        """
        Get single action by ID.
        
        Raises:
            ValueError: If action not found
        """
        try:
            action = follow_up_db.get_follow_up_action_by_id(action_id)
            
            if not action:
                raise ValueError(f"Action with ID {action_id} not found")
            
            return action
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to fetch action: {str(e)}")
    
    @staticmethod
    def update_follow_up_action(
        action_id: int,
        due_date: Optional[str] = None,
        assigned_to: Optional[str] = None,
        priority: Optional[str] = None,
        status: Optional[str] = None,
        notes: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update action with status transition validation.
        
        Raises:
            ValueError: If action not found or invalid transition
        """
        try:
            # Get current action
            current_action = follow_up_db.get_follow_up_action_by_id(action_id)
            if not current_action:
                raise ValueError(f"Action with ID {action_id} not found")
            
            # Validate status transition if status is being changed
            if status and status != current_action['status']:
                current_status = current_action['status']
                
                if current_status not in FollowUpService.VALID_TRANSITIONS:
                    raise ValueError(f"Unknown current status: {current_status}")
                
                if status not in FollowUpService.VALID_TRANSITIONS[current_status]:
                    raise ValueError(
                        f"Invalid status transition from '{current_status}' to '{status}'. "
                        f"Allowed transitions: {FollowUpService.VALID_TRANSITIONS[current_status]}"
                    )
            
            # Validate date format if provided
            if due_date:
                try:
                    datetime.strptime(due_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid due_date format. Use YYYY-MM-DD")
            
            # Append notes if provided
            final_notes = notes
            if notes and notes != current_action.get('notes', ''):
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                user_id_str = user_id or 0
                current_notes = current_action.get('notes', '')
                append_text = f"[{timestamp}] (user_id={user_id_str}): {notes}"
                final_notes = (current_notes + '\n' + append_text).strip() if current_notes else append_text
            
            # Update action
            updated_action = follow_up_db.update_follow_up_action(
                action_id=action_id,
                due_date=due_date,
                assigned_to=assigned_to,
                priority=priority,
                status=status,
                notes=final_notes,
                last_updated_by_user_id=user_id
            )
            
            if not updated_action:
                raise ValueError(f"Failed to update action {action_id}")
            
            return updated_action
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to update action: {str(e)}")
    
    @staticmethod
    def complete_follow_up_action(
        action_id: int,
        completion_notes: Optional[str] = None,
        completed_date: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Mark action as completed.
        
        Raises:
            ValueError: If action not found or already completed
        """
        try:
            # Get current action
            current_action = follow_up_db.get_follow_up_action_by_id(action_id)
            if not current_action:
                raise ValueError(f"Action with ID {action_id} not found")
            
            # Check if already completed
            if current_action['status'] == 'completed':
                raise ValueError(f"Action {action_id} is already completed")
            
            # Validate date format if provided
            if completed_date:
                try:
                    datetime.strptime(completed_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid completed_date format. Use YYYY-MM-DD")
            
            # Complete action
            completed_action = follow_up_db.complete_follow_up_action(
                action_id=action_id,
                completion_notes=completion_notes,
                completed_date=completed_date,
                last_updated_by_user_id=user_id
            )
            
            if not completed_action:
                raise ValueError(f"Failed to complete action {action_id}")
            
            return completed_action
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to complete action: {str(e)}")
    
    @staticmethod
    def delay_follow_up_action(
        action_id: int,
        delay_days: int,
        reason: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Delay action by specified number of days.
        
        Raises:
            ValueError: If action not found, already completed, or invalid delay
        """
        try:
            # Get current action
            current_action = follow_up_db.get_follow_up_action_by_id(action_id)
            if not current_action:
                raise ValueError(f"Action with ID {action_id} not found")
            
            # Check if already completed
            if current_action['status'] == 'completed':
                raise ValueError(f"Cannot delay completed action {action_id}")
            
            # Validate delay days
            if delay_days <= 0:
                raise ValueError("Delay days must be a positive number")
            
            # Delay action
            delayed_action = follow_up_db.delay_follow_up_action(
                action_id=action_id,
                delay_days=delay_days,
                reason=reason,
                last_updated_by_user_id=user_id
            )
            
            if not delayed_action:
                raise ValueError(f"Failed to delay action {action_id}")
            
            return delayed_action
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to delay action: {str(e)}")
    
    @staticmethod
    def reopen_follow_up_action(
        action_id: int,
        reopen_reason: str,
        new_due_date: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Reopen a completed action back to pending status.
        
        Raises:
            ValueError: If action not found or not completed
        """
        try:
            # Get current action
            current_action = follow_up_db.get_follow_up_action_by_id(action_id)
            if not current_action:
                raise ValueError(f"Action with ID {action_id} not found")
            
            # Check if action is completed
            if current_action['status'] != 'completed':
                raise ValueError(f"Only completed actions can be reopened. Current status: {current_action['status']}")
            
            # Validate new due date if provided
            if new_due_date:
                try:
                    datetime.strptime(new_due_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid new_due_date format. Use YYYY-MM-DD")
            
            # Validate reason is provided
            if not reopen_reason or not reopen_reason.strip():
                raise ValueError("Reopen reason is required")
            
            # Reopen action
            reopened_action = follow_up_db.reopen_follow_up_action(
                action_id=action_id,
                reopen_reason=reopen_reason,
                new_due_date=new_due_date,
                last_updated_by_user_id=user_id
            )
            
            if not reopened_action:
                raise ValueError(f"Failed to reopen action {action_id}")
            
            return reopened_action
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to reopen action: {str(e)}")
    
    @staticmethod
    def get_action_history(action_id: int) -> Dict[str, Any]:
        """
        Get change history for an action.
        
        Raises:
            ValueError: If action not found
        """
        try:
            # Verify action exists
            action = follow_up_db.get_follow_up_action_by_id(action_id)
            if not action:
                raise ValueError(f"Action with ID {action_id} not found")
            
            history = follow_up_db.get_action_history(action_id)
            
            return {
                'actionId': action_id,
                'history': history
            }
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to fetch action history: {str(e)}")
    
    @staticmethod
    def get_calendar_actions(
        year: int,
        month: int,
        department: Optional[str] = None,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get actions grouped by date for calendar view.
        
        Raises:
            ValueError: If invalid year/month
        """
        try:
            # Validate year and month
            if year < 2000 or year > 2100:
                raise ValueError(f"Invalid year: {year}. Must be between 2000 and 2100")
            
            if month < 1 or month > 12:
                raise ValueError(f"Invalid month: {month}. Must be between 1 and 12")
            
            calendar_data = follow_up_db.get_calendar_actions(
                year=year,
                month=month,
                department=department,
                status=status
            )
            
            return calendar_data
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to fetch calendar actions: {str(e)}")
    
    @staticmethod
    def create_follow_up_action(
        action_title: str,
        action_description: Optional[str] = None,
        incident_case_id: Optional[int] = None,
        seasonal_report_id: Optional[int] = None,
        department_id: Optional[int] = None,
        assigned_to: Optional[str] = None,
        priority: str = 'medium',
        due_date: str = None,
        notes: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a new follow-up action.
        
        Raises:
            ValueError: If validation fails
        """
        try:
            # Validate required fields
            if not action_title or not action_title.strip():
                raise ValueError("Action title is required")
            
            if not due_date:
                raise ValueError("Due date is required")
            
            # Validate due date format
            try:
                datetime.strptime(due_date, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Invalid due_date format. Use YYYY-MM-DD")
            
            # Validate priority
            if priority not in ['high', 'medium', 'low']:
                raise ValueError(f"Invalid priority: {priority}. Must be high, medium, or low")
            
            # Validate source linking (exactly one or neither)
            if incident_case_id and seasonal_report_id:
                raise ValueError("Cannot link to both incident and seasonal report. Choose one.")
            
            # Create action
            created_action = follow_up_db.create_follow_up_action(
                action_title=action_title,
                action_description=action_description,
                incident_case_id=incident_case_id,
                seasonal_report_id=seasonal_report_id,
                department_id=department_id,
                assigned_to=assigned_to,
                priority=priority,
                status='pending',
                due_date=due_date,
                notes=notes,
                created_by_user_id=user_id
            )
            
            if not created_action:
                raise ValueError("Failed to create action")
            
            return created_action
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to create action: {str(e)}")
    
    @staticmethod
    def bulk_complete_actions(
        action_ids: List[int],
        completion_notes: Optional[str] = None,
        completed_date: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Mark multiple actions as completed.
        
        Raises:
            ValueError: If validation fails
        """
        try:
            if not action_ids or len(action_ids) == 0:
                raise ValueError("Action IDs list cannot be empty")
            
            # Validate completed date if provided
            if completed_date:
                try:
                    datetime.strptime(completed_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid completed_date format. Use YYYY-MM-DD")
            
            result = follow_up_db.bulk_complete_actions(
                action_ids=action_ids,
                completion_notes=completion_notes,
                completed_date=completed_date,
                last_updated_by_user_id=user_id
            )
            
            return result
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to bulk complete actions: {str(e)}")
    
    @staticmethod
    def bulk_delay_actions(
        action_ids: List[int],
        delay_days: int,
        reason: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Delay multiple actions by specified days.
        
        Raises:
            ValueError: If validation fails
        """
        try:
            if not action_ids or len(action_ids) == 0:
                raise ValueError("Action IDs list cannot be empty")
            
            if delay_days <= 0:
                raise ValueError("Delay days must be a positive number")
            
            result = follow_up_db.bulk_delay_actions(
                action_ids=action_ids,
                delay_days=delay_days,
                reason=reason,
                last_updated_by_user_id=user_id
            )
            
            return result
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to bulk delay actions: {str(e)}")
    
    @staticmethod
    def bulk_update_actions(
        action_ids: List[int],
        assigned_to: Optional[str] = None,
        priority: Optional[str] = None,
        department_id: Optional[int] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update multiple actions with same values.
        
        Raises:
            ValueError: If validation fails
        """
        try:
            if not action_ids or len(action_ids) == 0:
                raise ValueError("Action IDs list cannot be empty")
            
            # Validate priority if provided
            if priority and priority not in ['high', 'medium', 'low']:
                raise ValueError(f"Invalid priority: {priority}. Must be high, medium, or low")
            
            # Check at least one field to update
            if assigned_to is None and priority is None and department_id is None:
                raise ValueError("At least one field must be provided for update")
            
            result = follow_up_db.bulk_update_actions(
                action_ids=action_ids,
                assigned_to=assigned_to,
                priority=priority,
                department_id=department_id,
                last_updated_by_user_id=user_id
            )
            
            return result
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to bulk update actions: {str(e)}")
