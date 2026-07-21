

### ActionItems

| Function Name                    | Method Signature                                                                                                                                                                                          | Operation Type   | Entity / Table   | Purpose                                                                             | Key Rules / Constraints                                                                       | Return Type          |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | -------------------- |
| `create_action_item`             | `(*, action_title: str, created_by_user_id: int, incident_case_id: int \| None = None, season_case_id: int \| None = None, action_description: str \| None = None, due_date: date \| None = None) -> int` | CREATE           | `APP_ActionItem` | Create a new action item linked to **either** an Incident case **or** a Season case | Exactly one of `incident_case_id` or `season_case_id` must be provided                        | `ActionItemID (int)` |
| `get_action_item_by_id`          | `(action_item_id: int) -> dict \| None`                                                                                                                                                                   | READ             | `APP_ActionItem` | Retrieve a single action item by primary key                                        | Return `None` if not found                                                                    | `dict \| None`       |
| `list_action_items_for_incident` | `(incident_case_id: int) -> list[dict]`                                                                                                                                                                   | READ (List)      | `APP_ActionItem` | List all action items related to a specific Incident case                           | Ordered by `CreatedAt DESC`                                                                   | `list[dict]`         |
| `list_action_items_for_season`   | `(season_case_id: int) -> list[dict]`                                                                                                                                                                     | READ (List)      | `APP_ActionItem` | List all action items related to a specific Season case                             | Ordered by `CreatedAt DESC`                                                                   | `list[dict]`         |
| `update_action_item`             | `(action_item_id: int, updates: dict) -> None`                                                                                                                                                            | UPDATE (Partial) | `APP_ActionItem` | Partially update editable fields of an action item                                  | Allowed fields only: `ActionTitle`, `ActionDescription`, `DueDate`, `IsDone`, `DateSubmitted` | `None`               |
| `mark_action_item_done`          | `(action_item_id: int) -> None`                                                                                                                                                                           | UPDATE (State)   | `APP_ActionItem` | Mark an action item as completed and set submission date to today                   | Sets `IsDone = 1`, `DateSubmitted = GETDATE()`                                                | `None`               |

### admin_units

|Function Name|Method Signature|Operation Type|Entity / Table|Purpose|Key Rules / Constraints|Return Type|
|---|---|---|---|---|---|---|
|`get_admin_unit_by_id`|`(admin_unit_id: int)`|READ (Single)|`AdminsrationUnit`|Retrieve one administration unit by its `UniqueID`|Must return exactly one row or `None`|`row \| None`|
|`get_admin_unit_children`|`(parent_id: int)`|READ (List)|`AdminsrationUnit`|Retrieve direct child units of a given administration unit|Uses `ParentID` relationship only (no recursion)|`list[row]`|
|`get_admin_unit_parent`|`(admin_unit_id: int)`|READ (Single / Join)|`AdminsrationUnit`|Retrieve the direct parent unit of a given administration unit|Self-join on `ParentID → UniqueID`|`row \| None`|
|`get_admin_unit_tree`|`()`|READ (List)|`AdminsrationUnit`|Retrieve **all** administration units (flat list)|Tree/hierarchy construction handled outside db_layer|`list[row]`|
|`get_admin_unit_leaves`|`()`|READ (List / Join)|`AdminsrationUnit`|Retrieve administration units that have **no children** (leaf nodes)|Uses LEFT JOIN where no child exists|`list[row]`|
|`get_active_admin_units`|`()`|READ (List)|`AdminsrationUnit`|Retrieve administration units that are not frozen|`Frozen = 0` only|`list[row]`|

### incident_case

|Function Name|Method Signature|Operation Type|Entity / Table|Purpose|Key Rules / Constraints|Return Type|
|---|---|---|---|---|---|---|
|`create_incident_case`|`(data: dict) -> int`|CREATE|`APP_IncidentCase`|Create a new incident case record|Inserts all core incident fields; required keys must exist in `data`|`IncidentRequestCaseID (int)`|
|`get_incident_case_by_id`|`(incident_id: int) -> dict \| None`|READ (Single)|`APP_IncidentCase`|Retrieve a single incident case by primary key|Return `None` if not found|`dict \| None`|
|`list_incident_cases`|`() -> list[dict]`|READ (List)|`APP_IncidentCase`|Retrieve all incident cases ordered by creation date (newest first)|Ordering by `CreatedAt DESC`|`list[dict]`|
|`update_incident_case`|`(incident_id: int, updates: dict) -> None`|UPDATE (Partial)|`APP_IncidentCase`|Partially update an incident case|Only fields in `UPDATABLE_FIELDS` may be updated|`None`|
|`soft_delete_incident_case`|`(incident_id: int, closed_status_id: int) -> None`|UPDATE (Soft Delete)|`APP_IncidentCase`|Soft-delete an incident case by updating its status|Does **not** delete row; sets `CaseStatusID` only|`None`|

### incident_case_doctor

|Function Name|Method Signature|Operation Type|Entity / Table|Purpose|Key Rules / Constraints|Return Type|
|---|---|---|---|---|---|---|
|`add_doctor_to_case`|`(incident_id: int, doctor_id: int, assigned_by_user_id: int, is_primary: bool = False) -> int`|CREATE|`APP_IncidentCaseDoctor`|Assign a doctor to an incident case|If `is_primary=True`, all existing doctors for the case must be unset as primary before insert|`IncidentCaseDoctorID (int)`|
|`list_case_doctors`|`(incident_id: int) -> list[dict]`|READ (List)|`APP_IncidentCaseDoctor`|List all doctors assigned to a specific incident case|Ordered by `IsPrimary DESC`, then `AssignedAt ASC`|`list[dict]`|
|`set_primary_doctor`|`(incident_id: int, incident_case_doctor_id: int) -> None`|UPDATE (State)|`APP_IncidentCaseDoctor`|Set a specific doctor assignment as the primary doctor for a case|Must ensure **only one** primary doctor per case|`None`|
|`remove_doctor_from_case`|`(incident_case_doctor_id: int) -> None`|DELETE|`APP_IncidentCaseDoctor`|Remove a doctor assignment from an incident case|Physical delete of assignment row|`None`|


### incident_case_feedback

|Function Name|Method Signature|Operation Type|Entity / Table|Purpose|Key Rules / Constraints|Return Type|
|---|---|---|---|---|---|---|
|`add_doctor_to_case`|`(incident_id: int, doctor_id: int, assigned_by_user_id: int, is_primary: bool = False) -> int`|CREATE|`APP_IncidentCaseDoctor`|Assign a doctor to an incident case|If `is_primary = True`, all existing doctors for the case must be unset as primary before insertion|`IncidentCaseDoctorID (int)`|
|`list_case_doctors`|`(incident_id: int) -> list[dict]`|READ (List)|`APP_IncidentCaseDoctor`|Retrieve all doctors assigned to a specific incident case|Order by `IsPrimary DESC`, then `AssignedAt ASC`|`list[dict]`|
|`set_primary_doctor`|`(incident_id: int, incident_case_doctor_id: int) -> None`|UPDATE (State)|`APP_IncidentCaseDoctor`|Set one doctor assignment as the primary doctor for a case|Must guarantee **exactly one** primary doctor per case|`None`|
|`remove_doctor_from_case`|`(incident_case_doctor_id: int) -> None`|DELETE|`APP_IncidentCaseDoctor`|Remove a doctor assignment from an incident case|Hard delete of assignment row|`None`|


### incident_case_target_department

|Function Name|Method Signature|Operation Type|Entity / Table|Purpose|Key Rules / Constraints|Return Type|
|---|---|---|---|---|---|---|
|`add_target_department`|`(incident_id: int, department_id: int, assigned_by_user_id: int, is_primary: bool = False) -> int`|CREATE|`APP_IncidentCaseTargetDepartment`|Assign a target department to an incident case|If `is_primary = True`, all existing target departments for the case must be unset before insertion|`TargetID (int)`|
|`list_target_departments`|`(incident_id: int) -> list[dict]`|READ (List)|`APP_IncidentCaseTargetDepartment`|Retrieve all target departments assigned to a specific incident case|Ordered by `IsPrimary DESC`, then `AssignedAt ASC`|`list[dict]`|
|`set_primary_department`|`(incident_id: int, target_id: int) -> None`|UPDATE (State)|`APP_IncidentCaseTargetDepartment`|Set one target department as the primary department for a case|Must guarantee **exactly one** primary department per case|`None`|
|`remove_target_department`|`(target_id: int) -> None`|DELETE|`APP_IncidentCaseTargetDepartment`|Remove a target department assignment from an incident case|Hard delete of assignment row|`None`|



### lookups

|Function Name|Method Signature|Operation Type|Entity / Table(s)|Purpose|Key Rules / Constraints|Return Type|
|---|---|---|---|---|---|---|
|`_fetch_all`|`(query: str, params: tuple = ()) -> list[dict]`|READ (Helper)|Generic|Execute a read-only query and return rows as dictionaries|Internal helper; no business logic; reusable by lookup functions|`list[dict]`|
|`get_case_stages`|`() -> list[dict]`|READ (List)|`APP_LOOKUP_CASE_STAGE`|Retrieve all case stages|Ordered by `StageOrder`|`list[dict]`|
|`get_case_statuses`|`(active_only: bool = True) -> list[dict]`|READ (List)|`APP_LOOKUP_CASE_STATUS`|Retrieve case statuses|If `active_only=True`, filter `IsActive = 1`; ordered by `DisplayOrder`|`list[dict]`|
|`get_domains`|`() -> list[dict]`|READ (List)|`APP_LOOKUP_DOMAIN`|Retrieve all domains|Ordered by `DomainOrder`|`list[dict]`|
|`get_categories`|`(domain_id: int \| None = None) -> list[dict]`|READ (List)|`APP_LOOKUP_CATEGORY`|Retrieve categories, optionally filtered by domain|If `domain_id` provided, filter by `DomainID`; ordered by `CategoryOrder`|`list[dict]`|
|`get_subcategories`|`(category_id: int \| None = None) -> list[dict]`|READ (List)|`APP_LOOKUP_SUBCATEGORY`|Retrieve subcategories, optionally filtered by category|If `category_id` provided, filter by `CategoryID`; ordered by name|`list[dict]`|
|`get_classifications`|`(subcategory_id: int \| None = None) -> list[dict]`|READ (List)|`APP_LOOKUP_CLASSIFICATION`|Retrieve classifications, optionally filtered by subcategory|If `subcategory_id` provided, filter by `SubCategoryID`; ordered by Arabic name|`list[dict]`|
|`get_clinical_risk_types`|`(active_only: bool = True) -> list[dict]`|READ (List)|`APP_LOOKUP_CLINICAL_RISK_TYPE`|Retrieve clinical risk types|If `active_only=True`, filter `IsActive = 1`; ordered by `DisplayOrder`|`list[dict]`|
|`get_feedback_intent_types`|`(active_only: bool = True) -> list[dict]`|READ (List)|`APP_LOOKUP_FEEDBACK_INTENT_TYPE`|Retrieve feedback intent types|If `active_only=True`, filter `IsActive = 1`; ordered by `DisplayOrder`|`list[dict]`|
|`get_harm_levels`|`() -> list[dict]`|READ (List)|`APP_LOOKUP_HARM_LEVEL`|Retrieve harm levels|Ordered by `SeverityOrder`|`list[dict]`|
|`get_explanation_statuses`|`() -> list[dict]`|READ (List)|`APP_LOOKUP_EXPLANATION_STATUS`|Retrieve explanation statuses|Ordered by status name|`list[dict]`|
|`get_doctors`|`(active_only: bool = True) -> list[dict]`|READ (List)|`APP_LOOKUP_DOCTOR`|Retrieve doctors|If `active_only=True`, filter `IsActive = 1`; ordered by doctor name|`list[dict]`|


### org_unit_policy

| Function Name                             | Method Signature                                                                                                                                                                                                                                                                                                                                                 | Operation Type   | Entity / Table(s)                       | Purpose                                                                                                | Key Rules / Constraints                                                              | Return Type    |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ | -------------- |
| `get_policy_by_unit_id`                   | `(org_unit_id: int) -> dict \| None`                                                                                                                                                                                                                                                                                                                             | READ (Single)    | `APP_OrgUnitPolicy`                     | Retrieve the policy configuration for any organizational unit (administration, department, or section) | Return `None` if no policy exists for the unit                                       | `dict \| None` |
| `get_administration_policy`               | `(administration_id: int) -> dict \| None`                                                                                                                                                                                                                                                                                                                       | READ (Alias)     | `APP_OrgUnitPolicy`                     | Retrieve policy for an administration unit                                                             | Alias of `get_policy_by_unit_id`                                                     | `dict \| None` |
| `get_department_policy`                   | `(department_id: int) -> dict \| None`                                                                                                                                                                                                                                                                                                                           | READ (Alias)     | `APP_OrgUnitPolicy`                     | Retrieve policy for a department unit                                                                  | Alias of `get_policy_by_unit_id`                                                     | `dict \| None` |
| `get_section_policy`                      | `(section_id: int) -> dict \| None`                                                                                                                                                                                                                                                                                                                              | READ (Alias)     | `APP_OrgUnitPolicy`                     | Retrieve policy for a section unit                                                                     | Alias of `get_policy_by_unit_id`                                                     | `dict \| None` |
| `update_policy_for_unit`                  | `(org_unit_id: int, *, low_severity_limit: int, medium_severity_limit: int, high_severity_limit: int, clinical_domain_limit: int, management_domain_limit: int, relational_domain_limit: int, enable_low_rule: bool, enable_medium_rule: bool, enable_high_percentage_rule: bool, enable_high_percentage_by_domain_rule: bool, updated_by_user_id: int) -> None` | UPDATE (Single)  | `APP_OrgUnitPolicy`                     | Update policy values for a single organizational unit                                                  | **No cascading**; updates only the specified unit                                    | `None`         |
| `update_policy_for_unit_with_descendants` | `(org_unit_id: int, *, policy_data: dict, updated_by_user_id: int) -> None`                                                                                                                                                                                                                                                                                      | UPDATE (Cascade) | `APP_OrgUnitPolicy`, `AdminsrationUnit` | Update policy for an organizational unit **and all its descendants**                                   | Must use **iterative traversal (no recursion)**; must prevent cycles via visited set | `None`         |

### season_cases

|Function Name|Method Signature|Operation Type|Entity / Table|Purpose|Key Rules / Constraints|Return Type|
|---|---|---|---|---|---|---|
|`create_season_case`|`(*, season_id: int, department_id: int, season_case_status_id: int, created_by_user_id: int, seasonal_report_text: str \| None = None, seasonal_report_department_feedback: str \| None = None) -> int`|CREATE|`APP_SeasonCase`|Create a seasonal case for a department within a season|One season case per department per season is assumed (not enforced here)|`SeasonCaseID (int)`|
|`get_season_case_by_id`|`(season_case_id: int) -> dict \| None`|READ (Single)|`APP_SeasonCase`|Retrieve a single season case by primary key|Return `None` if not found|`dict \| None`|
|`list_season_cases`|`(season_id: int \| None = None, department_id: int \| None = None) -> list[dict]`|READ (List)|`APP_SeasonCase`|List season cases, optionally filtered by season or department|Filters are optional; ordered by `CreatedAt DESC`|`list[dict]`|
|`update_season_case`|`(season_case_id: int, updates: dict) -> None`|UPDATE (Partial)|`APP_SeasonCase`|Partially update editable season case fields|Allowed fields only: `SeasonalReportText`, `SeasonalReportDepartmentFeedback`, `SeasonCaseStatusID`|`None`|