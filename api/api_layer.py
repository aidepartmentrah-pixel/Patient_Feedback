from api.universal_object import UniversalIncidentRecord, IncidentCase, CaseAction
from db_layer import IncidentRequestDB, IncidentRequestCaseDB, IncidentRequestCaseActionDB  # adjust import
from datetime import datetime




def fetch_incident_records(
    search_name=None,
    issuing=None,
    target=None,
    feedback_category=None,
    severity=None,
    stage=None,
    harm=None,
    domain=None,
    category=None,
    subcategory=None,
    status=None,
    start_date=None,
    end_date=None,
    include_cases=True,
    include_actions=True
):
    """
    Fetch incidents with optional filters, including related cases and actions if requested.
    Returns a list of incidents, each with optional 'cases' and 'actions' keys.
    """
    # ------------------ 1. Fetch incidents ------------------
    incidents = IncidentRequestDB.get_records(
        search_name=search_name,
        issuing=issuing,
        target=target,
        feedback_category=feedback_category,
        severity=severity,
        stage=stage,
        harm=harm,
        domain=domain,
        category=category,
        subcategory=subcategory,
        status=status,
        start_date=start_date,
        end_date=end_date
    )

    # ------------------ 2. Attach cases ------------------
    if include_cases:
        for incident in incidents:
            incident_id = incident.get("UniqueID")
            cases = IncidentRequestCaseDB.get_records(incident_request_id=incident_id)
            incident["cases"] = cases

            # ------------------ 3. Attach actions ------------------
            if include_actions:
                for case in incident["cases"]:
                    case_id = case.get("UniqueID")
                    actions = IncidentRequestCaseActionDB.get_records(case_id=case_id)
                    case["actions"] = actions

    return incidents

def add_incident(
    feedback_received_date=None,
    record_id=None,
    patient_full_name=None,
    issuing_department=None,   # must be INT (SourceSectionID)
    target_department=None,    # must be INT (SourceDepartmentID)
    source_1=None,             # must be INT (IncidentSourceID)
    feedback_type=None,        # must be INT (IncidentRequesterTypeID)
    domain=None,
    category=None,
    sub_category=None,
    classification_ar=None,
    classification_en=None,
    complaint_text=None,
    immediate_action=None,
    taken_action=None,
    severity_level=None,       # must be INT
    stage=None,
    harm_level=None,
    status=None,               # must be INT
    improvement_opportunity_type=None,
    **extra_fields
):
    """
    Add a new incident record safely with proper type handling for INT and string fields.

    Returns:
        int: UniqueID of the newly created incident.
    """
    incident_data = type("IncidentData", (), {})()  # empty object

    # Required numeric fields with safe defaults
    incident_data.YearCounter = datetime.now().year
    incident_data.PatientTypeID = 0
    incident_data.DoctorID = 0
    incident_data.EmployeeID = 0
    incident_data.MRN = None

    # Map API fields to DB-layer fields
    incident_data.DateAndTimeRecieved = feedback_received_date or datetime.now()
    incident_data.Code = record_id or f"REC-{datetime.now().timestamp()}"
    incident_data.PatientName = patient_full_name or "Unknown"
    incident_data.SourceSectionID = int(issuing_department) if issuing_department else 0
    incident_data.SourceDepartmentID = int(target_department) if target_department else 0
    incident_data.Source = int(source_1) if source_1 else 0
    incident_data.IncidentRequesterTypeID = int(feedback_type) if feedback_type else 1
    incident_data.Domain = domain or "General"
    incident_data.Category = category or "General"
    incident_data.SubCategory = sub_category or "General"
    incident_data.ClassificationAR = classification_ar or ""
    incident_data.ClassificationEN = classification_en or ""
    incident_data.Note = complaint_text or ""
    incident_data.ImmediateAction = immediate_action or ""
    incident_data.TakenAction = taken_action or ""
    incident_data.Severity = int(severity_level) if severity_level else 1
    incident_data.Stage = stage or ""
    incident_data.Harm = harm_level or ""
    incident_data.Status = int(status) if status else 1
    incident_data.ImprovementOpportunityType = bool(improvement_opportunity_type) if improvement_opportunity_type is not None else False

    # Extra fields from DB-layer if needed
    for key, value in extra_fields.items():
        setattr(incident_data, key, value)

    # Add the record via DB-layer
    new_id = IncidentRequestDB.add(incident_data)
    return new_id

def edit_incident(
    unique_id,
    feedback_received_date=None,
    record_id=None,
    patient_full_name=None,
    issuing_department=None,
    target_department=None,
    source_1=None,
    feedback_type=None,
    domain=None,
    category=None,
    sub_category=None,
    classification_ar=None,
    classification_en=None,
    complaint_text=None,
    immediate_action=None,
    taken_action=None,
    severity_level=None,
    stage=None,
    harm_level=None,
    status=None,
    improvement_opportunity_type=None,
    **extra_fields
) -> UniversalIncidentRecord:
    """
    Edit an existing incident and return the unified object with cases and actions.
    """
    # Build a dynamic object for DB-layer
    incident_data = type("IncidentData", (), {})()
    incident_data.UniqueID = unique_id

    # Map only non-None fields to DB-layer
    if feedback_received_date is not None:
        incident_data.DateAndTimeRecieved = feedback_received_date
    if record_id is not None:
        incident_data.Code = record_id
    if patient_full_name is not None:
        incident_data.PatientName = patient_full_name
    if issuing_department is not None:
        incident_data.SourceDepartmentName = issuing_department
    if target_department is not None:
        incident_data.SourceDepartmentID = target_department
    if source_1 is not None:
        incident_data.Source = source_1
    if feedback_type is not None:
        incident_data.IncidentRequesterTypeID = feedback_type
    if domain is not None:
        incident_data.Domain = domain
    if category is not None:
        incident_data.Category = category
    if sub_category is not None:
        incident_data.SubCategory = sub_category
    if classification_ar is not None:
        incident_data.ClassificationAR = classification_ar
    if classification_en is not None:
        incident_data.ClassificationEN = classification_en
    if complaint_text is not None:
        incident_data.Note = complaint_text
    if immediate_action is not None:
        incident_data.ImmediateAction = immediate_action
    if taken_action is not None:
        incident_data.TakenAction = taken_action
    if severity_level is not None:
        incident_data.Severity = severity_level
    if stage is not None:
        incident_data.Stage = stage
    if harm_level is not None:
        incident_data.Harm = harm_level
    if status is not None:
        incident_data.Status = status
    if improvement_opportunity_type is not None:
        incident_data.ImprovementOpportunityType = improvement_opportunity_type

    # Include extra fields if provided
    for key, value in extra_fields.items():
        setattr(incident_data, key, value)

    # ------------------ Update in DB ------------------
    IncidentRequestDB.edit(incident_data)

    # ------------------ Fetch updated record ------------------
    incident_record = IncidentRequestDB.get_records(search_name=record_id)
    if not incident_record:
        return None  # Record not found

    incident_dict = incident_record[0]

    # Build UniversalIncidentRecord
    universal_incident = UniversalIncidentRecord(
        feedback_received_date=incident_dict.get("DateAndTimeRecieved"),
        record_id=incident_dict.get("Code"),
        patient_full_name=incident_dict.get("PatientName"),
        issuing_department=incident_dict.get("SourceDepartmentName"),
        source_1=incident_dict.get("Source"),
        feedback_type=incident_dict.get("IncidentRequesterTypeID"),
    )

    # Attach cases and actions
    cases = IncidentRequestCaseDB.get_records(incident_request_id=incident_dict.get("UniqueID"))
    for case_dict in cases:
        incident_case = IncidentCase(
            domain=case_dict.get("IncidentCaseCategoryID"),
            category=case_dict.get("IncidentCaseSubCategoryID"),
            sub_category=case_dict.get("IncidentCaseSubCategoryID"),
            classification_ar=case_dict.get("ClassificationAR"),
            classification_en=case_dict.get("ClassificationEN"),
            complaint_text=case_dict.get("Description"),
            severity_level=case_dict.get("Severity"),
            stage=case_dict.get("Stage"),
            harm_level=case_dict.get("Harm"),
            status=case_dict.get("IncidentRequestCaseStatusID"),
            target_department=case_dict.get("SectionID"),
        )

        # Attach actions for this case
        actions = IncidentRequestCaseActionDB.get_records(case_id=case_dict.get("UniqueID"))
        for action_dict in actions:
            case_action = CaseAction(
                immediate_action=action_dict.get("Description"),
                taken_action=action_dict.get("SectionNote"),
                improvement_opportunity_type=action_dict.get("IsImprovementForm")
            )
            incident_case.actions.append(case_action)

        universal_incident.cases.append(incident_case)

    return universal_incident


def delete_incident(unique_id: int) -> bool:
    """
    Delete an incident record by its UniqueID.
    Returns True if deletion was successful, False otherwise.
    """
    try:
        # Delete related actions first
        cases = IncidentRequestCaseDB.get_records(incident_request_id=unique_id)
        for case in cases:
            case_id = case.get("UniqueID")
            IncidentRequestCaseActionDB.edit(case_id, {"Status": "Deleted"})  # Optional soft-delete
            # Or to hard delete, you can add a DB-layer delete function

        # Delete related cases
        for case in cases:
            case_id = case.get("UniqueID")
            # Implement hard delete if needed; for now, we could soft delete:
            IncidentRequestCaseDB.edit(case_id, {"Status": "Deleted"})

        # Delete main incident (soft delete example)
        IncidentRequestDB.edit(type("IncidentData", (), {"UniqueID": unique_id, "Status": "Deleted"})())

        return True
    except Exception as e:
        print(f"Error deleting incident: {e}")
        return False


def get_incident(unique_id: int = None, record_id: str = None) -> UniversalIncidentRecord:
    """
    Fetch a single incident by UniqueID or record_id.
    Returns a UniversalIncidentRecord including cases and actions.
    """
    if unique_id is None and record_id is None:
        raise ValueError("Either unique_id or record_id must be provided")

    # Fetch record
    if unique_id:
        records = IncidentRequestDB.get_records()
        incident_dict_list = [i for i in records if i.get("UniqueID") == unique_id]
    else:
        incident_dict_list = IncidentRequestDB.get_records(search_name=record_id)

    if not incident_dict_list:
        return None

    incident_dict = incident_dict_list[0]

    # Build UniversalIncidentRecord
    universal_incident = UniversalIncidentRecord(
        feedback_received_date=incident_dict.get("DateAndTimeRecieved"),
        record_id=incident_dict.get("Code"),
        patient_full_name=incident_dict.get("PatientName"),
        issuing_department=incident_dict.get("SourceDepartmentName"),
        source_1=incident_dict.get("Source"),
        feedback_type=incident_dict.get("IncidentRequesterTypeID"),
    )

    # Attach cases and actions
    cases = IncidentRequestCaseDB.get_records(incident_request_id=incident_dict.get("UniqueID"))
    for case_dict in cases:
        incident_case = IncidentCase(
            domain=case_dict.get("IncidentCaseCategoryID"),
            category=case_dict.get("IncidentCaseSubCategoryID"),
            sub_category=case_dict.get("IncidentCaseSubCategoryID"),
            classification_ar=case_dict.get("ClassificationAR"),
            classification_en=case_dict.get("ClassificationEN"),
            complaint_text=case_dict.get("Description"),
            severity_level=case_dict.get("Severity"),
            stage=case_dict.get("Stage"),
            harm_level=case_dict.get("Harm"),
            status=case_dict.get("IncidentRequestCaseStatusID"),
            target_department=case_dict.get("SectionID"),
        )

        # Attach actions for this case
        actions = IncidentRequestCaseActionDB.get_records(case_id=case_dict.get("UniqueID"))
        for action_dict in actions:
            case_action = CaseAction(
                immediate_action=action_dict.get("Description"),
                taken_action=action_dict.get("SectionNote"),
                improvement_opportunity_type=action_dict.get("IsImprovementForm")
            )
            incident_case.actions.append(case_action)

        universal_incident.cases.append(incident_case)

    return universal_incident
