from api.universal_object import UniversalIncidentRecord, IncidentCase, CaseAction
from db_layer import IncidentRequestDB, IncidentRequestCaseDB, IncidentRequestCaseActionDB  # adjust import
from datetime import datetime
from encoding_mapper import map_feedback_to_ids



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
    Returns a list of UniversalIncidentRecord objects.
    """
    # ------------------ 1. Fetch incidents from DB ------------------
    filters = {
        "PatientName": search_name,
        "SourceSectionID": issuing,
        "SourceDepartmentID": target,
        "IncidentRequesterTypeID": feedback_category,
        "Severity": severity,
        "Stage": stage,
        "Harm": harm,
        "Domain": domain,
        "Category": category,
        "SubCategory": subcategory,
        "IncidentStatusID": status,
        "DateAndTimeRecieved__start": start_date,
        "DateAndTimeRecieved__end": end_date
    }
    incident_list = IncidentRequestDB.get_records(filters=filters)

    universal_incidents = []

    # ------------------ 2. Convert each incident ------------------
    for incident in incident_list:
        u_incident = UniversalIncidentRecord(
            unique_id=getattr(incident, "UniqueID", None),
            feedback_received_date=getattr(incident, "DateAndTimeRecieved", None),
            record_id=getattr(incident, "Code", None) or str(getattr(incident, "UniqueID", "")),
            patient_full_name=getattr(incident, "PatientName", None),
            issuing_department=getattr(incident, "SourceSectionID", None),
            source_1=getattr(incident, "IncidentSourceID", None),
            feedback_type=getattr(incident, "IncidentRequesterTypeID", None),
            cases=[]
        )

        if include_cases and u_incident.unique_id is not None:
            # Fetch related cases
            case_list = IncidentRequestCaseDB.get_records(filters={"IncidentRequestID": u_incident.unique_id})
            for case in case_list:
                incident_case = IncidentCase(
                    domain=case.domain,
                    category=case.category,
                    sub_category=case.sub_category,
                    classification_ar=case.classification_ar,
                    classification_en=case.classification_en,
                    complaint_text=case.complaint_text,
                    severity_level=case.severity_level,
                    stage=case.stage,
                    harm_level=case.harm_level,
                    status=case.status,
                    target_department=case.target_department,
                    actions=[]
                )

                if include_actions:
                    # Fetch related actions
                    actions_list = IncidentRequestCaseActionDB.get_records(
                        filters={"IncidentRequestCaseID": getattr(case, "UniqueID", None)}
                    )
                    for action in actions_list:
                        case_action = CaseAction(
                            immediate_action=action.immediate_action,
                            taken_action=action.taken_action,
                            improvement_opportunity_type=action.improvement_opportunity_type
                        )
                        incident_case.actions.append(case_action)

                # Append case to the incident
                u_incident.cases.append(incident_case)

        universal_incidents.append(u_incident)

    return universal_incidents

def add_incident(
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
    immediate_action =None,
    taken_action = None,
    severity_level=None,
    stage=None,
    harm_level=None,
    status=None,
    improvement_opportunity_type=None
):
    """
    Add a new incident record from raw UI data (strings/numbers).
    Converts string values to DB IDs using map_feedback_to_ids.
    Handles main incident, cases, and actions if provided in extra_fields['cases'].
    Returns the new incident ID.
    """

    # ------------------ 1. Map UI strings to IDs ------------------
    feedback_dict = {
        "issuing_department": issuing_department,
        "target_department": target_department,
        "source_1": source_1,
        "feedback_type": feedback_type,
        "domain": domain,
        "category": category,
        "subcategory": sub_category,
        "classification_ar": classification_ar,
        "classification_en": classification_en,
        "severity": severity_level,
        "stage": stage,
        "harm": harm_level,
        "status": status,
        "improvement_opportunity_type": improvement_opportunity_type,
        "immediate_action" : immediate_action,
        "taken_action":taken_action
    }

    ids_map = map_feedback_to_ids(feedback_dict)
    # ------------------ 2. Build main incident object ------------------
    incident_data = type("IncidentData", (), {})()
    incident_data.YearCounter = datetime.now().year
    incident_data.PatientTypeID = 0
    incident_data.DoctorID = 0
    incident_data.EmployeeID = 0
    incident_data.MRN = None
    incident_data.DateAndTimeRecieved = feedback_received_date or datetime.now()
    incident_data.Code = record_id or f"REC-{datetime.now().timestamp()}"
    incident_data.PatientName = patient_full_name or "Unknown"

    # Use mapped IDs and fallback to None (safe for FK)
    incident_data.SourceSectionID = ids_map.get("issuing_id") or None
    incident_data.SourceDepartmentID = None  # will set per case if needed
    incident_data.SourceAdminID = ids_map.get("admin_id") or None
    incident_data.Source = ids_map.get("source_id") or None
    incident_data.IncidentRequesterTypeID = ids_map.get("feedback_type_id") or None

    # Default values
    incident_data.Domain = "General"
    incident_data.Category = "General"
    incident_data.SubCategory = "General"
    incident_data.Note = complaint_text or ""
    incident_data.ClassificationAR = classification_ar or ""
    incident_data.ClassificationEN = classification_en or ""
    incident_data.Severity = ids_map.get("severity_id") or 1
    incident_data.Stage = stage or ""
    incident_data.Harm = harm_level or ""
    incident_data.Status = ids_map.get("status_id") or 1
    incident_data.ImprovementOpportunityType = bool(improvement_opportunity_type) if improvement_opportunity_type is not None else False

    # ------------------ 3. Add main incident ------------------
    new_incident_id = IncidentRequestDB.add(incident_data)

    return new_incident_id

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
    Soft-delete an incident record by its UniqueID, including related cases and actions.
    Returns True if deletion was successful, False otherwise.
    """
    try:
        # ------------------ 1. Fetch all cases for this incident ------------------
        cases = IncidentRequestCaseDB.get_records(filters={"IncidentRequestID": unique_id})

        # ------------------ 2. Delete related actions ------------------
        for case in cases:
            case_id = case.get("UniqueID")
            actions = IncidentRequestCaseActionDB.get_records(filters={"IncidentRequestCaseID": case_id})
            for action in actions:
                action_id = action.get("UniqueID")
                IncidentRequestCaseActionDB.edit(action_id, {"Status": "Deleted"})  # soft-delete

        # ------------------ 3. Delete related cases ------------------
        for case in cases:
            case_id = case.get("UniqueID")
            IncidentRequestCaseDB.edit(case_id, {"Status": "Deleted"})  # soft-delete

        # ------------------ 4. Delete main incident ------------------
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

    # ------------------ 1. Fetch the incident ------------------
    filters = {}
    if unique_id:
        filters["UniqueID"] = unique_id
    elif record_id:
        filters["Code"] = record_id

    incident_list = IncidentRequestDB.get_records(filters=filters)
    if not incident_list:
        return None

    incident_dict = incident_list[0]

    # ------------------ 2. Build UniversalIncidentRecord ------------------
    universal_incident = UniversalIncidentRecord(
        feedback_received_date=incident_dict.get("DateAndTimeRecieved"),
        record_id=incident_dict.get("Code"),
        patient_full_name=incident_dict.get("PatientName"),
        issuing_department=incident_dict.get("SourceSectionID"),
        source_1=incident_dict.get("IncidentSourceID"),
        feedback_type=incident_dict.get("IncidentRequesterTypeID"),
    )

    # ------------------ 3. Attach cases ------------------
    cases = IncidentRequestCaseDB.get_records(filters={"IncidentRequestID": incident_dict.get("UniqueID")})
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

        # ------------------ 4. Attach actions for this case ------------------
        actions = IncidentRequestCaseActionDB.get_records(filters={"IncidentRequestCaseID": case_dict.get("UniqueID")})
        for action_dict in actions:
            case_action = CaseAction(
                immediate_action=action_dict.get("Description"),
                taken_action=action_dict.get("SectionNote") or action_dict.get("DepartmentNote"),
                improvement_opportunity_type=action_dict.get("IsImprovementForm")
            )
            incident_case.actions.append(case_action)

        universal_incident.cases.append(incident_case)

    return universal_incident
