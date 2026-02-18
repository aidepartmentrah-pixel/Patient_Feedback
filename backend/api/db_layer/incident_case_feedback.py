from core.database import get_connection

# -----------------------------
# CREATE
# -----------------------------

def create_incident_case_feedback(
    incident_id: int,
    feedback_data: dict,
    created_by_user_id: int,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    INSERT INTO dbo.APP_IncidentCaseFeedback (
        IncidentRequestCaseID,

        Cause_Staff_Training,
        Cause_Staff_Incentives,
        Cause_Staff_Competency,
        Cause_Staff_Understaffed,
        Cause_Staff_NonCompliance,
        Cause_Staff_NoCoordination,
        Cause_Staff_Other,
        Cause_Staff_OtherText,

        Cause_Process_NotComprehensive,
        Cause_Process_Unclear,
        Cause_Process_MissingProtocol,
        Cause_Process_Other,
        Cause_Process_OtherText,

        Cause_Equipment_NotAvailable,
        Cause_Equipment_SystemIncomplete,
        Cause_Equipment_HardToApply,
        Cause_Equipment_Other,
        Cause_Equipment_OtherText,

        Cause_Environment_PlaceNature,
        Cause_Environment_Surroundings,
        Cause_Environment_WorkConditions,
        Cause_Environment_Other,
        Cause_Environment_OtherText,

        Preventive_MonthlyMeetings,
        Preventive_TrainingPrograms,
        Preventive_IncreaseStaff,
        Preventive_MMCommitteeActions,
        Preventive_Other,
        Preventive_OtherText,

        DepartmentExplanationText,
        DepartmentExplanationStatusID,
        DepartmentExplanationReceivalDate,

        CreatedByUserID
    )
    VALUES (
        ?,  ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
        ?
    )
    """

    params = [
        incident_id,

        feedback_data.get("Cause_Staff_Training"),
        feedback_data.get("Cause_Staff_Incentives"),
        feedback_data.get("Cause_Staff_Competency"),
        feedback_data.get("Cause_Staff_Understaffed"),
        feedback_data.get("Cause_Staff_NonCompliance"),
        feedback_data.get("Cause_Staff_NoCoordination"),
        feedback_data.get("Cause_Staff_Other"),
        feedback_data.get("Cause_Staff_OtherText"),

        feedback_data.get("Cause_Process_NotComprehensive"),
        feedback_data.get("Cause_Process_Unclear"),
        feedback_data.get("Cause_Process_MissingProtocol"),
        feedback_data.get("Cause_Process_Other"),
        feedback_data.get("Cause_Process_OtherText"),

        feedback_data.get("Cause_Equipment_NotAvailable"),
        feedback_data.get("Cause_Equipment_SystemIncomplete"),
        feedback_data.get("Cause_Equipment_HardToApply"),
        feedback_data.get("Cause_Equipment_Other"),
        feedback_data.get("Cause_Equipment_OtherText"),

        feedback_data.get("Cause_Environment_PlaceNature"),
        feedback_data.get("Cause_Environment_Surroundings"),
        feedback_data.get("Cause_Environment_WorkConditions"),
        feedback_data.get("Cause_Environment_Other"),
        feedback_data.get("Cause_Environment_OtherText"),

        feedback_data.get("Preventive_MonthlyMeetings"),
        feedback_data.get("Preventive_TrainingPrograms"),
        feedback_data.get("Preventive_IncreaseStaff"),
        feedback_data.get("Preventive_MMCommitteeActions"),
        feedback_data.get("Preventive_Other"),
        feedback_data.get("Preventive_OtherText"),

        feedback_data.get("DepartmentExplanationText"),
        feedback_data.get("DepartmentExplanationStatusID"),
        feedback_data.get("DepartmentExplanationReceivalDate"),

        created_by_user_id,
    ]

    cursor.execute(sql, params)
    conn.commit()
    conn.close()


# -----------------------------
# READ
# -----------------------------

def get_incident_case_feedback(incident_id: int) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM dbo.APP_IncidentCaseFeedback
        WHERE IncidentRequestCaseID = ?
        """,
        incident_id,
    )

    row = cursor.fetchone()
    columns = [col[0] for col in cursor.description]

    conn.close()

    return dict(zip(columns, row)) if row else None


# -----------------------------
# UPDATE
# -----------------------------

def update_incident_case_feedback(
    incident_id: int,
    updates: dict,
) -> None:
    """
    Partial update of feedback record.
    """
    fields = []
    values = []

    for key, value in updates.items():
        fields.append(f"{key} = ?")
        values.append(value)

    if not fields:
        return

    values.append(incident_id)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"""
        UPDATE dbo.APP_IncidentCaseFeedback
        SET {", ".join(fields)}
        WHERE IncidentRequestCaseID = ?
        """,
        values,
    )

    conn.commit()
    conn.close()
