import pyodbc
from datetime import datetime

# ==============================
# DB CONNECTION
# ==============================
def get_connection():
    return pyodbc.connect(
        "Driver={ODBC Driver 18 for SQL Server};"
        "Server=SOCIALMEDIA;"
        "Database=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

# ==============================
# DB LAYER CLASSES
# ==============================

class IncidentRequestDB:
    @staticmethod
    def add(incident_request):
        conn = get_connection()
        cursor = conn.cursor()
        query = """
        INSERT INTO IncidentRequest (
            YearCounter, IncidentRequesterTypeID, Code, PatientTypeID,
            DoctorID, EmployeeID, MRN, SourceBuilding, PatientName, PatientID,
            DateAndTimeRecieved, IncidentStatusID, IncidentSourceID, SourceSectionID,
            SourceDepartmentID, SourceDepartmentName, SourceAdminID, SourceAdminName,
            RequesterName, Note, IsFeedbackRequested, IsFeedbackGiven, IsInPatient,
            DateAndTimeFeedbackGiven, SatisfactoryID, DateAndTimeCreated, DateAndTimeUpdated,
            CreatedByApplicationUserID, UpdatedByApplicationUserID, Frozen
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """
        values = (
            incident_request.YearCounter,
            incident_request.IncidentRequesterTypeID,
            incident_request.Code,
            incident_request.PatientTypeID,
            incident_request.DoctorID,
            incident_request.EmployeeID,
            incident_request.MRN,
            incident_request.SourceBuilding,
            incident_request.PatientName,
            incident_request.PatientID,
            incident_request.DateAndTimeRecieved,
            incident_request.IncidentStatusID,
            incident_request.IncidentSourceID,
            incident_request.SourceSectionID,
            incident_request.SourceDepartmentID,
            incident_request.SourceDepartmentName,
            incident_request.SourceAdminID,
            incident_request.SourceAdminName,
            incident_request.RequesterName,
            incident_request.Note,
            incident_request.IsFeedbackRequested,
            incident_request.IsFeedbackGiven,
            incident_request.IsInPatient,
            incident_request.DateAndTimeFeedbackGiven,
            incident_request.SatisfactoryID,
            incident_request.DateAndTimeCreated,
            incident_request.DateAndTimeUpdated,
            incident_request.CreatedByApplicationUserID,
            incident_request.UpdatedByApplicationUserID,
            incident_request.Frozen
        )
        cursor.execute(query, values)
        conn.commit()
        new_id = cursor.execute("SELECT @@IDENTITY AS ID").fetchone()[0]
        conn.close()
        return new_id

    @staticmethod
    def edit(incident_request):
        conn = get_connection()
        cursor = conn.cursor()
        query = """
        UPDATE IncidentRequest SET
            YearCounter=?, IncidentRequesterTypeID=?, Code=?, PatientTypeID=?,
            DoctorID=?, EmployeeID=?, MRN=?, SourceBuilding=?, PatientName=?, PatientID=?,
            DateAndTimeRecieved=?, IncidentStatusID=?, IncidentSourceID=?, SourceSectionID=?,
            SourceDepartmentID=?, SourceDepartmentName=?, SourceAdminID=?, SourceAdminName=?,
            RequesterName=?, Note=?, IsFeedbackRequested=?, IsFeedbackGiven=?, IsInPatient=?,
            DateAndTimeFeedbackGiven=?, SatisfactoryID=?, DateAndTimeCreated=?, DateAndTimeUpdated=?,
            CreatedByApplicationUserID=?, UpdatedByApplicationUserID=?, Frozen=?
        WHERE UniqueID=?
        """
        values = (
            incident_request.YearCounter,
            incident_request.IncidentRequesterTypeID,
            incident_request.Code,
            incident_request.PatientTypeID,
            incident_request.DoctorID,
            incident_request.EmployeeID,
            incident_request.MRN,
            incident_request.SourceBuilding,
            incident_request.PatientName,
            incident_request.PatientID,
            incident_request.DateAndTimeRecieved,
            incident_request.IncidentStatusID,
            incident_request.IncidentSourceID,
            incident_request.SourceSectionID,
            incident_request.SourceDepartmentID,
            incident_request.SourceDepartmentName,
            incident_request.SourceAdminID,
            incident_request.SourceAdminName,
            incident_request.RequesterName,
            incident_request.Note,
            incident_request.IsFeedbackRequested,
            incident_request.IsFeedbackGiven,
            incident_request.IsInPatient,
            incident_request.DateAndTimeFeedbackGiven,
            incident_request.SatisfactoryID,
            incident_request.DateAndTimeCreated,
            incident_request.DateAndTimeUpdated,
            incident_request.CreatedByApplicationUserID,
            incident_request.UpdatedByApplicationUserID,
            incident_request.Frozen,
            incident_request.UniqueID
        )
        cursor.execute(query, values)
        conn.commit()
        conn.close()

class IncidentRequestCaseDB:

    @staticmethod
    def add(case_obj):
        """
        Add a new IncidentRequestCase record.
        case_obj: dict containing all needed fields for IncidentRequestCase
        """
        conn = get_connection()
        cursor = conn.cursor()

        columns = ", ".join(case_obj.keys())
        placeholders = ", ".join("?" for _ in case_obj)
        values = list(case_obj.values())

        query = f"INSERT INTO dbo.IncidentRequestCase ({columns}) VALUES ({placeholders})"
        cursor.execute(query, values)
        conn.commit()
        conn.close()

    @staticmethod
    def edit(case_id, update_obj):
        """
        Edit an existing IncidentRequestCase record.
        case_id: UniqueID of the case to update
        update_obj: dict containing fields to update
        """
        conn = get_connection()
        cursor = conn.cursor()

        set_clause = ", ".join(f"{k}=?" for k in update_obj)
        values = list(update_obj.values())
        values.append(case_id)

        query = f"UPDATE dbo.IncidentRequestCase SET {set_clause} WHERE UniqueID=?"
        cursor.execute(query, values)
        conn.commit()
        conn.close()

class IncidentRequestCaseActionDB:

    @staticmethod
    def add(action_obj):
        """
        Add a new IncidentRequestCaseAction record.
        action_obj: dict containing all needed fields for IncidentRequestCaseAction
        """
        conn = get_connection()
        cursor = conn.cursor()

        columns = ", ".join(action_obj.keys())
        placeholders = ", ".join("?" for _ in action_obj)
        values = list(action_obj.values())

        query = f"INSERT INTO dbo.IncidentRequestCaseAction ({columns}) VALUES ({placeholders})"
        cursor.execute(query, values)
        conn.commit()
        conn.close()

    @staticmethod
    def edit(action_id, update_obj):
        """
        Edit an existing IncidentRequestCaseAction record.
        action_id: UniqueID of the action to update
        update_obj: dict containing fields to update
        """
        conn = get_connection()
        cursor = conn.cursor()

        set_clause = ", ".join(f"{k}=?" for k in update_obj)
        values = list(update_obj.values())
        values.append(action_id)

        query = f"UPDATE dbo.IncidentRequestCaseAction SET {set_clause} WHERE UniqueID=?"
        cursor.execute(query, values)
        conn.commit()
        conn.close()