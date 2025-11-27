import pyodbc
from api.universal_object import UniversalIncidentRecord, IncidentCase,CaseAction
from datetime import datetime


def get_connection():
    return pyodbc.connect(
        "Driver={ODBC Driver 18 for SQL Server};"
        "Server=SOCIALMEDIA;"
        "Database=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

class IncidentRequestDB:

    @staticmethod
    def add(incident_request):
        conn = get_connection()
        cursor = conn.cursor()

        # ------------------ Dynamically gather attributes ------------------
        # Only include attributes that the object actually has
        columns = []
        values = []
        for column in [
            "YearCounter", "IncidentRequesterTypeID", "Code", "PatientTypeID",
            "DoctorID", "EmployeeID", "MRN", "SourceBuilding", "PatientName",
            "PatientID", "DateAndTimeRecieved", "IncidentStatusID", "IncidentSourceID",
            "SourceSectionID", "SourceDepartmentID", "SourceDepartmentName",
            "SourceAdminID", "SourceAdminName", "RequesterName", "Note",
            "IsFeedbackRequested", "IsFeedbackGiven", "IsInPatient",
            "DateAndTimeFeedbackGiven", "SatisfactoryID", "DateAndTimeCreated",
            "DateAndTimeUpdated", "CreatedByApplicationUserID", "UpdatedByApplicationUserID",
            "Frozen"
        ]:
            if hasattr(incident_request, column):
                columns.append(column)
                values.append(getattr(incident_request, column))

        placeholders = ", ".join("?" for _ in values)
        columns_str = ", ".join(columns)

        query = f"INSERT INTO IncidentRequest ({columns_str}) VALUES ({placeholders})"
        cursor.execute(query, values)
        conn.commit()

        cursor.execute(query, values)  # Insert your record
        cursor.execute("SELECT SCOPE_IDENTITY()")  # Fetch the last inserted ID in this scope
        new_id = cursor.fetchone()[0]  # Get the ID from the result
        conn.commit()
        conn.close()
        return new_id

    @staticmethod
    def edit(incident_request):
        """
        Update an existing incident in DB.
        Accepts either a DB-layer object or UniversalIncidentRecord.
        """
        if isinstance(incident_request, UniversalIncidentRecord):
            incident_request = IncidentRequestDB.from_universal(incident_request)

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

    @staticmethod
    def get_records(filters: dict = None):
        """
        Fetch incidents from DB and return as a list of UniversalIncidentRecord objects.
        """
        conn = get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM IncidentRequest WHERE 1=1"
        params = []

        if filters:
            for key, value in filters.items():
                if value is not None:
                    query += f" AND {key} = ?"
                    params.append(value)

        cursor.execute(query, params)
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        # Convert each row to UniversalIncidentRecord
        universal_list = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            universal_list.append(IncidentRequestDB.to_universal(row_dict))

        return universal_list

    @staticmethod
    def to_universal(row: dict) -> UniversalIncidentRecord:
        """
        Convert a DB row dict to a UniversalIncidentRecord object.
        """
        return UniversalIncidentRecord(
            feedback_received_date=row.get("DateAndTimeRecieved"),
            record_id=row.get("Code"),
            patient_full_name=row.get("PatientName"),
            issuing_department=row.get("SourceSectionID"),
            source_1=row.get("IncidentSourceID"),
            feedback_type=row.get("IncidentRequesterTypeID"),
            cases=[]  # cases will be attached separately
        )

    @staticmethod
    def from_universal(universal: UniversalIncidentRecord):
        """
        Convert a UniversalIncidentRecord into a DB-ready object.
        Returns a simple object with attributes matching DB columns.
        """
        db_obj = type("IncidentData", (), {})()
        db_obj.YearCounter = datetime.now().year
        db_obj.IncidentRequesterTypeID = universal.feedback_type or 1
        db_obj.Code = universal.record_id or f"REC-{datetime.now().timestamp()}"
        db_obj.PatientTypeID = 0
        db_obj.DoctorID = 0
        db_obj.EmployeeID = 0
        db_obj.MRN = None
        db_obj.SourceBuilding = ""
        db_obj.PatientName = universal.patient_full_name or "Unknown"
        db_obj.PatientID = None
        db_obj.DateAndTimeRecieved = universal.feedback_received_date or datetime.now()
        db_obj.IncidentStatusID = 1
        db_obj.IncidentSourceID = universal.source_1 or 0
        db_obj.SourceSectionID = universal.issuing_department or 0
        db_obj.SourceDepartmentID = 0
        db_obj.SourceDepartmentName = ""
        db_obj.SourceAdminID = 0
        db_obj.SourceAdminName = ""
        db_obj.RequesterName = ""
        db_obj.Note = ""
        db_obj.IsFeedbackRequested = False
        db_obj.IsFeedbackGiven = False
        db_obj.IsInPatient = False
        db_obj.DateAndTimeFeedbackGiven = None
        db_obj.SatisfactoryID = None
        db_obj.DateAndTimeCreated = datetime.now()
        db_obj.DateAndTimeUpdated = datetime.now()
        db_obj.CreatedByApplicationUserID = 0
        db_obj.UpdatedByApplicationUserID = 0
        db_obj.Frozen = False
        db_obj.UniqueID = getattr(universal, "unique_id", None)
        return db_obj

class IncidentRequestCaseDB:

    @staticmethod
    def add(case_obj):
        """
        Add a new IncidentRequestCase record.
        Accepts either a dict or an IncidentCase object (Universal).
        """
        # Convert Universal object to DB-ready dict if needed
        if isinstance(case_obj, IncidentCase):
            case_obj = IncidentRequestCaseDB.from_universal(case_obj)

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
    def edit(case_id: int, update_obj):
        """
        Edit an existing IncidentRequestCase record.
        Accepts either a dict or an IncidentCase object.
        """
        if isinstance(update_obj, IncidentCase):
            update_obj = IncidentRequestCaseDB.from_universal(update_obj)

        conn = get_connection()
        cursor = conn.cursor()

        set_clause = ", ".join(f"{k}=?" for k in update_obj)
        values = list(update_obj.values())
        values.append(case_id)

        query = f"UPDATE dbo.IncidentRequestCase SET {set_clause} WHERE UniqueID=?"
        cursor.execute(query, values)
        conn.commit()
        conn.close()

    @staticmethod
    def get_records(filters: dict = None):
        """
        Fetch IncidentRequestCase records with optional filters.
        Returns a list of IncidentCase objects.
        """
        conn = get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM dbo.IncidentRequestCase WHERE 1=1"
        params = []

        if filters:
            for key, value in filters.items():
                if value is not None:
                    query += f" AND {key} = ?"
                    params.append(value)

        cursor.execute(query, params)
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        # Convert each row to IncidentCase (Universal)
        universal_cases = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            universal_cases.append(IncidentRequestCaseDB.to_universal(row_dict))

        return universal_cases

    @staticmethod
    def to_universal(row: dict) -> IncidentCase:
        """
        Convert a DB row dict to an IncidentCase object.
        """
        return IncidentCase(
            domain=row.get("IncidentCaseCategoryID"),
            category=row.get("IncidentCaseSubCategoryID"),
            sub_category=row.get("IncidentCaseSubCategoryID"),
            classification_ar=row.get("ClassificationAR"),
            classification_en=row.get("ClassificationEN"),
            complaint_text=row.get("Description"),
            severity_level=row.get("Severity"),
            stage=row.get("Stage"),
            harm_level=row.get("Harm"),
            status=row.get("IncidentRequestCaseStatusID"),
            target_department=row.get("SectionID"),
            actions=[]  # actions will be attached separately
        )

    @staticmethod
    def from_universal(universal: IncidentCase):
        """
        Convert an IncidentCase (Universal) to a DB-ready dict.
        """
        return {
            "IncidentCaseCategoryID": universal.domain or 0,
            "IncidentCaseSubCategoryID": universal.category or 0,
            "ClassificationAR": universal.classification_ar or "",
            "ClassificationEN": universal.classification_en or "",
            "Description": universal.complaint_text or "",
            "Severity": universal.severity_level or 1,
            "Stage": universal.stage or "",
            "Harm": universal.harm_level or "",
            "IncidentRequestCaseStatusID": universal.status or 1,
            "SectionID": universal.target_department or 0
        }

class IncidentRequestCaseActionDB:

    @staticmethod
    def add(action_obj):
        """
        Add a new IncidentRequestCaseAction record.
        Accepts either a dict or a CaseAction object (Universal).
        """
        # Convert Universal object to DB-ready dict if needed
        if isinstance(action_obj, CaseAction):
            action_obj = IncidentRequestCaseActionDB.from_universal(action_obj)

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
    def edit(action_id: int, update_obj):
        """
        Edit an existing IncidentRequestCaseAction record.
        Accepts either a dict or a CaseAction object.
        """
        if isinstance(update_obj, CaseAction):
            update_obj = IncidentRequestCaseActionDB.from_universal(update_obj)

        conn = get_connection()
        cursor = conn.cursor()

        set_clause = ", ".join(f"{k}=?" for k in update_obj)
        values = list(update_obj.values())
        values.append(action_id)

        query = f"UPDATE dbo.IncidentRequestCaseAction SET {set_clause} WHERE UniqueID=?"
        cursor.execute(query, values)
        conn.commit()
        conn.close()

    @staticmethod
    def get_records(filters: dict = None):
        """
        Fetch IncidentRequestCaseAction records with optional filters.
        Returns a list of CaseAction objects.
        """
        conn = get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM dbo.IncidentRequestCaseAction WHERE 1=1"
        params = []

        if filters:
            for key, value in filters.items():
                if value is not None:
                    query += f" AND {key} = ?"
                    params.append(value)

        cursor.execute(query, params)
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        # Convert each row to CaseAction (Universal)
        universal_actions = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            universal_actions.append(IncidentRequestCaseActionDB.to_universal(row_dict))

        return universal_actions

    @staticmethod
    def to_universal(row: dict) -> CaseAction:
        """
        Convert a DB row dict to a CaseAction object.
        """
        return CaseAction(
            immediate_action=row.get("Description"),
            taken_action=row.get("SectionNote") or row.get("DepartmentNote"),
            improvement_opportunity_type=row.get("IsImprovementForm")
        )

    @staticmethod
    def from_universal(universal: CaseAction):
        """
        Convert a CaseAction (Universal) to a DB-ready dict.
        """
        return {
            "Description": universal.immediate_action or "",
            "SectionNote": universal.taken_action or "",
            "IsImprovementForm": bool(universal.improvement_opportunity_type) if universal.improvement_opportunity_type is not None else False
        }
