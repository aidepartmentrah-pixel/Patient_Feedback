import os

def get_project_root():
    """Always returns the Patient_Feedback root directory."""
    return os.path.abspath(
        os.path.dirname(__file__)
    )


def get_db_path():
    return os.path.join(get_project_root(), "models_directory", "patient_feedback_ml.db")
