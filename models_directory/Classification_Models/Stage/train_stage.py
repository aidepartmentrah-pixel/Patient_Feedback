from models_directory.Classification_Models.Stage.Training_Internal_Metrics.Internal_metric_training_functions import \
    Embedd_Normalize, train_ML_Metric_Reader


def train_stage():
    METRICS = [
            "administration_delay", # ADMISSION_KEYWORDS
            "arrival", # ADMISSION_KEYWORDS
            "bed_room", # ADMISSION_KEYWORDS
            "billing_issues", # ADMISSION_KEYWORDS / DISCHARGE_KEYWORDS
            "cleanliness", # CARE_ON_WARD_KEYWORDS / CARE_ON_WARD_KEYWORDS
            "clinical_delay", # EXAMINATION_DIAGNOSIS_KEYWORDS /
            "communication_scheduling", # OPERATION_KEYWORDS
            "conflicting_or_wrong_diagnosis", #EXAMINATION_DIAGNOSIS_KEYWORDS
            "disagreement_with_discharge", # DISCHARGE_KEYWORDS
            "food", # CARE_ON_WARD_KEYWORDS
            "location", # CARE_ON_WARD_KEYWORDS
            "medical_clinical_errors", # OPERATION_KEYWORDS
            "staff_security_behavior" # CARE_ON_WARD_KEYWORDS
        ]


    for metric in METRICS:
        Embedd_Normalize(metric)
        train_ML_Metric_Reader(metric)

    print(f"Stage Training Complete !!! ")
