from models_directory.Classification_Models.Hierarchical_Classification_Model.hierarchical_predictor import hierarchical_predict_embeddings
from models_directory.Classification_Models.Severity_level.predict_severity import predict_severity_from_embedding
from models_directory.Classification_Models.Stage.model_package import classify_stage_Score_Based
from models_directory.Classification_Models.Harm_level.predict_harm import predict_harm_from_embedding
from models_directory.Classification_Models.Stage.modular_functions import get_embedding
from models_directory.Classification_Models.Classification_En.predict_classification_en import predict_classification_en_from_embedding


#Not Implemented
from models_directory.Classification_Models.improvement_opportunity_type.predict_improvement import predict_improvement_from_embedding
from models_directory.Classification_Models.feedback_type.predict_feedback_type import predict_feedback_type_from_embedding


import numpy as np
import time



DOMAIN_MAP = {
    1: "CLINICAL",
    2: "MANAGEMENT",
    3: "RELATIONAL",
}

CATEGORY_MAP = {
    1: "Communication",
    2: "Environement",
    3: "Institutional Processes",
    4: "Listening",
    5: "Quality of Care",
    6: "Respect & Patient Rights",
    7: "Safety",
}

SUBCATEGORY_MAP = {
    1: "Neglect - General",
    2: "Absent Communication",
    3: "Accomodation",
    4: "Bureaucracy",
    5: "Clinician - Errors",
    6: "Delay - Access",
    8: "Delay - General",
    9: "Delay - Procedure",
    10: "Delayed Communication",
    11: "Dismissing Patients",
    12: "Disrespect",
    13: "Documentation",
    14: "Equipment",
    15: "Error - Diagnosis",
    16: "Error - General",
    18: "Error - Medication",
    19: "Examination & Monitoring",
    21: "Failure to Provide",
    22: "Failure to Respond",
    23: "Ignoring Patients",
    24: "Incorrect Communication",
    26: "Neglect - Hygiene & Personal Care",
    27: "Rights",
    28: "Security",
    29: "Teamwork",
    30: "Visiting",
    31: "Ward Cleanliness",
}

# ============================================================
# MODEL LABEL MAPS — reflect what model was trained to output
# DO NOT change these to match DB — they represent model semantics
# ============================================================
SEVERITY_MODEL_MAP = {
    1: "HIGH",
    2: "LOW",
    3: "MEDIUM",
}

HARM_MODEL_MAP = {
    1: "Severe Harm",
    2: "Death",
    3: "High Severe",
    4: "Minor Harm",
    5: "Moderate Harm",
    6: "No Harm",
}

# ============================================================
# DATABASE LABEL MAPS — reflect APP_LOOKUP tables in database
# These are used for API response labels
# ============================================================
DB_SEVERITY_LABELS = {
    1: "Low",
    2: "Medium",
    3: "High",
}

DB_HARM_LABELS = {
    1: "No Harm",
    2: "Minor",
    3: "Moderate",
    4: "Severe",
    5: "Death",
}

# Legacy alias for backward compatibility (now points to DB labels)
SEVERITY_MAP = DB_SEVERITY_LABELS
HARM_MAP = DB_HARM_LABELS

STAGE_MAP = {
    1: "Examination & Diagnosis",
    2: "Admission",
    3: "Care on the Ward",
    4: "Operation / Procedure",
    5: "Discharge / Transfer",
    6: "Unspecified",
}


# ============================================================
# ADAPTER FUNCTIONS — convert model output to database IDs
# ============================================================
def adapt_severity_to_db(model_value: int) -> int:
    """
    Convert model severity output to database ID.
    Model: 1=HIGH, 2=LOW, 3=MEDIUM
    DB:    1=Low,  2=Medium, 3=High
    """
    MODEL_TO_DB = {
        1: 3,  # HIGH → DB High (3)
        2: 1,  # LOW → DB Low (1)
        3: 2,  # MEDIUM → DB Medium (2)
    }
    db_id = MODEL_TO_DB.get(model_value)
    if db_id is None:
        print(f"[WARNING] Severity model returned unexpected value: {model_value}. Defaulting to 1 (Low).")
        return 1
    return db_id


def adapt_harm_to_db(model_value: int) -> int:
    """
    Convert model harm output to database ID.
    Model: 1=Severe, 2=Death, 3=High Severe, 4=Minor, 5=Moderate, 6=No Harm
    DB:    1=No Harm, 2=Minor, 3=Moderate, 4=Severe, 5=Death
    """
    MODEL_TO_DB = {
        1: 4,  # Severe Harm → DB Severe (4)
        2: 5,  # Death → DB Death (5)
        3: 3,  # High Severe → DB Moderate (3) [closest match]
        4: 2,  # Minor Harm → DB Minor (2)
        5: 3,  # Moderate Harm → DB Moderate (3)
        6: 1,  # No Harm → DB No Harm (1)
    }
    db_id = MODEL_TO_DB.get(model_value)
    if db_id is None:
        print(f"[WARNING] Harm model returned unexpected value: {model_value}. Defaulting to 1 (No Harm).")
        return 1
    return db_id

FEEDBACK_TYPE_MAP = {
    1: "Complaint",
    2: "Compliment",
    3: "Suggestion",
    4: "Other",
}

IMPROVEMENT_OPPORTUNITY_MAP = {
    1: "Process Improvement",
    2: "Training Needed",
    3: "Resource Allocation",
}

CLASSIFICATION_EN_MAP = {
    78: "Classification 6",
    79: "Classification 27",
    80: "Classification 51",
    81: "Classification 44",
    82: "Classification 45",
    83: "Classification 49",
    84: "Classification 31",
    85: "Classification 35",
    86: "Classification 67",
    88: "Classification 8",
    89: "Classification 7",
    90: "Classification 10",
    91: "Classification 47",
    93: "Classification 57",
    94: "Classification 58",
    95: "Classification 59",
    96: "Classification 60",
    98: "Classification 62",
    99: "Classification 63",
    100: "Classification 22",
    102: "Classification 48",
    103: "Classification 37",
    104: "Classification 52",
    105: "Classification 68",
    106: "Classification 13",
    107: "Classification 15",
    108: "Classification 16",
    109: "Classification 33",
    110: "Classification 55",
    111: "Classification 32",
    112: "Classification 64",
    113: "Classification 66",
    114: "Classification 3",
    115: "Classification 12",
    116: "Classification 19",
    117: "Classification 20",
    118: "Classification 21",
    120: "Classification 2",
    121: "Classification 18",
    122: "Classification 26",
    123: "Classification 34",
    125: "Classification 24",
    126: "Classification 28",
    127: "Classification 29",
    128: "Classification 30",
    129: "Classification 4",
    130: "Classification 73",
    131: "Classification 36",
    132: "Classification 17",
    133: "Classification 23",
    134: "Classification 40",
    135: "Classification 43",
    136: "Classification 42",
    137: "Classification 70",
    138: "Classification 61",
    139: "Classification 65",
    140: "Classification 50",
    141: "Classification 1",
    142: "Classification 53",
    143: "Classification 5",
    144: "Classification 38",
    145: "Classification 11",
    146: "Classification 14",
    148: "Classification 41",
    149: "Classification 56",
    150: "Classification 71",
    151: "Classification 72",
    152: "Classification 39",
    153: "Classification 69",
    154: "Classification 46",
}



def classify_feedback(patient_text, text_2, text_3, Print = False):

    #Text Identification
    Patient_Text = patient_text

    #Text Embedding
    Patient_Embedding_RAW = get_embedding(Patient_Text)
    Patient_Embedding = np.frombuffer(Patient_Embedding_RAW, dtype=np.float32)




    result_embedding = hierarchical_predict_embeddings(Patient_Embedding)
    domain_id = result_embedding["domain"]
    category_id = result_embedding["category"]
    sub_category_id = result_embedding["subcategory"]


    # Severity prediction with adapter
    raw_severity = predict_severity_from_embedding(Patient_Embedding)
    severity_level_id = adapt_severity_to_db(raw_severity)
    print(f"[DEBUG] Severity: raw_model={raw_severity} ({SEVERITY_MODEL_MAP.get(raw_severity, 'UNKNOWN')}) → db_id={severity_level_id} ({DB_SEVERITY_LABELS.get(severity_level_id, 'UNKNOWN')})")
    
    # Stage classification returns a dict with chosen_stage being a string
    stage_result = classify_stage_Score_Based(Patient_Text, Print)
    # Map stage names to IDs matching reference API:
    # 1=Examination & Diagnosis, 2=Admission, 3=Care on Ward, 4=Operation, 5=Discharge, 6=Unspecified
    STAGE_ENCODINGS = {
        "examination_diagnosis": 1,
        "admission": 2,
        "care_on_ward": 3,
        "discharge": 5,
        "operation": 4,
        "unspecified": 6
    }
    chosen_stage = stage_result.get("chosen_stage") if isinstance(stage_result, dict) else None
    stage_id = STAGE_ENCODINGS.get(chosen_stage, 6) if chosen_stage else 6  # default to unspecified (id=6)

    # Harm prediction with adapter
    harm_result = predict_harm_from_embedding(Patient_Embedding)
    raw_harm = harm_result["harm_level"]
    harm_level_id = adapt_harm_to_db(raw_harm)
    print(f"[DEBUG] Harm: raw_model={raw_harm} ({HARM_MODEL_MAP.get(raw_harm, 'UNKNOWN')}) → db_id={harm_level_id} ({DB_HARM_LABELS.get(harm_level_id, 'UNKNOWN')})")

    # feedback_type and improvement functions return strings directly
    feedback_type_label = predict_feedback_type_from_embedding(Patient_Embedding)
    improvement_opportunity_label = predict_improvement_from_embedding(Patient_Embedding)
    classification_en_id = predict_classification_en_from_embedding(Patient_Embedding)

    # Reverse lookup for IDs (for compatibility)
    FEEDBACK_TYPE_REVERSE = {v: k for k, v in FEEDBACK_TYPE_MAP.items()}
    IMPROVEMENT_REVERSE = {v: k for k, v in IMPROVEMENT_OPPORTUNITY_MAP.items()}
    # Also check for actual returned labels from model
    FEEDBACK_LABEL_TO_ID = {
        "Improvement Opportunity": 1,
        "Notice": 2, 
        "Critique Suggestion": 3,
        "Other": 4,
        "Complaint": 1,
        "Compliment": 2,
        "Suggestion": 3,
    }
    IMPROVEMENT_LABEL_TO_ID = {
        "Ordinary Complaint": 1,
        "Red Flag": 2,
        "Never Event": 3,
        "Process Improvement": 1,
        "Training Needed": 2,
        "Resource Allocation": 3,
    }
    feedback_type_id = FEEDBACK_TYPE_REVERSE.get(feedback_type_label, FEEDBACK_LABEL_TO_ID.get(feedback_type_label, 4))
    improvement_opportunity_id = IMPROVEMENT_REVERSE.get(improvement_opportunity_label, IMPROVEMENT_LABEL_TO_ID.get(improvement_opportunity_label, 1))

    result = {
        "domain_id": domain_id,
        "domain": DOMAIN_MAP.get(domain_id, f"UNKNOWN ({domain_id})"),

        "category_id": category_id,
        "category": CATEGORY_MAP.get(category_id, f"UNKNOWN ({category_id})"),

        "sub_category_id": sub_category_id,
        "sub_category": SUBCATEGORY_MAP.get(sub_category_id, f"UNKNOWN ({sub_category_id})"),

        "severity_id": severity_level_id,
        "severity_level": SEVERITY_MAP.get(severity_level_id, f"UNKNOWN ({severity_level_id})"),

        "stage_id": stage_id,
        "stage": STAGE_MAP.get(stage_id, f"UNKNOWN ({stage_id})"),

        "harm_level_id": harm_level_id,
        "harm_level": HARM_MAP.get(harm_level_id, f"UNKNOWN ({harm_level_id})"),

        "feedback_type_id": feedback_type_id,
        "feedback_type": FEEDBACK_TYPE_MAP.get(feedback_type_id, f"UNKNOWN ({feedback_type_id})"),

        "improvement_opportunity_type_id": improvement_opportunity_id,
        "improvement_opportunity_type": IMPROVEMENT_OPPORTUNITY_MAP.get(improvement_opportunity_id, f"UNKNOWN ({improvement_opportunity_id})"),

        "classification_en_id": classification_en_id,
        "classification_en": CLASSIFICATION_EN_MAP.get(classification_en_id, f"UNKNOWN ({classification_en_id})"),
    }

    # ---------------------------------------------------------
    # READABLE DESCRIPTION
    # ---------------------------------------------------------
    if Print:
        print(f"The Feedback: {patient_text}")

        print("\n================ CLASSIFICATION RESULT ================\n")
        print(f"DOMAIN      : {result['domain']} ({result['domain_id']})")
        print(f"CATEGORY    : {result['category']} ({result['category_id']})")
        print(f"SUBCATEGORY : {result['sub_category']} ({result['sub_category_id']})")
        print(f"SEVERITY    : {result['severity_level']} ({result['severity_id']})")
        print(f"STAGE       : {result['stage']} ({result['stage_id']})")
        print(f"HARM LEVEL  : {result['harm_level']} ({result['harm_level_id']})")
        print(f"FEEDBACK TYPE: {result['feedback_type']} ({result['feedback_type_id']})")
        print(f"IMPROVEMENT : {result['improvement_opportunity_type']} ({result['improvement_opportunity_type_id']})")
        print(f"CLASSIFICATION EN: {result['classification_en']} ({result['classification_en_id']})")
        print("\n========================================================\n")
    return result


def classify_feedback_timed(text_1, text_2, text_3, Print=False):
    timings = {}

    # -------------------------------
    # Text Identification
    # -------------------------------
    start = time.time()
    Patient_Text = text_1
    Hospital_Text = text_2 + " " + text_3
    Combined_Text = Patient_Text + " " + Hospital_Text
    timings["text_processing"] = time.time() - start

    # -------------------------------
    # Text Embedding
    # -------------------------------
    start = time.time()
    Patient_Embedding_RAW = get_embedding(Patient_Text)
    Patient_Embedding = np.frombuffer(Patient_Embedding_RAW, dtype=np.float32)

    Hospital_Embedding_RAW = get_embedding(Hospital_Text)
    Hospital_Embedding = np.frombuffer(Hospital_Embedding_RAW, dtype=np.float32)

    Combined_Embedding_RAW = get_embedding(Combined_Text)
    Combined_Embedding = np.frombuffer(Combined_Embedding_RAW, dtype=np.float32)
    timings["embedding"] = time.time() - start

    # -------------------------------
    # Hierarchical Prediction
    # -------------------------------
    start = time.time()
    result_embedding = hierarchical_predict_embeddings(Patient_Embedding)
    domain_id = result_embedding["domain"]
    category_id = result_embedding["category"]
    sub_category_id = result_embedding["subcategory"]
    timings["hierarchical_prediction"] = time.time() - start

    # -------------------------------
    # Severity Prediction
    # -------------------------------
    start = time.time()
    severity_level_id = predict_severity_from_embedding(Patient_Embedding)
    timings["severity_prediction"] = time.time() - start

    # -------------------------------
    # Stage Classification
    # -------------------------------
    start = time.time()
    stage_id = classify_stage_Score_Based(Patient_Text)
    timings["stage_classification"] = time.time() - start

    # -------------------------------
    # Harm Prediction
    # -------------------------------
    start = time.time()
    harm_result = predict_harm_from_embedding(Combined_Embedding)
    harm_level_id = harm_result["harm_level"]
    timings["harm_prediction"] = time.time() - start

    # -------------------------------
    # Feedback Type Prediction
    # -------------------------------
    start = time.time()
    feedback_type_id = predict_feedback_type_from_embedding(Patient_Embedding)
    timings["feedback_type_prediction"] = time.time() - start

    # -------------------------------
    # Improvement Opportunity Prediction
    # -------------------------------
    start = time.time()
    improvement_opportunity_id = predict_improvement_from_embedding(Patient_Embedding)
    timings["improvement_opportunity_prediction"] = time.time() - start

    # -------------------------------
    # Aggregate Results
    # -------------------------------
    result = {
        "domain_id": domain_id,
        "domain": DOMAIN_MAP.get(domain_id, f"UNKNOWN ({domain_id})"),

        "category_id": category_id,
        "category": CATEGORY_MAP.get(category_id, f"UNKNOWN ({category_id})"),

        "sub_category_id": sub_category_id,
        "sub_category": SUBCATEGORY_MAP.get(sub_category_id, f"UNKNOWN ({sub_category_id})"),

        "severity_id": severity_level_id,
        "severity_level": SEVERITY_MAP.get(severity_level_id, f"UNKNOWN ({severity_level_id})"),

        "stage_id": stage_id,
        "stage": STAGE_MAP.get(stage_id, f"UNKNOWN ({stage_id})"),

        "harm_level_id": harm_level_id,
        "harm_level": HARM_MAP.get(harm_level_id, f"UNKNOWN ({harm_level_id})"),

        "feedback_type_id": feedback_type_id,
        "feedback_type": FEEDBACK_TYPE_MAP.get(feedback_type_id, f"UNKNOWN ({feedback_type_id})"),

        "improvement_opportunity_type_id": improvement_opportunity_id,
        "improvement_opportunity_type": IMPROVEMENT_OPPORTUNITY_MAP.get(improvement_opportunity_id, f"UNKNOWN ({improvement_opportunity_id})"),
        "timings": timings
    }

    # -------------------------------
    # Print Readable Output
    # -------------------------------
    if Print:
        print("\n================ CLASSIFICATION RESULT ================\n")
        print(f"DOMAIN      : {result['domain']} ({result['domain_id']})")
        print(f"CATEGORY    : {result['category']} ({result['category_id']})")
        print(f"SUBCATEGORY : {result['sub_category']} ({result['sub_category_id']})")
        print(f"SEVERITY    : {result['severity_level']} ({result['severity_id']})")
        print(f"STAGE       : {result['stage']} ({result['stage_id']})")
        print(f"HARM LEVEL  : {result['harm_level']} ({result['harm_level_id']})")
        print("\n================ TIMINGS (seconds) ==================\n")
        for key, t in timings.items():
            print(f"{key:25s}: {t:.4f}")
        print("\n========================================================\n")

    return result

def classify_feedback_encoded(text_1, text_2, text_3, Print = False):

    #Text Identification
    Patient_Text = text_1
    Hospital_Text = text_2 + " "+  text_3
    Combined_Text = Patient_Text + " " + Hospital_Text


    #Text Embedding
    Patient_Embedding_RAW = get_embedding(Patient_Text)
    Patient_Embedding = np.frombuffer(Patient_Embedding_RAW, dtype=np.float32)

    Hospital_Embedding_RAW = get_embedding(Hospital_Text)
    Hospital_Embedding = np.frombuffer(Hospital_Embedding_RAW, dtype=np.float32)

    Combined_Embedding_RAW = get_embedding(Combined_Text)
    Combined_Embedding = np.frombuffer(Combined_Embedding_RAW, dtype=np.float32)



    result_embedding = hierarchical_predict_embeddings(Patient_Embedding)
    domain_id = result_embedding["domain"]
    category_id = result_embedding["category"]
    sub_category_id = result_embedding["subcategory"]


    severity_level_id = predict_severity_from_embedding(Patient_Embedding)
    stage_id = classify_stage_Score_Based(Patient_Text, Print)

    harm_result = predict_harm_from_embedding(Combined_Embedding)
    harm_level_id = harm_result["harm_level"]

    feedback_type_id = predict_feedback_type_from_embedding(Patient_Embedding)
    improvement_opportunity_id = predict_improvement_from_embedding(Patient_Embedding)

    # ---------------------------------------------------------
    # READABLE DESCRIPTION
    # ---------------------------------------------------------
    return {
        "domain_id": domain_id,
        "category_id": category_id,
        "sub_category_id": sub_category_id,
        "severity_id": severity_level_id,
        "stage_id": stage_id,
        "harm_level_id": harm_level_id,
        "feedback_type_id": feedback_type_id,
        "improvement_opportunity_type_id": improvement_opportunity_id
    }



if __name__ == "__main__":


    Patient_Feedback = """
    اعترض المريض "" حول موضوع تواصل موظف الأمن الغير لائق,أنه بتاريخ 21-3-2025 احضر ابنه (10 سنوات) لزيارة جده المريض في الرابع غربي لكن موظف الأمن لم يسمح له وأنه انتظر بجانب الإستعلامات حضور اخته فأخبره موظف الأمن بالإنتظار بالباحة الخارجية وحدث نقاش فيما بينهم فاعتبر الموظف انه يريد إدخال ابنه بالقوة:)كل العالم قاعدين جوا لأن كان الطقس صقعة صار بدو يضهرني لبرا, وأن اخاه احضر اولاده وهم اصغر سنا وسمح لهم( وأن الموظف (ذكر انه نفس الشاب الذي حصل معه المشكل سابقا) لم يقم بتفتيش الشباب الداخلين وقام بتفتيشه هو..
    """
    Hospital_Feedback = ""
    Hospital_Feedback_2 = ""
    result = classify_feedback(Patient_Feedback, Hospital_Feedback, Hospital_Feedback_2, True)
    print(result)

