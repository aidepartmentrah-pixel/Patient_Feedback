from model_training_2.Hierarchical_Classification_Model.hierarchical_predictor import hierarchical_predict_embeddings
from model_training_2.Severity_level.predict_severity import predict_severity_from_embedding
from model_training_2.Stage.model_package import classify_stage_Score_Based
from model_training_2.Harm_level.predict_harm import predict_harm_from_embedding
from model_training_2.Stage.modular_functions import get_embedding
import numpy as np
import json
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

SEVERITY_MAP = {
    1: "HIGH",
    2: "LOW",
    3: "MEDIUM",
    6: "Moderate",
}

STAGE_MAP = {
    1: "Examination & Diagnosis",
    2: "Admissions",
    4: "Care on the Ward",
    6: "Discharge/Transfer",
    8: "Operation/Procedure",
    9: "Unspecified",
}

HARM_MAP = {
    1: "Severe Harm",
    2: "Death",
    3: "High Severe",
    4: "Minor Harm",
    5: "Moderate Harm",
    6: "No Harm",
}



def classify_feedback(text_1, text_2, text_3, Print = False):

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
    stage_id = classify_stage_Score_Based(Patient_Text)
    harm_result = predict_harm_from_embedding(Combined_Embedding)
    harm_level_id = harm_result["harm_level"]

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
    }

    # ---------------------------------------------------------
    # READABLE DESCRIPTION
    # ---------------------------------------------------------
    if Print:
        print(f"The Feedback{Patient_Feedback} ")

        print("\n================ CLASSIFICATION RESULT ================\n")
        print(f"DOMAIN      : {result['domain']} ({result['domain_id']})")
        print(f"CATEGORY    : {result['category']} ({result['category_id']})")
        print(f"SUBCATEGORY : {result['sub_category']} ({result['sub_category_id']})")
        print(f"SEVERITY    : {result['severity']} ({result['severity_id']})")
        print(f"STAGE       : {result['stage']} ({result['stage_id']})")
        print(f"HARM LEVEL  : {result['harm_level']} ({result['harm_level_id']})")
        print("\n========================================================\n")
    return result




if __name__ == "__main__":


    Patient_Feedback = """
    اعترض المريض "" حول موضوع تواصل موظف الأمن الغير لائق,أنه بتاريخ 21-3-2025 احضر ابنه (10 سنوات) لزيارة جده المريض في الرابع غربي لكن موظف الأمن لم يسمح له وأنه انتظر بجانب الإستعلامات حضور اخته فأخبره موظف الأمن بالإنتظار بالباحة الخارجية وحدث نقاش فيما بينهم فاعتبر الموظف انه يريد إدخال ابنه بالقوة:)كل العالم قاعدين جوا لأن كان الطقس صقعة صار بدو يضهرني لبرا, وأن اخاه احضر اولاده وهم اصغر سنا وسمح لهم( وأن الموظف (ذكر انه نفس الشاب الذي حصل معه المشكل سابقا) لم يقم بتفتيش الشباب الداخلين وقام بتفتيشه هو..
    """
    Hospital_Feedback = ""
    Hospital_Feedback_2 = ""
    start = time.time()  # Start timer
    result = classify_feedback(Patient_Feedback, Hospital_Feedback, Hospital_Feedback_2)
    end = time.time()
    print(result)
    print(f"\n⏱️ Time taken: {end - start:.4f} seconds")

