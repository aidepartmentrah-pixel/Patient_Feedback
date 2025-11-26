from model_training_2.Stage.model_package import classify_stage_Score_Based

if __name__ == "__main__":
    Patient_Feedback = """
    اعترض المريض "" حول موضوع تواصل موظف الأمن الغير لائق,أنه بتاريخ 21-3-2025 احضر ابنه (10 سنوات) لزيارة جده المريض في الرابع غربي لكن موظف الأمن لم يسمح له وأنه انتظر بجانب الإستعلامات حضور اخته فأخبره موظف الأمن بالإنتظار بالباحة الخارجية وحدث نقاش فيما بينهم فاعتبر الموظف انه يريد إدخال ابنه بالقوة:)كل العالم قاعدين جوا لأن كان الطقس صقعة صار بدو يضهرني لبرا, وأن اخاه احضر اولاده وهم اصغر سنا وسمح لهم( وأن الموظف (ذكر انه نفس الشاب الذي حصل معه المشكل سابقا) لم يقم بتفتيش الشباب الداخلين وقام بتفتيشه هو..
    """
    Hospital_Feedback = ""
    Hospital_Feedback_2 = ""
    # classify_feedback(Patient_Feedback, Hospital_Feedback, Hospital_Feedback_2)
    classify_stage_Score_Based(Patient_Feedback, True)