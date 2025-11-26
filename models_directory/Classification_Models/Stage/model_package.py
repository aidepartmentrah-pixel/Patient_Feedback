from models_directory.Classification_Models.Stage.modular_functions import get_embedding
from models_directory.Classification_Models.Stage.Training_Internal_Metrics.internal_metrics import split_max_score
import numpy as np
import joblib
import json
import os




def classify_stage_Numerical(sentence: str) -> str:
    # Path to the trained model folder
    MODEL_DIR = r"/models_directory/Stage/Training_Internal_Metrics/vocab_models/train_ML_Metric_Mapper_Numeric"

    # Model and label map filenames
    MODEL_PATH = os.path.join(MODEL_DIR, "mapper_model.pkl")
    LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

    """
    Classify a sentence into one of the Stage categories using the trained
    numerical mapper model. Returns the stage name as a string.
    """

    # Edge case: empty
    if sentence is None or str(sentence).strip() == "":
        return None

    # Load trained classifier
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    clf = joblib.load(MODEL_PATH)

    # Load label mapping
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found at: {LABEL_MAP_PATH}")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    # --- Convert input sentence to embedding ---
    vec = get_embedding(sentence)     # Must return a 1D numpy vector
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec, dtype=np.float32)
    # Predict
    pred = clf.predict(vec.reshape(1, -1))[0]
    # Map numeric → label text
    return label_map.get(str(int(pred)), None)

def classify_stage_Score_Based(sentence: str, Troubleshoot = False):
    STAGE_GROUPS = {
        "admission": [
            "administration_delay",
            "arrival",
            "bed_room",
            "billing_issues"
        ],
        "care_on_ward": [
            "cleanliness",
            "food",
            "location",
            "staff_security_behavior"
        ],
        "discharge": [
            "billing_issues",
            "disagreement_with_discharge"
        ],
        "examination_diagnosis": [
            "clinical_delay",
            "conflicting_or_wrong_diagnosis"
        ],
        "operation": [
            "communication_scheduling",
            "medical_clinical_errors"
        ]
    }
    STAGE_ENCODINGS = {
        "examination_diagnosis": 1,
        "admission": 2,
        "care_on_ward": 4,
        "discharge": 6,
        "operation": 8,
        "unspecified": 9
    }

    # Run the metric scoring using your function
    sorted_metrics, per_segment_scores = split_max_score(paragraph = sentence,Troubleshoot= Troubleshoot)
    if len(sorted_metrics) == 0:
        return {
            "stage_scores": {},
            "chosen_stage": None,
            "sorted_individual_metrics": [],
            "segment_scores": per_segment_scores
        }

    # Convert sorted list to dict {metric: score}
    metric_dict = {m: s for m, s in sorted_metrics}
    stage_scores = {}

    # Compute score for each stage
    for stage, metrics in STAGE_GROUPS.items():
        values = [metric_dict.get(m, 0.0) for m in metrics]
        if len(values) == 0:
            stage_scores[stage] = {
                "max": 0.0,
                "avg": 0.0,
                "metrics_used": {}
            }
            continue

        stage_scores[stage] = {
            "max": max(values),
            "avg": sum(values) / len(values),
            "metrics_used": {m: metric_dict.get(m, 0.0) for m in metrics}
        }

    # ---- Decision Step ----
    # 1️⃣ Find highest max across stages
    best_max = max(stage_scores[s]["max"] for s in stage_scores)

    # Filter stages that share this max
    candidates = [s for s in stage_scores if stage_scores[s]["max"] == best_max]

    if len(candidates) == 1:
        chosen_stage = candidates[0]
    else:
        # 2️⃣ Use average to break ties
        chosen_stage = max(
            candidates,
            key=lambda s: stage_scores[s]["avg"]
        )

    variable =  {
        "stage_scores": stage_scores,
        "chosen_stage": chosen_stage,
        "sorted_individual_metrics": sorted_metrics[:3],
        "segment_scores": per_segment_scores
    }

    if Troubleshoot:
        # 🔷 PRINT THE FINAL CLASSIFICATION
        print(f'==============================================')
        print(f"\n🧩 Sentence classified as: {chosen_stage}")
        print("Top contributing metrics:")
        for m, score in sorted_metrics[:3]:
            print(f" - {m}: {score:.3f}")
        print(variable)


    stage_encoding = STAGE_ENCODINGS.get(chosen_stage, 9)

    return stage_encoding

text = """
اعترض مرافق المريضة ""أنه بتاريخ 20-4-2025 .حضروا إلى الطوارئ وإعتراضهم حول عدم نظافة الحمامات بالطوارئ(" الحمامات بالطوارئ ابدا مش نظيفة... بقينا يوم ونص بالطوارئ الحمامات ما بينفات عليهن ابدا...")
"""

if __name__ == "__main__":
    number = classify_stage_Score_Based(text, True)
    print(number)
    sentences_examination = [
        "اعترض إبن المريضة أنه بتاريخ 26-12-2024 حضروا إلى قسم الطوارئ (روايا، إلتهابات، مشاكل بالقلب) ومن ثم إجراء دخول إلى الcard2 غرفة 1203 لدى د. مالك موسى وتم وضع مكنة لسحب السوائل، وبتاريخ 3-1-2025 طلب د. مالك نزع المكنة لكن لم يحضر أحد لنزعها من مبارح الظهر ولهلق اليوم الساعة 10.00",
        "اعترض مرافق المريضة أنه بتاريخ 3-1-2025 تم إجراء دخول للمعالجة لدى د. علي سريج وبقيت لمدة 8 أيام وتم طلب استشارة د. قنبر لإجراء تمييل لشرايين الساق وأخبرهم أنه يجب معاودة إدخال المريضة بعد 10 أيام ووصف إبر للسيلان وأن يتم أخذها قبل الدخول ولم يتم الاتصال بهم من قبل مكتب الدخول فتواصلوا معهم فأخبروهم بعدم وجود حجز في العمليات وعند المتابعة مع الطبيب تقرر إدخال المريضة بتاريخ 27-01-2025 وعند إجراءات ما قبل الدخول حضر طبيب البنج وأخبرهم بوجود خطر على المريضة فانتظروا حضور د. قنبر الذي أكد الموضوع فرفضوا إجراء العملية معترضين أنه كان من المفترض معرفة د. قنبر بذلك سابقا ووضع الأهل بالصورة، وأنه تم وصف إبر بقيمة 50$ بدون حاجة، وأنه عند المغادرة لم تترك وصفة العلاج وتم الطلب منهم الحضور لعيادته",
        "اعترض المريض أنه بتاريخ 13-1-2025 حضر إلى الطوارئ (نزيف في الدماغ) ومن ثم دخول لإجراء عملية لدى د. دانيال عباس وبقي يومين في العناية ثم نقل إلى غرفة 432 وذكر أنه شعر بألم شديد في قدمه فتم طلب دكتور الأعصاب وإجراء فحوصات (MRI وتخطيط) لكن الألم استمر وتم وصف دواء وأخبره الطبيب بإجراء علاج فيزيائي لكنه لم يقتنع وتقدم بالشكوى لكتابة أمر المغادرة وهو لا يزال موجوعا",
        "اعترض مرافق المريضة أنه بتاريخ 22-2-2025 حضروا إلى قسم الطوارئ (الزلال مرتفع وإلتهابات) وتم طلب إجراء MRI للكبد وإنزال المريضة للصورة لكنها لم تتحمل فأعيدت وأعطيت منوم ثم أنزلت مجدداً وتم طلب طبيب بنج لإجراء الصورة وتم بعدها نقلها إلى الطابق بتاريخ 26-2-2025 واعتراضهم حول الانتظار الطويل لإجراء الصورة وأن الكلفة ترتفع في الطوارئ",
        "اعترض المريض أنه بتاريخ 20-2-2025 حضر إلى الطوارئ (التهابات) وأدخل إلى الرابع غربي غرفة 407 لدى الدكتورة مروة وطلب حضور طبيب مسالك بسبب ألم شديد فحضر د. نبيل الحركة ووضع ميل بتاريخ 22-2-2025 واعتراض المريض حول عدم حضور د. الحركة مرة أخرى لمعاينته وهو موجوع",
        "اعترض ذوو المريضة أنه بتاريخ 20-2-2025 أحضروا المريضة إلى الطوارئ (عوارض جلطة) ومن ثم دخولها وإجراء العملية لدى د. صعب ثم نقلت إلى CCU ومتابعتها من قبل د. يونس واعتراض الأهل على قرار الطبيب حيث قال إنها جلطة بينما رأي د. مروة ود. عودة كان مخالفا وأنها لا تعاني جلطة وطلب الأهل نقل الملف لطبيب آخر ورفضوا المتابعة مع د. يونس",
        "اعترض مرافق المريضة أنه بتاريخ 14-3-2025 أحضر ابنته للطوارئ (حادثة وضربة على العين) وتفاجأ بعدم إمكانية استشارة طبيب عيون في الطوارئ وأن عليه أخذ موعد في العيادات الخارجية وأن ذلك غير مقبول برأيه",
        "اعترضت ابنة المريض أنه بتاريخ 2-3-2025 (مشاكل في المصران) وأجري له صورة في إحدى المستشفيات في الجنوب ولم يتواجد أحد لقراءتها فأعطوهم CD وطلبوا المتابعة في مستشفى آخر فتم إحضاره بشكل عاجل للطوارئ وكان بصق دماً وتمت معاينته وذكرت الملاحظات أن CD لم يفتح فقرر الطبيب إعادة الصورة ثم تبين لاحقاً أنه يمكن فتحه ولم تلغ الصورة ثم صدرت نتيجتها فجراً وأخبرهم طبيب الطوارئ أنه لا يمكن فعل شيء حتى حضور الطبيب صباحاً ووصل د. حطيط الساعة 9 وتم تحويله لإجراء عملية ثم علموا بعد العملية أن التأخر في إنزاله أدى لتدهور حالته",
        "اعترضت والدة المريضة (مشاكل بالكبد 19 سنة) وأنه بتاريخ 8-3-2025 تم إدخالها إلى CCU وساءت حالتها وفقدت وعيها وتم وضعها في العناية القلبية لعدم توفر سرير في ICU وبرأي الأهل أن الطبيب لا يحضر للاطمئنان وطلبوا نقلها إلى ICU ليتمكن الطبيب من متابعتها",
        "اعترضت المريضة أنه بتاريخ 26-12-2025 أدخلت إلى المستشفى غرفة 335 لدى د. السيد وبعد إجراء العملية ونقلها للطابق كانت تعاني من مغص قوي وإسهال فحضر الطبيب المناوب د. قعيق وأخبرها أن الأمر طبيعي ووصف Buscopan لكن الألم استمر ولم يفعل شيئاً حتى حضر د. السيد وطلب صورة وتبين إصابتها بجرثومة",
        "اعترض ذوو المريضة أنه بتاريخ 3-1-2025 حضروا للطوارئ (ضيق تنفس ومشاكل على الرئتين) وتمت معاينتها ثم طلبت صورة لكن رأي الطبيب أنه لا داعي بينما أصر الأهل وتم التواصل بالطبيب الذي يتابع الحالة وقرر إعطاء علاج للتشنجات وتم تعليق مصل وعند السؤال عن الدواء لم تحصل على توضيح ثم بعد المتابعة تبين لاحقاً أنها تعاني من التهاب رئوي",
        "حضرت ابنة المريض وذكرت أنه بتاريخ 4-1-2025 أدخل المريض لدى د. بلال ضامن (رئتين) وبعد ثلاثة أيام نقل إلى غرفة 411 ومن ضمن الإجراءات طلب إجراء صورة للبنكرياس ثم استشارة دكتور جهاز هضمي وذكرت أن الطبيبة أخبرته أمامه أنه قد يكون سرطان وأن عليه صورة إضافية غير متوفرة إلا في مستشفى آخر مما أدى لقلق المريض الشديد واستعان بأولاده",
        "اعترض ذوو الجريح أنه بتاريخ 9-11-2024 أصيب في قدمه ثم بتاريخ 14-11 حضر الطوارئ وبقي 13 يوماً ثم نقل للطابق وتدهورت حالته بعد 10 أيام وتم نقله للعناية واعتراضهم حول إعطائه Morphine 7.5mg عدة مرات وأنه توقف عن الطعام وتم استشارة أطباء وتم إجراء ناظور ولم يظهر مشكلة عضوية ثم تم وصف له إبرة سيلان وبعدها حصل نزيف ولم يحضر الطبيب لمعاينته حتى اليوم التالي وتدهورت حالته ثم تم نقل الجريح للعناية ثم بدأت لديه نوبات وذكروا أنه وصل لشبه موت دماغي",
        "اعترض المريض أنه حضر بتاريخ 18-2-2025 لإجراء تخطيط لدى د. وزنة ثم استلم النتيجة وأخبره طبيبه أن التخطيط أجري لمفصل اليد بدلاً من الكتف فتم إعادة كتابته وإعطاؤه موعد جديد وبعد محاولة إعادة الفحص طُلب منه دفع تكلفة إضافية وبرأيه لم يكن يجب أن يتحمل كلفة خطأ الطبيب",
        "اعترضت المريضة أنها أجرت MRI للكبد وتواصلت مع قسم الأشعة عبر واتساب بخصوص جهوزية التقرير وعند الحضور لاستلامه تفاجأت أنه غير موجود وأن عليها الانتظار 20 دقيقة وقالت الموظفة ربما استلمته من قبل مما سبب استياءها لأنها لديها موعد طبيب ولا يمكنها الانتظار",
        "اشتكى المريض أنه حضر لإجراء فحوصات بتاريخ 12-01-2025 وأعطى رقمه الجديد ولم يتم إرسالها ثم حضر لاحقاً للاستعلام وأخبرته الموظفة أن النتيجة أرسلت على الرقم القديم رغم إعطائه الرقم الجديد مجدداً ولم يتم الإرسال أيضاً",
        "اعترضت المريضة أنها بتاريخ 7-2-2025 حضرت لقسم الأشعة لإجراء Echo طارئ وانتظرت من 11:30 إلى 1:15 ثم دخلت وجهزت نفسها وانتظرت نصف ساعة دون حضور الطبيبة فرفضت إجراء الصورة وغادرت وهي غير راضية عن الانتظار بهذا الشكل",
        "اعترض ابن المريضة أنه بتاريخ 26-2-2025 دخلوا لدى د. مالك (قوى الأمن) وتم إبلاغه بإجراء فحصين أحدهما غير متوفر ويجب إرساله لمختبر خارجي بكلفة عالية فذهب للجهة الضامنة وأخذ موافقة دون تكلفة ثم عند العودة أخبروه أن الفحصين أرسلوا بالفعل للمختبر وأنه لا يمكن تقسيم الخزعة مما سبب استياءه بعد أن بذل جهداً كبيراً للحصول على الموافقة",
        "اعترض المريض أنه قبل الحرب بتاريخ 12-9-2024 حصل معه حادث سير ودخل لدى د. يونس ثم غادر على أساس متابعة لاحقة لكن الحرب منعته ثم بعد عودته أجرى دخولاً بتاريخ 20-1-2025 للعملية وفوجئ بإبلاغه أن الفحوصات غير طبيعية ويتوجب متابعة طبيب أمراض دم ولم يقتنع وطلب إعادة الفحوصات فقالوا إن الطبيب أقفل الملف ثم تبين أن النتائج طبيعية وشعر أنه خسر الثقة واضطر للمتابعة خارج المستشفى وانتظر أسبوعاً إضافياً رغم أن الموعد كان محدداً سابقاً"
    ]

    # for sentence in sentences_care:
    #     print(f"The sentence is {sentence}")
    #     print(split_max_score(sentence))
    # sorted_metrics, per_segment_scores = split_max_score(sentences_examination[0])
    # print(f"Sorted metrics {sorted_metrics}")
    # print(f"Per-segment scores {per_segment_scores}")

