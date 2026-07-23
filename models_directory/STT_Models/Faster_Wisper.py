import os
import time
from faster_whisper import WhisperModel

MODEL_SIZE = "medium"
DEVICE = "cpu"

# WHISPER_MODEL_PATH (preferred, offline-safe): a local directory containing
# pre-downloaded CTranslate2 model files (see scripts/export_whisper_model.sh).
# When set, faster_whisper loads directly from disk and never contacts
# huggingface.co -- this is what ships in the offline release. If unset,
# MODEL_SIZE is passed as a name and faster_whisper downloads it from the
# Hugging Face Hub on first use -- only safe with internet access, e.g. local
# iteration on an online engineering machine. Not what ships offline.
WHISPER_MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH", "").strip()

# Toggle Arabic correction pass after transcription
USE_ARABIC_CORRECTION = False
CORRECTION_MODEL_NAME = "CAMeL-Lab/arabart-qalb15-gec-ged-13"

# Medical glossary: common STT mishearings -> correct form.
# Sorted by key length (desc) at runtime so longer phrases match first.
MEDICAL_GLOSSARY = {

    # Procedures
    "سكشن": "suction",
    "سوكشن": "suction",
    "شفط": "suction",
    "تيوب": "tube",
    "تيوبس": "tubes",
    "لتيوب": "tube",
    "كاتيتر": "catheter",
    "قسطرة": "catheter",
    "تركيب قسطرة": "catheter insertion",
    "ترايك": "tracheostomy",
    "تراك": "tracheostomy",
    "ترايكستومي": "tracheostomy",
    "تراكيستومي": "tracheostomy",
    "فولي": "foley catheter",
    "كانيولا": "cannula",
    "كانيولا وريدية": "IV cannula",

    # Devices
    "فنتيليتر": "ventilator",
    "فنتيلاتور": "ventilator",
    "جهاز تنفس": "ventilator",
    "مونيتور": "monitor",
    "مونيتر": "monitor",
    "بامب": "pump",
    "سرنج بامب": "syringe pump",
    "انفيوجن بامب": "infusion pump",
    "سيرنج بامب": "syringe pump",
    "أوكسجين": "oxygen",
    "اوكسجين": "oxygen",
    "أكسجين": "oxygen",
    "اكسجين": "oxygen",

    # Lab
    "سي بي سي": "CBC",
    "سي ار بي": "CRP",
    "اي اس ار": "ESR",
    "بي سي آر": "PCR",
    "بي سي ار": "PCR",
    "هيموغلوبين": "hemoglobin",
    "بلتلت": "platelets",
    "بلاتلت": "platelets",

    # Infection
    "سالمونيلا": "salmonella",
    "سلمونالة": "salmonella",
    "سلمونيلا": "salmonella",
    "كلوستريديوم": "clostridium",
    "كلبسيلة": "klebsiella",
    "سودوموناس": "pseudomonas",
    "ستاف": "staph",
    "ام آر اس ايه": "MRSA",
    "كورونا": "COVID-19",
    "كوفيد": "COVID-19",

    # Medications
    "باراسيتامول": "paracetamol",
    "بنادول": "Panadol",
    "بروفين": "Brufen",
    "فولتارين": "Voltaren",
    "أوغمنتين": "Augmentin",
    "اوغمنتين": "Augmentin",
    "روسبين": "Rocephin",
    "ميرونيم": "Meronem",
    "فانكو": "Vancomycin",

    # Clinical Terms
    "هبوط ضغط": "hypotension",
    "ارتفاع ضغط": "hypertension",
    "نقص أكسجة": "hypoxia",
    "نقص اكسجة": "hypoxia",
    "تسرع قلب": "tachycardia",
    "بطء قلب": "bradycardia",
    "حمى": "fever",
    "حرارة": "fever",
    "اختلاج": "seizure",
    "تشنج": "seizure",

    # Imaging
    "سي تي": "CT",
    "سيتي سكان": "CT scan",
    "سي تي سكان": "CT scan",
    "أم آر آي": "MRI",
    "ام آر آي": "MRI",
    "ام ار اي": "MRI",
    "إيكو": "echo",
    "ايكو": "echo",
    "ألتراساوند": "ultrasound",
    "التراساوند": "ultrasound",

    # Supplies
    "بطل سكشن": "suction bottle",
    "البطل سكشن": "suction bottle",
    "غلوفز": "gloves",
    "غلوفز": "gloves",
    "ماسك": "mask",
    "شاش": "gauze",
    "سرنج": "syringe",
    "سيرنج": "syringe",
    "إبرة": "needle",
    "ابرة": "needle",

    # Hospital Units
    "آي سي يو": "ICU",
    "اي سي يو": "ICU",
    "عناية فائقة": "ICU",
    "سي سي يو": "CCU",
    "طوارئ": "ER",
    "اسعاف": "ER",
    "غرفة عمليات": "OR",
    "او آر": "OR",

    # Common STT mistakes
    "التمريد": "التمريض",
    "المربض": "المريض",
    "مستشفانه": "مستشفانا",
    "مستشفانة": "مستشفانا",
    "يتعالش": "يتعالج",
    "التهبات": "التهابات",
    "البلغ": "البلغم",
    "استعمال": "استعمال",
    "استياء": "استياء",
    "احضار": "إحضار",
    # Cardiology
    "اي سي جي": "ECG",
    "تخطيط قلب": "ECG",
    "انجيو": "angiography",
    "قسطرة قلب": "cardiac catheterization",
    "احتشاء": "myocardial infarction",
    "جلطة قلبية": "myocardial infarction",
    "ذبحة": "angina",
    "رجفان": "atrial fibrillation",
    "فشل قلب": "heart failure",
    "قصور قلب": "heart failure",

    # Respiratory
    "نيبولايزر": "nebulizer",
    "نيبولايزر": "nebulizer",
    "استنشاق": "inhalation",
    "ربو": "asthma",
    "التهاب رئة": "pneumonia",
    "ذات الرئة": "pneumonia",
    "انسداد رئوي": "pulmonary embolism",
    "بلغم": "sputum",
    "تنبيب": "intubation",
    "انبوب تنفس": "endotracheal tube",

    # Neurology
    "جلطة دماغية": "stroke",
    "سكتة دماغية": "stroke",
    "نزف دماغي": "intracranial hemorrhage",
    "غيبوبة": "coma",
    "وعي": "consciousness",
    "اختلاجات": "seizures",
    "صرع": "epilepsy",
    "شلل": "paralysis",
    "خدر": "numbness",
    "تنميل": "paresthesia",

    # Gastroenterology
    "منظار": "endoscopy",
    "تنظير": "endoscopy",
    "قولون": "colon",
    "تنظير قولون": "colonoscopy",
    "معدة": "stomach",
    "قرحة": "ulcer",
    "نزيف هضمي": "GI bleeding",
    "إسهال": "diarrhea",
    "اسهال": "diarrhea",
    "إمساك": "constipation",

    # Renal
    "غسيل كلى": "dialysis",
    "دياليز": "dialysis",
    "قصور كلوي": "renal failure",
    "فشل كلوي": "renal failure",
    "حصى": "stone",
    "حصوة": "stone",
    "بول": "urine",
    "التهاب بول": "UTI",
    "زرع بول": "urine culture",
    "كرياتينين": "creatinine",

    # Surgical
    "خياطة": "suturing",
    "غرز": "sutures",
    "جرح": "wound",
    "ضماد": "dressing",
    "تغيير ضماد": "dressing change",
    "شق جراحي": "surgical incision",
    "عملية": "surgery",
    "جراحة": "surgery",
    "استئصال": "resection",
    "خراج": "abscess",

    # Orthopedics
    "كسر": "fracture",
    "جبصين": "cast",
    "جبيرة": "splint",
    "مفصل": "joint",
    "ورك": "hip",
    "ركبة": "knee",
    "كتف": "shoulder",
    "عمود فقري": "spine",
    "فقرات": "vertebrae",
    "عظم": "bone",

    # Obstetrics & Gynecology
    "حمل": "pregnancy",
    "ولادة": "delivery",
    "قيصرية": "cesarean section",
    "طلق": "labor",
    "جنين": "fetus",
    "مشيمة": "placenta",
    "إجهاض": "abortion",
    "اسقاط": "abortion",
    "رحم": "uterus",
    "مبيض": "ovary",

    # Pediatrics
    "خداج": "prematurity",
    "حضانة": "NICU",
    "رضيع": "infant",
    "طفل": "child",
    "لقاح": "vaccine",
    "تطعيم": "vaccination",
    "حرارة": "fever",
    "نمو": "growth",
    "وزن": "weight",
    "تغذية": "nutrition",

    # Oncology
    "سرطان": "cancer",
    "ورم": "tumor",
    "كيماوي": "chemotherapy",
    "علاج شعاعي": "radiotherapy",
    "خزعة": "biopsy",
    "انتشار": "metastasis",
    "حميد": "benign",
    "خبيث": "malignant",
    "عقدة": "nodule",
    "كتلة": "mass",

    # Laboratory
    "زرع": "culture",
    "مزرعة": "culture",
    "خضاب": "hemoglobin",
    "هيماتوكريت": "hematocrit",
    "كريات بيض": "WBC",
    "كريات حمر": "RBC",
    "سكر": "glucose",
    "إنزيمات": "enzymes",
    "يوريا": "urea",
    "شوارد": "electrolytes",

    # Nursing
    "ملاحظة تمريضية": "nursing note",
    "علامات حيوية": "vital signs",
    "نبض": "pulse",
    "ضغط": "blood pressure",
    "تشبع": "oxygen saturation",
    "حرارة جسم": "body temperature",
    "ميزان سوائل": "fluid balance",
    "إخراج": "output",
    "مدخول": "intake",
    "مراقبة": "monitoring",

    # Administration / Complaints
    "شكوى": "complaint",
    "مراجع": "patient attendant",
    "مرافق": "attendant",
    "استبيان": "survey",
    "رضى": "satisfaction",
    "تقييم": "evaluation",
    "خدمة مرضى": "patient services",
    "تحقيق": "investigation",
    "تصعيد": "escalation",
    "إغلاق": "closure",

    # Pharmacy
    "صيدلية": "pharmacy",
    "وصفة": "prescription",
    "جرعة": "dose",
    "حبوب": "tablets",
    "كبسولات": "capsules",
    "شراب": "syrup",
    "حقنة": "injection",
    "وريدي": "IV",
    "عضلي": "IM",
    "تحت الجلد": "subcutaneous",

    # Radiology
    "أشعة": "radiology",
    "صورة صدر": "chest X-ray",
    "أشعة مقطعية": "CT",
    "رنين": "MRI",
    "ألتراساوند": "ultrasound",
    "سونار": "ultrasound",
    "دوبلر": "doppler",
    "تصوير": "imaging",
    "صبغة": "contrast",
    "قراءة الأشعة": "radiology report",

    # Hospital workflow
    "دخول": "admission",
    "خروج": "discharge",
    "تحويل": "transfer",
    "موعد": "appointment",
    "عيادة": "clinic",
    "مراجعة": "follow-up",
    "استشارة": "consultation",
    "إحالة": "referral",
    "تنسيق": "coordination",
    "سرير": "bed",

    # Common English terms often spoken in Arabic
    "كونسلت": "consult",
    "ريفيو": "review",
    "فولواپ": "follow-up",
    "ابديت": "update",
    "ريبورت": "report",
    "فايل": "file",
    "سيستم": "system",
    "داتا": "data",
    "فورم": "form",
    "بروسس": "process"
}

# Lazy load models to support Windows multiprocessing
_model = None
_correction_tokenizer = None
_correction_model = None


def get_whisper_model():
    """Lazy load Whisper model on first use."""
    global _model
    if _model is None:
        if WHISPER_MODEL_PATH:
            if not os.path.isdir(WHISPER_MODEL_PATH):
                raise RuntimeError(
                    f"WHISPER_MODEL_PATH={WHISPER_MODEL_PATH!r} does not exist or is "
                    "not a directory. Run scripts/export_whisper_model.sh and mount "
                    "the resulting asset, or unset WHISPER_MODEL_PATH to fall back to "
                    "name-based download (requires internet)."
                )
            model_source = WHISPER_MODEL_PATH
        else:
            model_source = MODEL_SIZE
        print(f"[STT] Loading Faster-Whisper model: {model_source}")
        _model = WhisperModel(
            model_source,
            device=DEVICE,
            compute_type="int8",
            cpu_threads=8,
            num_workers=1
        )
    return _model


def get_correction_model():
    """Lazy load the Arabic correction model on first use."""
    global _correction_tokenizer, _correction_model
    if _correction_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        print(f"[Correction] Loading correction model: {CORRECTION_MODEL_NAME}")
        _correction_tokenizer = AutoTokenizer.from_pretrained(CORRECTION_MODEL_NAME)
        _correction_model = AutoModelForSeq2SeqLM.from_pretrained(CORRECTION_MODEL_NAME)
    return _correction_tokenizer, _correction_model


def correct_arabic_text(text: str) -> str:
    """Run raw Arabic text through the correction model and return corrected text."""
    tokenizer, model = get_correction_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    output_ids = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + 64)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def normalize_medical_terms(text: str) -> str:
    """Replace known STT mishearings with correct medical terms via MEDICAL_GLOSSARY.

    Longer phrases are matched before shorter ones to avoid partial replacements
    (e.g. "البطل سكشن" before "سكشن").
    """
    for key in sorted(MEDICAL_GLOSSARY, key=len, reverse=True):
        text = text.replace(key, MEDICAL_GLOSSARY[key])
    print("[Glossary] Medical glossary normalization applied")
    return text


def transcribe_arabic(audio_path: str) -> str:
    start = time.time()
    model = get_whisper_model()
    segments, info = model.transcribe(
        audio_path,
        language="ar",
        beam_size=1
    )
    raw_text = " ".join(segment.text for segment in segments)
    elapsed = time.time() - start
    print(f"[STT] Transcription time: {elapsed:.3f} sec")

    normalized_text = normalize_medical_terms(raw_text)

    print(f"[Correction] Arabic correction enabled: {USE_ARABIC_CORRECTION}")
    if not USE_ARABIC_CORRECTION:
        return normalized_text

    try:
        correction_start = time.time()
        corrected_text = correct_arabic_text(normalized_text)
        correction_elapsed = time.time() - correction_start
        print(f"[Correction] Correction time: {correction_elapsed:.3f} sec")
        return normalize_medical_terms(corrected_text)
    except Exception as e:
        print(f"[Correction] Correction failed, returning raw transcription: {e}")
        return normalized_text


if __name__ == "__main__":
    sample = "ذكر انه كان يريد اجراء سكشن للمريض فطلب من التمريد احضار تيوبس وكان لديه سلمونالة"
    result = normalize_medical_terms(sample)
    print(f"[Test] Input:  {sample}")
    print(f"[Test] Output: {result}")
    assert "suction" in result,   "Expected 'suction'"
    assert "tubes" in result,     "Expected 'tubes'"
    assert "salmonella" in result, "Expected 'salmonella'"
    assert "التمريض" in result,   "Expected 'التمريض'"
    print("[Test] All assertions passed.")
