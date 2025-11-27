# solution_gliner_arabic.py
from gliner import GLiNER
import re

# ---------------- Load Arabic GLiNER model ----------------
gliner_model = GLiNER.from_pretrained("NAMAA-Space/gliner_arabic-v2.1")

# ---------------- Prefixes for validation ----------------
PATIENT_PREFIXES = ["المريض", "المريضة", "الجريح", "الجريحة", "الطفل", "الطفلة"]
DOCTOR_PREFIXES = ["الدكتور", "الدكتورة"]

# ---------------- Common trailing words to remove ----------------
TRAILING_WORDS = ["بخير", "الآن", "تم", "و", "في", "على"]

# ---------------- Arabic normalization function ----------------
def normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[ؤئ]", "و", text)
    text = re.sub(r"ـ", "", text)  # remove tatweel
    return text

# ---------------- Name cleaning helper ----------------
def clean_name(name: str) -> str:
    tokens = name.strip().split()
    # Remove trailing common words
    tokens = [t for t in tokens if t not in TRAILING_WORDS]
    # Limit to 3 words max
    if len(tokens) > 3:
        tokens = tokens[:3]
    return " ".join(tokens)

# ---------------- Main extraction function ----------------
def extract_names_gliner_arabic(text: str) -> dict:
    """
    Extract Patient and Doctor names from Arabic clinical text.
    Uses GLiNER Arabic model + validation.
    Returns:
        dict: {"patients": [...], "doctors": [...]}
    """
    text = normalize_arabic(text)
    predictions = gliner_model.predict_entities(text, labels=["PATIENT", "DOCTOR"])

    patients = []
    doctors = []

    for p in predictions:
        if isinstance(p, dict):
            name = p.get("text", "").strip()
            entity = p.get("entity") or p.get("label") or "UNKNOWN"
        else:
            name = str(p).strip()
            entity = "UNKNOWN"

        name = clean_name(name)

        # Validate by entity or prefix
        if entity.upper() == "PATIENT" or any(name.startswith(prefix) for prefix in PATIENT_PREFIXES):
            patients.append(name)
        elif entity.upper() == "DOCTOR" or any(name.startswith(prefix) for prefix in DOCTOR_PREFIXES):
            doctors.append(name)

    return {"patients": patients, "doctors": doctors}

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    text = """
    اعترضت ابنة المريضة "نظيرة علي جعفر" أنه بتاريخ 29-12-2024, تم إحضار المريضة إلى قسم الطوارئ لدى د.عودة 
    ونقلها إلى الCCU. وبعدها إلى الغرفة 1200. الدكتورة فاطمة محمد عالجتها. الطفل يوسف سعيد بخير.
    """

    result = extract_names_gliner_arabic(text)
    print(result)
    # Expected output:
    # {'patients': ['نظيرة علي جعفر', 'يوسف سعيد'], 'doctors': ['د.عودة', 'فاطمة محمد']}
