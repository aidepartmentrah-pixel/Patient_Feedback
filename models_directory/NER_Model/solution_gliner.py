# -*- coding: utf-8 -*-

from gliner import GLiNER
import re


# ============================================================
# Load Arabic GLiNER model
# ============================================================
gliner_model = GLiNER.from_pretrained("NAMAA-Space/gliner_arabic-v2.1")


# ============================================================
# Text Normalization
# ============================================================
def normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[ؤئ]", "و", text)
    text = re.sub(r"ـ", "", text)
    return text


# ============================================================
# Role Dictionaries
# ============================================================

DOCTOR_HINTS = [
    "د.", "دكتور", "الدكتور", "دكتورة",
    "طبيب", "طبيبة",
    "طبيب الاشعة", "طبيب التخدير", "طبيب الجراحة",
    "استشاري", "استشارية",
    "أخصائي", "أخصائية"
]

EMPLOYEE_HINTS = [
    "ممرض", "ممرضة", "التمريض",
    "فني", "فني اشعة", "فني الأشعة",
    "فني مختبر", "فني المختبر",
    "فني تخدير", "فني التأهيل",
    "فني الانعاش", "فنيي الانعاش",
    "فني بنك الدم", "فنيي بنك الدم",
    "عامل", "عاملات", "عاملات النظافة",
    "النظافة", "المغسل",
    "موظف", "موظفة",
    "الاستعلامات", "السنترال",
    "المحاسبة", "المشتريات",
    "مسؤول القسم",
    "المعلوماتية", "تقنية المعلومات",
    "فريق العمليات", "فريق التلقيح",
    "طاقم العمليات", "طاقم التلقيح"
    "موظف الاستعلامات", "موظفة الاستعلامات"

]

PATIENT_HINTS = [
    "المريض", "المريضة",
    "الطفل", "الطفلة",
    "المسن", "المسنة",
    "المصاب", "المصابة",
    "المراجع", "المراجعة"
    "الجربح ", "الجرحى" , "المصابين", "المصابات"

]

TRAILING_NOISE = {
    "بخير", "جيد", "الان", "الآن", "مستقر", "تمام"
}

ROLE_WORDS = set(DOCTOR_HINTS + EMPLOYEE_HINTS + PATIENT_HINTS)


# ============================================================
# Helpers
# ============================================================

def clean_name(name: str) -> str:
    name = name.strip()

    # Remove role prefixes
    for p in DOCTOR_HINTS:
        if name.startswith(p):
            name = name.replace(p, "", 1).strip()

    # Remove stray feminine letter
    if name.startswith("ة "):
        name = name[2:]

    words = name.split()
    words = [w for w in words if w not in TRAILING_NOISE]

    if len(words) > 3:
        words = words[:3]

    return " ".join(words)


def is_valid_arabic_name(name: str) -> bool:
    if len(name.split()) < 2:
        return False

    for role in ROLE_WORDS:
        if role in name:
            return False

    if not re.fullmatch(r"[ء-ي\s]+", name):
        return False

    return True


# ============================================================
# Role Detection (Priority-based)
# ============================================================

def detect_role(name: str, text: str) -> str | None:
    window = 25
    idx = text.find(name)

    if idx == -1:
        return None

    context = text[max(0, idx - window): idx + window]

    # Priority matters
    # Highest confidence first
    if any(d in context for d in DOCTOR_HINTS):
        return "doctor"

    if any(e in context for e in EMPLOYEE_HINTS):
        return "employee"

    if any(p in context for p in PATIENT_HINTS):
        return "patient"

    return None


# ============================================================
# Main Extractor
# ============================================================

def extract_names_gliner_arabic(text: str) -> dict:
    text = normalize_arabic(text)

    predictions = gliner_model.predict_entities(
        text,
        labels=["PERSON"]
    )

    patients = set()
    doctors = set()
    employees = set()

    for ent in predictions:
        name = clean_name(ent.get("text", "").strip())

        if not is_valid_arabic_name(name):
            continue

        role = detect_role(name, text)

        if role == "doctor":
            doctors.add(name)
        elif role == "employee":
            employees.add(name)
        elif role == "patient":
            patients.add(name)

    return {
        "patients": sorted(patients),
        "doctors": sorted(doctors),
        "employees": sorted(employees)
    }


# ============================================================
# Example Run
# ============================================================

if __name__ == "__main__":
    text = """
    اعترضت ابنة المريضة نظيرة علي جعفر.
    تم إحضارها إلى الطوارئ لدى د.عودة.
    الدكتورة فاطمة محمد أشرفت على الحالة.
    حضر فني الأشعة محمد حسن.
    كما حضر موظف الاستعلامات أحمد علي.
    الطفل يوسف سعيد بخير.
    """

    print(extract_names_gliner_arabic(text))
