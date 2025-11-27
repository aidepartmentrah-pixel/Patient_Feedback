# solution_regex_arabic.py
import re


def extract_names_regex_arabic(text):
    """
    Extract Patient and Doctor names from Arabic text using regex.
    Handles common prefixes for patients and doctors:
    - Patients: "المريض", "المريضة", "الجريح", "الجريحة", "الطفل", "الطفلة"
    - Doctors: "الدكتور", "الدكتورة"

    Returns:
        dict: {"patients": [...], "doctors": [...]}
    """
    # Patterns for Doctors (الدكتور / الدكتورة)
    # Capture up to 3 Arabic words (to avoid very long nonsensical strings)
    doctor_pattern = r"(?:الدكتور|الدكتورة)\s+([ء-ي]{2,15}(?:\s[ء-ي]{2,15}){0,2})"

    # Patterns for Patients (المريض / المريضة / الجريح / الجريحة / الطفل / الطفلة)
    patient_pattern = r"(?:المريض|المريضة|الجريح|الجريحة|الطفل|الطفلة)\s+([ء-ي]{2,15}(?:\s[ء-ي]{2,15}){0,2})"

    doctors = re.findall(doctor_pattern, text)
    patients = re.findall(patient_pattern, text)

    return {"patients": patients, "doctors": doctors}


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    text = """
      اعترضت ابنة المريضة "نظيرة علي جعفر" أنه بتاريخ 29-12-2024, تم إحضار المريضة إلى قسم الطوارئ لدى د.عودة 
      ونقلها إلى الCCU. وبعدها إلى الغرفة 1200. الدكتورة فاطمة محمد عالجتها. الطفل يوسف سعيد بخير.
      """
    result = extract_names_regex_arabic(text)
    print(result)
