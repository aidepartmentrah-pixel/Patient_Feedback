from backend.core.id_mapping import remap_classification_output

def _get_classifier():
    """Lazy import classifier to avoid heavy startup downloads on reload."""
    from models_directory.Classification_Models.package_models import classify_feedback
    return classify_feedback
from models_directory.NER_Model.solution_gliner import extract_names_gliner_arabic

# -----------------------------------------
# MAIN ENTRY POINT FOR INSERT RECORD PAGE
# -----------------------------------------

def run_intelligence_pipeline(
    *,
    text: str | None = None,
    run_ner: bool = True,
    run_classification: bool = True,
) -> dict:
    """
    Runs STT, NER, and Classification for Insert Record page.
    """

    result = {
        "text": text,
        "ner": {},
        "classification": {},
    }

    # -----------------------------
    # 1. SPEECH TO TEXT (OPTIONAL)
    # -----------------------------


    if not text:
        return {
            "success": True,
            "result": result,
        }

    # -----------------------------
    # 2. NER
    # -----------------------------
    if run_ner:
        ner_result = extract_names_gliner_arabic(text)
        result["ner"] = ner_result

    # -----------------------------
    # 3. CLASSIFICATION (ALL 8)
    # -----------------------------
    if run_classification:
        classifier = _get_classifier()
        classification_result = classifier(
            text_1=text,
            text_2="",
            text_3="",
            Print=False,
        )
        # Apply mapping so the Insert Page sees new IDs
        classification_result = remap_classification_output(classification_result)
        result["classification"] = classification_result

    return {
        "success": True,
        "result": result,
    }