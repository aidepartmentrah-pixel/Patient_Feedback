"""
label_mapping_helper.py
Loads temp_to_label mappings (XGBoost internal class index -> real database
label) from small JSON sidecar files shipped alongside each model file.

Each of the 10 category/subcategory models needs to translate the model's
raw output (0, 1, 2, ...) back into a real domain/category/subcategory ID.
That mapping is a handful of integers, fully determined at training time
(unique_sorted = sorted(np.unique(y_train)); temp_to_label = {i: v for i, v
in enumerate(unique_sorted)}) -- it never changes after training and never
needs the original training data present at inference time.

This module previously reconstructed that mapping by querying
table_feedback_train live, out of a 116MB SQLite file containing real
patient complaint text, on every server startup. That meant the deployable
artifact for these 10 tiny integer maps was "the entire historical training
dataset" -- real patient data with no reason to travel with a production
deployment. Each model's label map is now generated once at training time
(see scripts/generate_label_maps.py) and saved as a JSON sidecar next to the
model file; this module just reads that file. Zero patient data involved.
"""

import json
import os


def load_temp_to_label(label_map_path: str) -> dict:
    """
    Load a temp_to_label mapping from its JSON sidecar file.

    Args:
        label_map_path: path to the '<model_name>_label_map.json' file,
            written once at training time next to the model's .pkl/.json
            files (same vocab_models/ directory).

    Returns:
        dict mapping XGB internal index (int) -> real label (int)

    Raises:
        RuntimeError: if the sidecar file doesn't exist -- this model's
            label mapping was never generated (or the model itself needs
            retraining -- see ML_CLASSIFICATION_ISSUE_FOR_DEV_TEAM.md for
            Category 1 / Category 2 subcategory specifically).
    """
    if not os.path.isfile(label_map_path):
        raise RuntimeError(
            f"Label map not found: {label_map_path}. Run "
            f"scripts/generate_label_maps.py after training this model to "
            f"produce it, or this model genuinely isn't trained/available yet."
        )
    with open(label_map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def validate_model_mapping(model, temp_to_label: dict, file_name: str, model_path: str) -> None:
    """
    Validate that the XGB model's class count matches the mapping size.

    Args:
        model: The loaded XGBClassifier
        temp_to_label: The derived mapping dict
        file_name: Name of the predictor file (for error messages)
        model_path: Path to the model file (for error messages)

    Raises:
        RuntimeError: If there's a mismatch between model classes and mapping size
    """
    expected = model.n_classes_
    actual = len(temp_to_label)

    if expected != actual:
        raise RuntimeError(
            f"Label mapping mismatch in {file_name}: "
            f"model classes={expected}, mapping size={actual}, mapping={temp_to_label}. "
            f"Model path: {model_path}"
        )


def log_predictor_init(file_name: str, model_path: str, n_classes: int, temp_to_label: dict) -> None:
    """
    Log predictor initialization details for debugging/verification.
    """
    print(f"[PREDICTOR_INIT] {file_name}")
    print(f"  Model path: {model_path}")
    print(f"  Model n_classes: {n_classes}")
    print(f"  Derived temp_to_label: {temp_to_label}")
