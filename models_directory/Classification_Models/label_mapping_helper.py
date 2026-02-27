"""
label_mapping_helper.py
Dynamically builds temp_to_label mappings from the SQLite training database.
This ensures predictor mappings match exactly what was used during training.
"""

import sqlite3
from project_paths import get_db_path


def build_temp_to_label_for_domain(domain_id: int) -> dict:
    """
    Build temp_to_label mapping for category prediction within a domain.
    
    Queries DISTINCT category values from table_feedback_train WHERE domain = domain_id,
    sorts them, and returns {0: first_label, 1: second_label, ...}.
    
    This replicates the training encoding:
        unique_sorted = sorted(np.unique(y_train))
        temp_to_label = {i: v for i, v in enumerate(unique_sorted)}
    
    Args:
        domain_id: The domain ID (1, 2, or 3)
    
    Returns:
        dict mapping XGB internal index -> real category label
    
    Raises:
        RuntimeError: If no categories found for the domain
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT DISTINCT category FROM table_feedback_train WHERE domain = ? ORDER BY category",
        (domain_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise RuntimeError(
            f"No categories found in table_feedback_train for domain={domain_id}. "
            f"DB path: {db_path}"
        )
    
    unique_sorted = sorted([row[0] for row in rows])
    temp_to_label = {i: label for i, label in enumerate(unique_sorted)}
    
    return temp_to_label


def build_temp_to_label_for_category(category_id: int) -> dict:
    """
    Build temp_to_label mapping for subcategory prediction within a category.
    
    Queries DISTINCT sub_category values from table_feedback_train WHERE category = category_id,
    sorts them, and returns {0: first_label, 1: second_label, ...}.
    
    This replicates the training encoding:
        unique_sorted = sorted(np.unique(y_train))
        temp_to_label = {i: v for i, v in enumerate(unique_sorted)}
    
    Args:
        category_id: The category ID (1-7)
    
    Returns:
        dict mapping XGB internal index -> real subcategory label
    
    Raises:
        RuntimeError: If no subcategories found for the category
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT DISTINCT sub_category FROM table_feedback_train WHERE category = ? ORDER BY sub_category",
        (category_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise RuntimeError(
            f"No subcategories found in table_feedback_train for category={category_id}. "
            f"DB path: {db_path}"
        )
    
    unique_sorted = sorted([row[0] for row in rows])
    temp_to_label = {i: label for i, label in enumerate(unique_sorted)}
    
    return temp_to_label


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
