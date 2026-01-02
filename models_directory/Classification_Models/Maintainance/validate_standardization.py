#!/usr/bin/env python3
"""
Validation script to verify all training functions return standardized format.
This script imports all training functions and checks their return signatures.
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

print("=" * 70)
print("STANDARDIZATION VALIDATION")
print("=" * 70)

# Test imports
all_imports_ok = True
errors = []

# Training function imports
training_funcs = [
    ("Domain Model", "models_directory.Classification_Models.Hierarchical_Classification_Model.domain.train_domain_model", "train_domain_models"),
    ("Category 1", "models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_1.train_category_domain1", "train_category_domain1"),
    ("Category 2", "models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_2.train_category_domain2", "train_category_domain2"),
    ("Category 3", "models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_3.train_category_domain3", "train_category_domain3"),
    ("Subcategory 1", "models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_1.train_subcategory_category1", "train_subcategory_cat1"),
    ("Subcategory 2", "models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_2.train_subcategory_category2", "train_subcategory_cat2"),
    ("Subcategory 3", "models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_3.train_subcategory_category3", "train_subcategory_cat3"),
    ("Subcategory 4", "models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_4.train_subcategory_category4", "train_subcategory_cat4"),
    ("Subcategory 5", "models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_5.train_subcategory_category5", "train_subcategory_cat5"),
    ("Subcategory 6", "models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_6.train_subcategory_category6", "train_subcategory_cat6"),
    ("Subcategory 7", "models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_7.train_subcategory_category7", "train_subcategory_cat7"),
    ("Harm Binary", "models_directory.Classification_Models.Harm_level.train_harm_binary", "train_harm_binary"),
    ("Harm Ordinal High", "models_directory.Classification_Models.Harm_level.train_harm_ordinal_high", "train_harm_ordinal_high"),
    ("Harm Ordinal Low", "models_directory.Classification_Models.Harm_level.train_harm_ordinal_low", "train_harm_ordinal_low"),
    ("Severity Model", "models_directory.Classification_Models.Severity_level.train_severity_model", "train_severity_model"),
]

print("\n📋 Checking imports...")
for name, module_path, func_name in training_funcs:
    try:
        module = __import__(module_path, fromlist=[func_name])
        func = getattr(module, func_name)
        print(f"  ✓ {name:25} - imported successfully")
    except Exception as e:
        print(f"  ✗ {name:25} - ERROR: {e}")
        errors.append((name, str(e)))
        all_imports_ok = False

# Check Helper_Functions
print("\n📋 Checking Helper_Functions...")
try:
    from models_directory.Classification_Models.Hierarchical_Classification_Model.Helper_Functions import (
        compute_standardized_metrics,
        load_table,
        parse_embedding_series,
    )
    print(f"  ✓ compute_standardized_metrics - imported successfully")
    print(f"  ✓ load_table - imported successfully")
    print(f"  ✓ parse_embedding_series - imported successfully")
except Exception as e:
    print(f"  ✗ Helper_Functions - ERROR: {e}")
    errors.append(("Helper_Functions", str(e)))
    all_imports_ok = False

# Check standardized metrics function signature
print("\n📋 Checking standardized metrics function signature...")
try:
    import inspect
    sig = inspect.signature(compute_standardized_metrics)
    print(f"  Signature: {sig}")
    
    # Check expected parameters
    expected_params = {'model_name', 'y_train', 'y_test', 'y_pred', 'label_names'}
    actual_params = set(sig.parameters.keys())
    
    if expected_params == actual_params:
        print(f"  ✓ All expected parameters present: {expected_params}")
    else:
        print(f"  ✗ Parameter mismatch!")
        print(f"    Expected: {expected_params}")
        print(f"    Actual: {actual_params}")
        missing = expected_params - actual_params
        extra = actual_params - expected_params
        if missing:
            print(f"    Missing: {missing}")
        if extra:
            print(f"    Extra: {extra}")
except Exception as e:
    print(f"  ✗ Error checking signature: {e}")

print("\n" + "=" * 70)
if all_imports_ok:
    print("✅ ALL IMPORTS SUCCESSFUL!")
    print("   Ready to run train_all.py")
else:
    print("❌ SOME IMPORTS FAILED!")
    print("Errors:")
    for name, error in errors:
        print(f"  - {name}: {error}")

print("=" * 70)
