# COMPLETE FIX PLAN: Classification Pipeline

## Problem Summary

The SQLite training database is CORRECT. Models predict CORRECT IDs.
But validation code in `hierarchical_predictor.py` has WRONG mappings that OVERRIDE correct predictions with wrong values.

---

## SQLite Training Database (SOURCE OF TRUTH)

```
DOMAIN -> CATEGORY:
  Domain 1 -> Categories: [5, 6, 7]
  Domain 2 -> Categories: [2, 4, 5]
  Domain 3 -> Categories: [1, 2, 3, 4]

CATEGORY -> SUBCATEGORY:
  Category 1 -> Subcategories: [1, 2, 3, 4]
  Category 2 -> Subcategories: [3, 5, 6, 14]
  Category 3 -> Subcategories: [7, 8]
  Category 4 -> Subcategories: [9, 10, 11, 12]
  Category 5 -> Subcategories: [13, 14, 15, 16, 17, 18, 19]
  Category 6 -> Subcategories: [19, 20, 21]
  Category 7 -> Subcategories: [22, 24, 25, 26, 27]

SEVERITY_LEVEL: [1, 2, 3, 6]
HARM_LEVEL: [1, 2, 3, 4, 5, 6]
STAGE: [1, 2, 3, 4, 5, 6, 8]
```

---

## FILES TO FIX (in order)

### 1. hierarchical_predictor.py (CRITICAL)

Path: `models_directory/Classification_Models/Hierarchical_Classification_Model/hierarchical_predictor.py`

#### Change 1A: DOMAIN_TO_CATEGORIES (Line ~59)

CURRENT (WRONG):
```python
DOMAIN_TO_CATEGORIES = {
    1: [6, 7],
    2: [4, 5],
    3: [1, 2, 3]
}
```

REPLACE WITH:
```python
DOMAIN_TO_CATEGORIES = {
    1: [5, 6, 7],
    2: [2, 4, 5],
    3: [1, 2, 3, 4]
}
```

#### Change 1B: CATEGORY_TO_SUBCATEGORIES (Line ~65)

CURRENT (WRONG):
```python
CATEGORY_TO_SUBCATEGORIES = {
    1: [1, 2, 4, 3],
    2: [6, 5],
    3: [7, 8],
    4: [9, 11, 12, 10],
    5: [13, 14, 15, 16, 18, 17],
    6: [20, 19, 21],
    7: [5, 15, 16, 18, 22, 29]
}
```

REPLACE WITH:
```python
CATEGORY_TO_SUBCATEGORIES = {
    1: [1, 2, 3, 4],
    2: [3, 5, 6, 14],
    3: [7, 8],
    4: [9, 10, 11, 12],
    5: [13, 14, 15, 16, 17, 18, 19],
    6: [19, 20, 21],
    7: [22, 24, 25, 26, 27]
}
```

---

### 2. package_models.py (HIGH PRIORITY)

Path: `models_directory/Classification_Models/package_models.py`

#### Change 2A: CATEGORY_MAP (Line ~25)

The category labels should match what the frontend/database expects for each ID.
Verify with your SQL Server APP_LOOKUP_CATEGORY table and update accordingly.

#### Change 2B: SUBCATEGORY_MAP (Line ~35)

Update to match what subcategory IDs mean in your system.

#### Change 2C: SEVERITY Adapter

The model outputs 1-4 (ordinal + 1).
But training DB has severity values: [1, 2, 3, 6]
Production DB has: 1=Low, 2=Medium, 3=High

If models were trained on [1, 2, 3, 6] and output means:
- 1 = Severity level 1
- 2 = Severity level 2  
- 3 = Severity level 3
- 4 = Severity level 6 (ordinal 3 + 1)

Then adapter should map model output to production DB IDs.

#### Change 2D: HARM Adapter

The model outputs 1-6 (binary split then ordinal).
Training DB has harm values: [1, 2, 3, 4, 5, 6]
Production DB has only 5 levels: 1=No Harm, 2=Minor, 3=Moderate, 4=Severe, 5=Death

Need to verify what each training value represents and map to production.

---

### 3. predict_category_domain2.py (LOW PRIORITY - TYPO)

Path: `models_directory/Classification_Models/Hierarchical_Classification_Model/category/domain_2/predict_category_domain2.py`

Line 42: Change `_xgxb_model_path` to `_xgb_model_path`

---

## PIPELINE FLOW

```
                    TRAINING
                       |
train_all.py ----------+
      |                |
      +-> Domain Model (predicts 1, 2, 3)
      +-> Category Models (one per domain)
      +-> Subcategory Models (one per category)
      +-> Severity Model
      +-> Harm Model
      +-> Stage Model
                       |
                       v
              Models saved to disk
                       |
                       |
                  PREDICTION
                       |
hierarchical_predictor.py <-------- PROBLEM HERE (validation maps wrong)
      |
      +-> predict_domain() -> domain_id
      +-> predict_category_domain{N}() -> category_id
      +-> VALIDATION: if category not in DOMAIN_TO_CATEGORIES[domain]...  <-- WRONG MAP!
      +-> predict_subcategory_category{N}() -> subcategory_id
      +-> VALIDATION: if subcategory not in CATEGORY_TO_SUBCATEGORIES[category]...  <-- WRONG MAP!
                       |
                       v
package_models.py::classify_feedback()
      |
      +-> hierarchical_predict_embeddings() -> domain, category, subcategory
      +-> predict_severity_from_embedding() -> severity_id (might need adapter)
      +-> predict_harm_from_embedding() -> harm_id (might need adapter)
      +-> classify_stage_Score_Based() -> stage_id
      +-> Returns result dict
                       |
                       v
classification_service.py -> Returns to frontend
```

---

## WHAT HAPPENS CURRENTLY (BROKEN)

1. Model correctly predicts `category=5` for `domain=1`
2. `hierarchical_predictor.py` validation says: "Domain 1 only has [6, 7]"
3. Since 5 is not in [6, 7], code OVERRIDES to first valid: `category=6`
4. Frontend receives WRONG category

---

## WHAT WILL HAPPEN AFTER FIX

1. Model correctly predicts `category=5` for `domain=1`
2. `hierarchical_predictor.py` validation says: "Domain 1 has [5, 6, 7]"
3. Since 5 IS in [5, 6, 7], prediction passes through unchanged
4. Frontend receives CORRECT category

---

## IMPLEMENTATION STEPS

### Step 1: Fix hierarchical_predictor.py (CRITICAL)

Update both maps to match SQLite training data.

### Step 2: Fix predict_category_domain2.py typo (LOW)

Just fix the typo for clean code.

### Step 3: Review package_models.py maps (HIGH)

Ensure CATEGORY_MAP, SUBCATEGORY_MAP labels match your database.

### Step 4: Review Severity/Harm adapters (HIGH)

Verify the mapping between what models output and what production DB expects.

### Step 5: Test end-to-end

Run classification on sample text and verify all IDs are correct.

---

## QUICK VERIFICATION AFTER FIX

Run this to verify the fix works:

```python
from models_directory.Classification_Models.package_models import classify_feedback

result = classify_feedback("The nurse did not respond quickly to my call.", "", "")
print(f"Domain: {result['domain_id']} - {result['domain']}")
print(f"Category: {result['category_id']} - {result['category']}")
print(f"Subcategory: {result['sub_category_id']} - {result['sub_category']}")
print(f"Severity: {result['severity_id']} - {result['severity_level']}")
print(f"Harm: {result['harm_level_id']} - {result['harm_level']}")
print(f"Stage: {result['stage_id']} - {result['stage']}")
```

Verify that:
1. IDs match your production database
2. Labels make semantic sense for the input text
3. No "UNKNOWN" labels appear
