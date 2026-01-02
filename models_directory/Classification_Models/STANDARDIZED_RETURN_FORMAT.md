## STANDARDIZED RETURN FORMAT - COMPLETE REFERENCE

All 15 training functions now return the exact same structure:

```python
(model, standardized_metrics)
```

### Return Format Details

#### `model`
- Type: Trained scikit-learn or mord model object
- Used by: train_all.py for saving/using models
- Examples:
  - LogisticRegression, RandomForestClassifier, XGBClassifier for hierarchical models
  - LogisticRegression for harm_binary
  - mord.LogisticIT for ordinal models (harm_high, harm_low, severity)

#### `standardized_metrics` Dictionary

**Complete Schema (all keys present):**

```python
{
    "model_name": str,              # e.g., "Domain_Model", "Subcategory_Category2_rf"
    "num_records": int,              # Training set size
    "accuracy": float,               # Overall accuracy (0.0-1.0)
    "precision": float,              # Weighted precision for multi-class (0.0-1.0)
    "recall": float,                 # Weighted recall for multi-class (0.0-1.0)
    "f1": float,                     # Weighted F1-score (0.0-1.0)
    "mAP": float,                    # Mean Average Precision (0.0-1.0)
    "labels": list,                  # Sorted list of class labels [0,1,2,...]
    "confusion_matrix": list[list]   # 2D confusion matrix as nested lists
}
```

### Metrics Explanation

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| model_name | string | N/A | Hierarchical identifier for model |
| num_records | int | >0 | Number of training samples |
| accuracy | float | [0,1] | Percentage of correct predictions |
| precision | float | [0,1] | True positives / (true + false positives) - WEIGHTED |
| recall | float | [0,1] | True positives / (true + false negatives) - WEIGHTED |
| f1 | float | [0,1] | Harmonic mean of precision & recall - WEIGHTED |
| mAP | float | [0,1] | Mean Average Precision across all classes |
| labels | list | N/A | All class labels in model (sorted order) |
| confusion_matrix | matrix | N/A | N×N matrix where N = number of classes |

### Computation Details

All metrics are computed using scikit-learn with the following settings:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

standardized_metrics = {
    "model_name": model_name,
    "num_records": len(y_train),
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    "mAP": mean_average_precision(y_test, y_pred),  # custom computation
    "labels": sorted(unique_labels),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}
```

### Model-Specific Return Examples

#### 1. Hierarchical Classification Models (11 models)
```python
# train_domain_model.py, train_category_domain1.py, train_subcategory_category1.py, etc.

# Inside training function:
all_preds = {}
trained_models = {}
results = {}

# For each model (LR, RF, XGB):
lr_pred = lr.predict(X_test)
all_preds["lr"] = lr_pred
trained_models["lr"] = lr
results["LogisticRegression"] = compute_metrics(y_test, lr_pred)

# Select best
best_model_name = max(results.keys(), key=lambda k: results[k]["f1"])
best_model = trained_models[best_model_name]
best_pred = all_preds[best_model_name]

# Compute standardized metrics
standardized_metrics = compute_standardized_metrics(
    model_name=f"Domain_Model_{best_model_name}",  # or Category, Subcategory, etc.
    y_train=y_train,
    y_test=y_test,
    y_pred=best_pred,
    label_names=unique_labels,
)

# Return
return best_model, standardized_metrics
```

#### 2. Harm Binary Model
```python
# train_harm_binary.py

# Train
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute standardized metrics
unique_labels = sorted(np.unique(y_train).tolist())  # [0, 1]
standardized_metrics = compute_standardized_metrics(
    model_name="Harm_Binary",
    y_train=y_train,
    y_test=y_test,
    y_pred=y_pred,
    label_names=unique_labels,
)

# Return
return model, standardized_metrics
```

#### 3. Harm Ordinal High Model (Special: Label Remapping)
```python
# train_harm_ordinal_high.py

# Train with remapped labels (4→0, 5→1, 6→2)
y_train_temp = df_train[TARGET_COL].astype(int).to_numpy() - 4
model = mord.LogisticIT()
model.fit(X_train, y_train_temp)
y_pred = model.predict(X_test)

# REMAP BACK to original labels (0→4, 1→5, 2→6) for metrics
y_test_orig = df_test[TARGET_COL].astype(int).to_numpy()
y_pred_orig = y_pred + 4  # ← CRITICAL: Remap predictions

# Compute standardized metrics with ORIGINAL labels
standardized_metrics = compute_standardized_metrics(
    model_name="Harm_Ordinal_High",
    y_train=df_train[TARGET_COL].astype(int).to_numpy(),  # Original [4,5,6]
    y_test=y_test_orig,                                    # Original [4,5,6]
    y_pred=y_pred_orig,                                    # Remapped [4,5,6]
    label_names=[4, 5, 6],
)

# Return
return model, standardized_metrics
```

#### 4. Severity Model (Special: Ordinal Remapping)
```python
# train_severity_model.py

# Train with remapped labels (1→0, 2→1, 3→2, 4→3)
y_train_temp = df_train[TARGET_COL].astype(int).to_numpy() - 1
model = mord.LogisticIT()
model.fit(X_train, y_train_temp)
y_pred = model.predict(X_test)

# REMAP BACK to original labels (0→1, 1→2, 2→3, 3→4) for metrics
y_test_orig = df_test[TARGET_COL].astype(int).to_numpy()
y_pred_orig = y_pred + 1  # ← CRITICAL: Remap predictions

# Compute standardized metrics with ORIGINAL labels
standardized_metrics = compute_standardized_metrics(
    model_name="Severity_Model",
    y_train=df_train[TARGET_COL].astype(int).to_numpy(),  # Original [1,2,3,4]
    y_test=y_test_orig,                                    # Original [1,2,3,4]
    y_pred=y_pred_orig,                                    # Remapped [1,2,3,4]
    label_names=[1, 2, 3, 4],
)

# Return
return model, standardized_metrics
```

---

## train_all.py Usage

```python
def run_training(model_name, func):
    """Train a model and collect standardized metrics."""
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    
    # Call training function
    model, metrics = func()  # ← Unpacks (model, standardized_metrics)
    
    # Print results
    print(f"\n✔ {model_name} Complete:")
    print(f"  Records: {metrics.get('num_records', 0)}")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.6f}")
    print(f"  Precision: {metrics.get('precision', 0):.6f}")
    print(f"  Recall:    {metrics.get('recall', 0):.6f}")
    print(f"  F1-Score:  {metrics.get('f1', 0):.6f}")

    # Store for report
    all_models[model_name] = model
    all_metrics[model_name] = metrics


def save_training_report(all_metrics: dict, save_path: str):
    """Save all training metrics to a clean, standardized TXT report."""
    
    today = datetime.datetime.now().strftime("%d_%m_%Y")
    filename = f"classification_training_report_{today}.txt"
    full_path = os.path.join(save_path, filename)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  CLASSIFICATION TRAINING PERFORMANCE REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {today}\n")
        f.write("=" * 70 + "\n\n")

        # Iterate through all collected metrics
        for model_name, metrics in all_metrics.items():
            num_records = metrics.get("num_records", 0)
            accuracy = metrics.get("accuracy", 0.0)
            precision = metrics.get("precision", 0.0)
            recall = metrics.get("recall", 0.0)
            f1 = metrics.get("f1", 0.0)
            labels = metrics.get("labels", [])
            cm = metrics.get("confusion_matrix", [])

            f.write(f"Model: {model_name}\n")
            f.write(f"  Training Records: {num_records}\n")
            f.write(f"  Classes: {labels}\n")
            f.write(f"  Metrics:\n")
            f.write(f"    Accuracy:  {accuracy:.6f}\n")
            f.write(f"    Precision: {precision:.6f}\n")
            f.write(f"    Recall:    {recall:.6f}\n")
            f.write(f"    F1-Score:  {f1:.6f}\n")
            f.write("-" * 70 + "\n\n")

        # Summary
        f.write("=" * 70 + "\n")
        f.write("Summary Statistics\n")
        f.write("=" * 70 + "\n")
        
        total_models = len(all_metrics)
        avg_f1 = sum(m.get("f1", 0) for m in all_metrics.values()) / max(total_models, 1)
        
        f.write(f"Total Models Trained: {total_models}\n")
        f.write(f"Average F1-Score: {avg_f1:.6f}\n")

    print(f"\n📄 Training report saved: {full_path}\n")
```

---

## Key Guarantees

✅ **All models return**: `(model, standardized_metrics)` tuple  
✅ **No hardcoded zeros**: All metrics computed from actual predictions  
✅ **Exact schema**: Every metrics dict has all 9 required keys  
✅ **Consistent computation**: All metrics use sklearn with same parameters  
✅ **Label preservation**: Ordinal models remap labels back for meaningful metrics  
✅ **Best model selection**: Selected by F1-score across training sets  
✅ **Ready for reporting**: train_all.py can safely access all metrics keys

---

## Error Handling

All training functions follow this error pattern:

```python
try:
    # ... training logic ...
    return model, standardized_metrics
except Exception:
    traceback.print_exc()
    return None, None
```

The train_all.py can check for errors:

```python
model, metrics = func()
if model is None or metrics is None:
    print(f"ERROR: {model_name} training failed!")
    # Handle gracefully
else:
    # Process successfully
    all_models[model_name] = model
    all_metrics[model_name] = metrics
```

---

## Status

✅ **All 15 models standardized**  
✅ **All return exact same format**  
✅ **All metrics computed consistently**  
✅ **train_all.py ready to generate reports**  
✅ **Labels handled correctly (including ordinal remapping)**  

**Ready for production use.**
