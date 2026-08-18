# ML Classification — Category/Subcategory Failure: Problem Statement for the Development Team

**Prepared:** 2026-07-23
**Audience:** Development team / ML owners
**Purpose:** Explain the exact technical problem, what has already been verified, and the two open decisions that need engineering/data-governance input before this installation can proceed.

---

## 1. The original symptom

The Patient Feedback System's "Extract & Classify" feature predicts 9 fields per complaint: domain, category, subcategory, severity, harm level, stage, feedback type, improvement opportunity type, and classification-EN. On this deployment, calling the classification endpoint returned a 500 error and no predictions at all.

## 2. Root cause

Two of the nine predictions — **category** and **subcategory** — work differently from the other seven. The trained model for each (an XGBoost classifier) doesn't output a real label directly; it outputs a plain integer class index (0, 1, 2, ...). The application code has to reconstruct what real label each index corresponds to by querying a table called `table_feedback_train` — the exact training-data snapshot used when that specific model was trained — and deriving the sorted list of distinct real label values that existed in that training run.

The other seven predictions (severity, harm, stage, feedback type, improvement opportunity, classification-EN, and domain) don't need this lookup; their label maps are static and already written into the code.

**The problem**: the SQLite file containing `table_feedback_train` (`patient_feedback_ml.db`, ~116MB) was never present anywhere in this deployment. It isn't tracked in git (`.gitignore` excludes `*.db` — the file is too large for source control) and wasn't copied onto the machine this release was built on. Because the code queries this table at **import time** (not per-request), and all 9 model families are imported together in one module, the missing table crashed the import for the category/subcategory models — and because of how the code was structured, that one failure was taking down all 9 predictions, not just the 2 that actually depend on the missing table.

## 3. What we already fixed (independent of data recovery)

We restructured the import so a failure in one model family can no longer take down the others. As of this fix:
- Domain, severity, harm level, stage, feedback type, improvement opportunity, classification-EN: **work correctly regardless of the table_feedback_train issue** (7 of 9).
- Category, subcategory: return a clean `"Not Available"` instead of crashing the whole endpoint, when their data dependency is unavailable.

This part is done and shipped. The open question below is about restoring category/subcategory specifically.

## 4. We found an archived copy of the missing file — and it partially, not fully, resolves the problem

Per `ML_ARCHITECTURE_DECISION_RECORD.md` (dated 2026-07-16), this file was migrated into a new SQL Server `ml` schema and formally retired on 2026-07-20, with a checksummed archive copy kept at:

```
C:\SQLBackup\ml_stage13_final_archive_20260720_123654\patient_feedback_ml.db
SHA-256: 2d7ccbc89a3951db0b764a75ec2d23af624e623f17e9371861aa8aee784c0851
```

We obtained a copy of this file and verified its checksum matches exactly (116,277,248 bytes, hash confirmed). We then compared its actual label distribution against what each of the 10 relevant model files (3 domain→category models, 7 category→subcategory models) actually expect (`model.n_classes_`, read directly from the deployed `.json`/`.pkl` model files):

| Model | Model expects (n_classes_) | Recovered data has (distinct labels) | Result |
|---|---|---|---|
| Domain 1 → Category | 3 | 3 | ✅ Match |
| Domain 2 → Category | 4 | 4 | ✅ Match |
| Domain 3 → Category | 4 | 4 | ✅ Match |
| Category 1 → Subcategory | 4 | **3** | ❌ **Mismatch** |
| Category 2 → Subcategory | 3 | **4** | ❌ **Mismatch** |
| Category 3 → Subcategory | 4 | 4 | ✅ Match |
| Category 4 → Subcategory | 4 | 4 | ✅ Match |
| Category 5 → Subcategory | 7 | 7 | ✅ Match |
| Category 6 → Subcategory | 4 | 4 | ✅ Match |
| Category 7 → Subcategory | 8 | 8 | ✅ Match |

**8 of 10 match exactly.** Wiring this recovered file back in restores real predictions for category (all 3 domains) and subcategory (5 of 7 categories) — live-tested and confirmed working.

**2 of 10 do not match** — Category 1's and Category 2's subcategory models are trained against a label set that doesn't match even this archived "ground truth" snapshot. We independently confirmed this isn't specific to our copy of the data: calling the classification endpoint on the **live production system** (170.70.32.34) for a complaint that routes through Category 1 produces the identical class of error (`model expects 4 classes, mapping size=3`) — meaning these two specific model files are stale relative to the training data everywhere we can check, not just here.

### Question for the dev team (technical)
Category 1 and Category 2's subcategory classifiers need retraining against current real data to be restored. Is that something the team wants to schedule, and if so, against what data source (the recovered 387-row snapshot, or something newer/larger that may exist elsewhere)?

## 5. The open decision that isn't ours to make (data governance)

The recovered file contains **387 rows of real complaint records** — actual patient-complaint free text (`complaint_text`, `immediate_action`, `taken_action` fields), not synthetic or anonymized data.

Wiring this file into the offline release properly (so it survives image rebuilds and travels with the installer, rather than being silently lost again) means packaging it as an asset bundle that ships physically with the release — it would end up sitting on the disk of every hospital server this release is installed onto.

### Question for the dev team (governance)
Is shipping this real complaint-narrative data inside the offline installer package acceptable, or does it need to be handled differently — e.g., loaded only into specifically-authorized environments, anonymized first, or excluded from the general release and handled as a separate, controlled step per deployment? We have not made this decision unilaterally and are not proceeding with packaging until we have direction here.

---

## Summary of what we need back from you

1. **Retraining decision**: do Category 1 and Category 2's subcategory models get retrained, by whom, against what data, and on what timeline?
2. **Data governance decision**: how should the recovered `patient_feedback_ml.db` (real complaint text) be handled for this and future offline installations — shipped with the release, handled as a separate controlled asset, or something else?

Everything else in this release (org units, user accounts, custom views, dashboard fixes, patient search, NER removal, STT fixes, drawer labels, model dashboard) is complete, tested, and already packaged — this ML classification item is the one piece waiting on your input before it can be finalized one way or the other.
