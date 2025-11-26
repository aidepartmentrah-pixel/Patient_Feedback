# train_hierarchical.py
# FULL WORKING VERSION (patched)
"""
✓ Normalizes domain/category/sub_category (excluding NaNs)
✓ Saves JSON label maps
✓ Skips NaN labels safely (option A)
✓ Compatible with XGBoost
✓ Hierarchical training (domain → category → subcategory)
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import traceback

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================
HERE = Path(__file__).resolve().parent

PRED_DIR = HERE / "predictions"
PRED_DIR.mkdir(exist_ok=True)

DB_PATH = HERE.parent / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text1"

DOMAIN_COL = "domain"
CATEGORY_COL = "category"
SUB_CATEGORY_COL = "sub_category"

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def load_table(table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
    return df


def parse_embedding_series(series: pd.Series) -> np.ndarray:
    out = []

    for i, v in enumerate(series):

        if isinstance(v, (bytes, bytearray)):            # stored as bytes
            arr = np.frombuffer(v, dtype=np.float32)

        elif isinstance(v, str):                         # stored as JSON string
            try:
                arr = np.array(json.loads(v), dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Row {i}: Failed to parse embedding JSON: {e}")

        elif isinstance(v, list):                        # already list
            arr = np.array(v, dtype=np.float32)

        else:
            raise TypeError(f"Row {i}: Unsupported embedding type: {type(v)}")

        out.append(arr)

    # Validate dimensional consistency
    lengths = {len(a) for a in out}
    if len(lengths) != 1:
        raise ValueError(f"Embedding vectors have inconsistent dimensionality: {lengths}")

    return np.vstack(out)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_labels(series_train, series_test, mapping_file=None):
    """
    NO-OP mapping: keep original labels as-is.

    We removed the previous global remapping because the hierarchical
    pipeline performs local mapping inside train_three_models. Keeping
    the original labels (e.g. 5,7 for category) is required so the
    local mapping later can correctly map them to local indices.
    """
    # Make copies to avoid accidental in-place edits later
    series_train_mapped = series_train.copy()
    series_test_mapped = series_test.copy()
    return series_train_mapped, series_test_mapped

def train_three_models(X_train, y_train, X_test, y_test, out_dir: Path, label_name: str):
    """
    Trains LogReg, RF, XGB on inputs.
    - X_train, X_test: numpy arrays (n_samples, n_features)
    - y_train, y_test: 1D arrays (can be pandas Series or numpy arrays) of labels (integers or strings)
    Returns:
      results: dict of preds arrays { 'lr': np.array(...), 'rf': ..., 'xgb': ... }
      kept_test_mask: boolean numpy array of length == len(y_test) indicating which test rows were kept
                      (True = prediction exists at corresponding index)
    """

    ensure_dir(out_dir)
    results = {"lr": np.array([], dtype=object), "rf": np.array([], dtype=object), "xgb": np.array([], dtype=object)}

    # Convert to numpy arrays for consistent indexing
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Initial kept masks (relative to the provided arrays)
    kept_mask_test = np.ones(len(y_test), dtype=bool)

    # ---------------- Remove NaN labels (train/test) ----------------
    train_notna_mask = ~pd.isna(y_train)
    test_notna_mask = ~pd.isna(y_test)

    if np.sum(train_notna_mask) == 0:
        # Nothing to train on
        return results, np.zeros(len(y_test), dtype=bool)

    if np.sum(test_notna_mask) == 0:
        # No valid test rows
        return results, np.zeros(len(y_test), dtype=bool)

    X_train_filtered = X_train[train_notna_mask]
    y_train_filtered = y_train[train_notna_mask]

    X_test_filtered = X_test[test_notna_mask]
    y_test_filtered = y_test[test_notna_mask]

    # keep track of mask relative to original y_test (subset)
    kept_mask_test = np.zeros(len(y_test), dtype=bool)
    # We'll fill kept_mask_test for positions where test_notna_mask is True and later where unseen classes are also excluded.

    # ---------------- Local mapping (classes from y_train_filtered) ----------------
    unique_vals = sorted(set(y_train_filtered))
    local_map = {v: i for i, v in enumerate(unique_vals)}

    # ---- SAVE LABEL ENCODER ----
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(unique_vals)
    joblib.dump(le, out_dir / f"{label_name}_label_encoder.pkl")

    # If after mapping there's only one class, sklearn may still work, but metrics can be trivial.
    y_train_local = np.array([local_map[v] for v in y_train_filtered])

    # Determine which test rows have classes seen in training
    valid_test_mask_local = np.array([v in local_map for v in y_test_filtered])

    if np.sum(valid_test_mask_local) == 0:
        # No test samples with seen classes
        return results, kept_mask_test  # all False

    X_test_final = X_test_filtered[valid_test_mask_local]
    y_test_final = y_test_filtered[valid_test_mask_local]
    y_test_local = np.array([local_map[v] for v in y_test_final])

    # Build final kept_mask_test relative to original y_test
    # Fill positions where test_notna_mask is True and valid_test_mask_local True
    kept_positions = np.where(test_notna_mask)[0]  # indices in original y_test corresponding to filtered test rows
    kept_positions = kept_positions[valid_test_mask_local]  # those that also passed valid_test_mask_local
    kept_mask_test[kept_positions] = True

    # ---------------- If we got here, we have training and test data ----------------
    # Fit vocab_models on X_train_filtered / y_train_local, predict on X_test_final
    try:
        # ---------------- LogReg ----------------
        lr = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
        lr.fit(X_train_filtered, y_train_local)
        pred_lr = lr.predict(X_test_final)
        joblib.dump(lr, out_dir / f"{label_name}_logreg.pkl")

        # ---------------- RF ----------------
        rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        rf.fit(X_train_filtered, y_train_local)
        pred_rf = rf.predict(X_test_final)
        joblib.dump(rf, out_dir / f"{label_name}_rf.pkl")

        # ---------------- XGB ----------------
        xgb = XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1
        )
        xgb.fit(X_train_filtered, y_train_local)
        pred_xgb = xgb.predict(X_test_final)
        joblib.dump(xgb, out_dir / f"{label_name}_xgb.pkl")

        # ---------------- Save classification reports (for the filtered test set) ----------------
        with open(out_dir / f"{label_name}_metrics.txt", "w", encoding="utf-8") as f:
            f.write("---- LOGREG (local labels) ----\n")
            f.write(classification_report(y_test_local, pred_lr, zero_division=0))
            f.write("\n\n---- RF (local labels) ----\n")
            f.write(classification_report(y_test_local, pred_rf, zero_division=0))
            f.write("\n\n---- XGB (local labels) ----\n")
            f.write(classification_report(y_test_local, pred_xgb, zero_division=0))

        # ---------------- MAP LOCAL PREDICTIONS → GLOBAL/ORIGINAL LABELS ----------------
        # unique_vals is the ordered list of original *global* labels seen in y_train_filtered
        inv_map = list(unique_vals)  # index i -> original global label (these are the values present in y_train_filtered)

        # If you have the top-level mapping file (mappings/category.json / mappings/sub_category.json / mappings/domain.json)
        # prefer to use it to convert to the exact original representation (int/string).
        mapping_file = None
        if label_name.startswith("category"):
            mapping_file = HERE / "mappings" / "category.json"
        elif label_name.startswith("subcat") or label_name.startswith("sub_category"):
            mapping_file = HERE / "mappings" / "sub_category.json"
        elif label_name == "domain":
            mapping_file = HERE / "mappings" / "domain.json"

        if mapping_file and mapping_file.exists():
            try:
                with open(mapping_file, "r", encoding="utf-8") as mf:
                    j = json.load(mf)
                    # reverse mapping in normalize_labels was saved as {int_index_str: original_label_str}
                    reverse_map = j.get("reverse", {})
                    # convert to python types when possible
                    def conv_val(v):
                        # try int, then float, else keep string
                        try:
                            return int(v)
                        except Exception:
                            try:
                                return float(v)
                            except Exception:
                                return v
                    # build map: global_index (int) -> original_label (int/float/str)
                    rev_map = {int(k): conv_val(v) for k, v in reverse_map.items()}
                    # now turn inv_map (which are global indices) into the original label values
                    inv_map = [rev_map[int(g)] for g in inv_map]
            except Exception:
                # if parsing fails, keep inv_map as-is (which are global indices)
                pass

        # Defensive checks
        max_local = max(len(inv_map) - 1, 0)
        if pred_lr.size > 0 and pred_lr.max() > max_local:
            raise ValueError(f"{label_name}: pred_lr contains local index > {max_local}")
        if pred_rf.size > 0 and pred_rf.max() > max_local:
            raise ValueError(f"{label_name}: pred_rf contains local index > {max_local}")
        if pred_xgb.size > 0 and pred_xgb.max() > max_local:
            raise ValueError(f"{label_name}: pred_xgb contains local index > {max_local}")

        # map local -> original (global) labels
        def map_local_to_global(pred_array):
            if pred_array.size == 0:
                return np.array([], dtype=object)
            return np.array([int(inv_map[int(i)]) for i in pred_array], dtype=object)

        pred_lr_global = map_local_to_global(pred_lr)
        pred_rf_global = map_local_to_global(pred_rf)
        pred_xgb_global = map_local_to_global(pred_xgb)

        # Save the local->global mapping JSON for debugging (human-readable)
        mapping_json = out_dir / f"{label_name}_local2global.json"
        clean_map = [int(x) for x in inv_map]  # convert all to pure Python ints
        with open(mapping_json, "w", encoding="utf-8") as f:
            json.dump({"local_to_global": clean_map}, f, ensure_ascii=False, indent=2)

        # ---------------- RETURN GLOBAL-LABELED PREDICTIONS ----------------
        results["lr"] = pred_lr_global
        results["rf"] = pred_rf_global
        results["xgb"] = pred_xgb_global

        return results, kept_mask_test


    except Exception as e:
        # In case model fitting fails, return empty results and mask
        print(f"Error training {label_name}: {e}")
        traceback.print_exc()
        return results, np.zeros(len(y_test), dtype=bool)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    try:
        print("\n=== Loading data ===")
        df_train = load_table(TRAIN_TABLE)
        df_test = load_table(TEST_TABLE)

        print("Parsing embeddings…")
        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test = parse_embedding_series(df_test[EMBED_COL])

        final_preds = {"domain": {}, "category": {}, "sub_category": {}}
        final_truths = {"domain": None, "category": None, "sub_category": None}

        # ====================================================
        # NORMALIZE LABELS
        # ====================================================
        print("\n--- Normalizing labels ---")
        os.makedirs("mappings", exist_ok=True)

        df_train[DOMAIN_COL], df_test[DOMAIN_COL] = normalize_labels(
            df_train[DOMAIN_COL], df_test[DOMAIN_COL], "mappings/domain.json"
        )

        df_train[CATEGORY_COL], df_test[CATEGORY_COL] = normalize_labels(
            df_train[CATEGORY_COL], df_test[CATEGORY_COL], "mappings/category.json"
        )

        df_train[SUB_CATEGORY_COL], df_test[SUB_CATEGORY_COL] = normalize_labels(
            df_train[SUB_CATEGORY_COL], df_test[SUB_CATEGORY_COL], "mappings/sub_category.json"
        )

        # ====================================================
        # 1. DOMAIN MODELS
        # ====================================================
        print("\n--- TRAINING DOMAIN MODELS ---")

        y_train_domain = df_train[DOMAIN_COL].to_numpy()
        y_test_domain = df_test[DOMAIN_COL].to_numpy()

        out_dir = HERE / "domain"
        preds_domain, kept_mask_domain = train_three_models(
            X_train, y_train_domain, X_test, y_test_domain, out_dir, "domain"
        )
        final_preds["domain"] = preds_domain
        final_truths["domain"] = np.array(y_test_domain)[kept_mask_domain]
        np.save(PRED_DIR / "y_true_domain.npy", final_truths["domain"])

        # ====================================================
        # 2. CATEGORY per DOMAIN
        # ====================================================
        print("\n--- TRAINING CATEGORY MODELS ---")

        domain_values = sorted(df_train[DOMAIN_COL].dropna().unique())

        # We'll collect predictions per-domain along with masks so we can rebuild global arrays later
        category_preds_per_domain = []  # list of dicts: { 'domain': dom, 'test_mask': test_mask, 'kept_mask': kept_mask, 'preds': preds }
        cat_preds_all = {"lr": [], "rf": [], "xgb": []}
        cat_truth = []

        for dom in domain_values:
            print(f"Domain {dom}")

            train_mask = (df_train[DOMAIN_COL] == dom).to_numpy()
            test_mask = (df_test[DOMAIN_COL] == dom).to_numpy()

            Xtr = X_train[train_mask]
            Xte = X_test[test_mask]

            ytr = df_train.loc[train_mask, CATEGORY_COL].to_numpy()
            yte = df_test.loc[test_mask, CATEGORY_COL].to_numpy()

            out_dir = HERE / "category" / f"domain_{dom}"
            preds, kept_mask = train_three_models(Xtr, ytr, Xte, yte, out_dir, f"category_d{dom}")

            # store per-domain info for global reconstruction
            category_preds_per_domain.append({
                "domain": dom,
                "test_mask": test_mask,         # boolean array length n_test that marks domain rows (global)
                "kept_mask": kept_mask,         # boolean array length == len(yte) (domain-local) marking which test rows remained
                "preds": preds                  # dict of model_name -> numpy array predictions for the filtered domain test rows (order matches kept_mask True subset)
            })

            # Extend concatenated lists (kept-order) for local final truth/preds (existing behavior)
            for k in preds:
                if preds[k].size > 0:
                    cat_preds_all[k].extend(preds[k].tolist())
            if np.any(kept_mask):
                cat_truth.extend(np.array(yte)[kept_mask].tolist())

        cat_truth = np.array(cat_truth)

        # Finalize concatenated per-domain predictions (kept order)
        cat_truth = np.array(cat_truth)
        final_preds["category"] = {k: np.array(v) for k, v in cat_preds_all.items()}

        # Save GLOBAL truth vector aligned with full test set
        y_true_cat_global = df_test[CATEGORY_COL].to_numpy()
        final_truths["category"] = y_true_cat_global
        np.save(PRED_DIR / "y_true_category.npy", y_true_cat_global)

        # DO NOT load global predictions here.
        # They are built later during the global reconstruction step.
        # ====================================================
        # 3. SUBCATEGORY per CATEGORY
        # ====================================================
        print("\n--- TRAINING SUBCATEGORY MODELS ---")

        category_values = sorted(df_train[CATEGORY_COL].dropna().unique())

        sub_preds_per_category = []  # similar structure to category_preds_per_domain
        sub_preds_all = {"lr": [], "rf": [], "xgb": []}
        sub_truth = []

        for cat in category_values:
            print(f"Category {cat}")

            train_mask = (df_train[CATEGORY_COL] == cat).to_numpy()
            test_mask = (df_test[CATEGORY_COL] == cat).to_numpy()

            Xtr = X_train[train_mask]
            Xte = X_test[test_mask]

            ytr = df_train.loc[train_mask, SUB_CATEGORY_COL].to_numpy()
            yte = df_test.loc[test_mask, SUB_CATEGORY_COL].to_numpy()

            out_dir = HERE / "sub_category" / f"cat_{cat}"
            preds, kept_mask = train_three_models(Xtr, ytr, Xte, yte, out_dir, f"subcat_c{cat}")

            # store per-category info for global reconstruction
            sub_preds_per_category.append({
                "category": cat,
                "test_mask": test_mask,    # boolean array length n_test marking category rows
                "kept_mask": kept_mask,    # boolean array length == len(yte) (category-local)
                "preds": preds
            })

            # Extend concatenated lists (existing)
            for k in preds:
                if preds[k].size > 0:
                    sub_preds_all[k].extend(preds[k].tolist())
            if np.any(kept_mask):
                sub_truth.extend(np.array(yte)[kept_mask].tolist())

        sub_truth = np.array(sub_truth)

        # True labels in full test set
        y_true_sub_global = df_test[SUB_CATEGORY_COL].to_numpy()
        final_truths["sub_category"] = y_true_sub_global
        np.save(PRED_DIR / "y_true_sub_category.npy", y_true_sub_global)




        # ====================================================
        # 4. FINAL REPORTS
        # ====================================================
        print("\n=== FINAL SUMMARY REPORTS ===")



        # ====================================================
        # 3B. BUILD GLOBAL PREDICTION VECTORS (for synthetic single-head reports)
        # ====================================================
        print("\n--- BUILDING GLOBAL PREDICTION ARRAYS ---")

        n_test = len(df_test)
        def write_final_report(name, y_true, preds_dict):
            with open(HERE / f"final_{name}_report.txt", "w", encoding="utf-8") as f:
                for model_name, preds in preds_dict.items():
                    f.write(f"\n=== {name.upper()} — {model_name.upper()} ===\n")

                    if y_true is None or preds is None or len(preds) == 0:
                        f.write("No samples / predictions for this split.\n")
                        continue

                    # Ensure equal lengths
                    if len(y_true) != len(preds):
                        f.write(f"Length mismatch: y_true={len(y_true)}, preds={len(preds)}\n")
                        continue

                    # --- FIX: remove NaN ---
                    mask = ~pd.isna(y_true) & ~pd.isna(preds)
                    yt = y_true[mask]
                    yp = preds[mask]

                    if len(yt) == 0:
                        f.write("No valid samples for report (all NaN).\n")
                    else:
                        f.write(classification_report(yt, yp, zero_division=0))

        # ---------- DOMAIN ----------
        print("\n--- FINAL REPORT: DOMAIN ---")

        # Truth (aligned with test set)
        y_true_domain_global = df_test["domain"].to_numpy()
        final_truths["domain"] = y_true_domain_global
        np.save(PRED_DIR / "y_true_domain.npy", y_true_domain_global)

        # Predictions (global vectors saved earlier)
        final_preds["domain"] = {
            "lr": np.load(PRED_DIR / "domain_pred_global_lr.npy"),
            "rf": np.load(PRED_DIR / "domain_pred_global_rf.npy"),
            "xgb": np.load(PRED_DIR / "domain_pred_global_xgb.npy"),
        }

        write_final_report("domain", final_truths["domain"], final_preds["domain"])

        # -------------- CATEGORY (3 per-domain vocab_models → 1 global vector per algo) --------------
        for model_name in ["lr", "rf", "xgb"]:
            vec = np.full(n_test, np.nan)

            # We must fill domain-by-domain using masks + kept masks + preds_per_domain
            for entry in category_preds_per_domain:
                test_mask_global = entry["test_mask"]  # global mask
                kept_mask_local = entry["kept_mask"]  # which rows kept within domain
                preds_local = entry["preds"][model_name]  # only kept rows

                # global positions of all rows in this domain
                global_indices_all = np.where(test_mask_global)[0]

                # positions that actually correspond to valid predictions (kept rows)
                kept_local_positions = np.where(kept_mask_local)[0]

                # these are the TRUE indices in the global test set where predictions go
                final_positions = global_indices_all[kept_local_positions]

                if len(final_positions) != len(preds_local):
                    raise ValueError(
                        f"Category mismatch: {len(final_positions)} slots vs {len(preds_local)} preds"
                    )

                vec[final_positions] = preds_local

            np.save(PRED_DIR / f"category_pred_global_{model_name}.npy", vec)


        # -------------- SUBCATEGORY (correct reconstruction using kept_mask) --------------
        for model_name in ["lr", "rf", "xgb"]:
            vec = np.full(n_test, np.nan)

            for entry in sub_preds_per_category:
                test_mask_global = entry["test_mask"]  # all category rows
                kept_mask_local = entry["kept_mask"]  # which ones kept
                preds_local = entry["preds"][model_name]  # predictions for kept rows only

                global_indices_all = np.where(test_mask_global)[0]
                kept_local_indices = np.where(kept_mask_local)[0]

                final_positions = global_indices_all[kept_local_indices]

                if len(final_positions) != len(preds_local):
                    raise ValueError(
                        f"Subcat mismatch: {len(final_positions)} slots vs {len(preds_local)} preds"
                    )

                vec[final_positions] = preds_local

            np.save(PRED_DIR / f"subcat_pred_global_{model_name}.npy", vec)

        print("✔ Saved global prediction arrays.")

        final_preds["category"] = {
            "lr": np.load(PRED_DIR / "category_pred_global_lr.npy"),
            "rf": np.load(PRED_DIR / "category_pred_global_rf.npy"),
            "xgb": np.load(PRED_DIR / "category_pred_global_xgb.npy"),
        }

        final_preds["sub_category"] = {
            "lr": np.load(PRED_DIR / "subcat_pred_global_lr.npy"),
            "rf": np.load(PRED_DIR / "subcat_pred_global_rf.npy"),
            "xgb": np.load(PRED_DIR / "subcat_pred_global_xgb.npy")
        }



        write_final_report("domain", final_truths["domain"], final_preds["domain"])
        write_final_report("category", final_truths["category"], final_preds["category"])
        write_final_report("sub_category", final_truths["sub_category"], final_preds["sub_category"])

        print("\n🎉 ALL MODELS + FINAL REPORTS GENERATED SUCCESSFULLY 🎉\n")

    except Exception:
        print("❌ ERROR:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
