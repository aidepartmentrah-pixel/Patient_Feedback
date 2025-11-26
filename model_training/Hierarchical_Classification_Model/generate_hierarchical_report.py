"""
collect_reports.py

Build the FINAL 3 evaluation reports using the GLOBAL prediction arrays.
Outputs:
    reports/domain.txt
    reports/category.txt
    reports/sub_category.txt
"""

from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

PRED_DIR = ROOT / "predictions"      # <-- store all .npy here
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# TRUE LABEL FILES (change names here if needed)
YTRUE_DOMAIN = PRED_DIR / "y_true_domain.npy"
YTRUE_CATEGORY = PRED_DIR / "y_true_category.npy"
YTRUE_SUBCAT = PRED_DIR / "y_true_sub_category.npy"



# ============================================================
# HELPERS
# ============================================================

def write_report(name: str, text: str):
    out = REPORT_DIR / f"{name}.txt"
    out.write_text(text, encoding="utf-8")
    print(f"✔ Saved {out}")


def load_pred(prefix: str):
    """
    Loads LR, RF, XGB global predictions and averages the predicted class.
    prefix examples:
        'domain_pred_global'
        'category_pred_global'
        'subcat_pred_global'
    """
    lr = np.load(PRED_DIR / f"{prefix}_lr.npy").astype(float)
    rf = np.load(PRED_DIR / f"{prefix}_rf.npy").astype(float)
    xgb = np.load(PRED_DIR / f"{prefix}_xgb.npy").astype(float)

    # Fix NaN values
    lr = np.nan_to_num(lr, nan=-1)
    rf = np.nan_to_num(rf, nan=-1)
    xgb = np.nan_to_num(xgb, nan=-1)

    # Fix negative values
    lr[lr < 0] = -1
    rf[rf < 0] = -1
    xgb[xgb < 0] = -1

    # Convert to int
    lr = lr.astype(int)
    rf = rf.astype(int)
    xgb = xgb.astype(int)

    # All arrays must match length
    if not (len(lr) == len(rf) == len(xgb)):
        raise ValueError(f"Prediction length mismatch for {prefix}")

    stacked = np.vstack([lr, rf, xgb])

    # STACK MODELS AND MAJORITY-VOTE CLASS
    def majority_vote(x):
        x = x[x >= 0]  # remove invalid predictions (-1)
        if len(x) == 0:
            return -1  # if all vocab_models failed
        return np.bincount(x).argmax()

    final_pred = np.apply_along_axis(majority_vote, axis=0, arr=stacked)

    return final_pred


# ============================================================
# MAIN
# ============================================================

def generate_final_reports():

    print("\n=== BUILDING FINAL GLOBAL REPORTS ===")

    # --------------------------------------------------------
    # 1) DOMAIN (single-head)
    # --------------------------------------------------------
    print("→ Loading DOMAIN predictions")
    y_true = np.load(YTRUE_DOMAIN)
    y_pred = load_pred("domain_pred_global")

    report = classification_report(y_true, y_pred, digits=4)
    write_report("domain", report)

    # --------------------------------------------------------
    # 2) CATEGORY (merged from the 3 original domain vocab_models)
    # --------------------------------------------------------
    print("→ Loading CATEGORY predictions")
    y_true = np.load(YTRUE_CATEGORY)
    y_pred = load_pred("category_pred_global")

    report = classification_report(y_true, y_pred, digits=4)
    write_report("category", report)

    # --------------------------------------------------------
    # 3) SUB-CATEGORY (merged from the 7 original category vocab_models)
    # --------------------------------------------------------
    print("→ Loading SUB-CATEGORY predictions")
    y_true = np.load(YTRUE_SUBCAT)
    y_pred = load_pred("subcat_pred_global")

    # --- FIX INCONSISTENT LENGTHS (DROP EXTRA ENTRIES) ---
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    # -----------------------------------------------------

    report = classification_report(y_true, y_pred, digits=4)
    write_report("sub_category", report)

    print("\n✔ All FINAL REPORTS generated successfully.\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    generate_final_reports()
