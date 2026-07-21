"""
Stage 9 (Slice A) — Versioned Training-Run Infrastructure.

Shared by the 3 trainers migrated in this slice (domain, harm_binary,
feedback_type — see train_all.py). Every training run gets its own folder
under training_runs/{run_id}/, never overwritten, with a durable
run_summary.json (created at run start, updated after each model, finalized
at the end) plus an artifact manifest (type/path/size/checksum) that Stage
10's ZIP download will build from.

Training here never writes to a family's fixed live-inference path as a side
effect — that would be silent, automatic promotion, which
ML_ARCHITECTURE_DECISION_RECORD.md explicitly says must never happen.
promote_run() is the one deliberate, explicit action that does that copy;
nothing in this module or in train_all.py calls it automatically.
"""

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

RUNS_ROOT = Path(__file__).resolve().parent / "training_runs"
SUMMARY_FILENAME = "run_summary.json"

# Run IDs are used as directory names — strict allowlist closes off path
# traversal for any externally-supplied run_id, not just ones we generate.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def generate_run_id() -> str:
    """YYYY_MM_DD_HHMMSS_<8-char-uuid> — the old YYYY_MM_DD_HHMM format plus
    the API's is_training_running() guard could still collide within the
    same minute (standalone calls, retries, tests); the uuid suffix makes
    collision practically impossible regardless of caller."""
    return f"{datetime.now().strftime('%Y_%m_%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def validate_run_id(run_id: str) -> None:
    if not run_id or not _RUN_ID_RE.match(run_id):
        raise ValueError(f"Invalid run_id: {run_id!r} (must match {_RUN_ID_RE.pattern})")


def get_run_dir(run_id: str) -> Path:
    """Creates and returns training_runs/{run_id}/. Raises on collision
    (exist_ok=False) rather than silently reusing/overwriting a prior run's
    folder."""
    validate_run_id(run_id)
    os.makedirs(RUNS_ROOT, exist_ok=True)
    run_dir = RUNS_ROOT / run_id
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


# ---------------------------------------------------------------------------
# Checksums / manifest
# ---------------------------------------------------------------------------

def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_entry(path: Path, run_dir: Path, artifact_type: str) -> dict:
    return {
        "type": artifact_type,
        "relative_path": str(path.relative_to(run_dir)),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_of(path),
    }


# ---------------------------------------------------------------------------
# Run summary lifecycle (atomic writes throughout)
# ---------------------------------------------------------------------------

def _summary_path(run_dir) -> Path:
    return Path(run_dir) / SUMMARY_FILENAME


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)  # atomic on POSIX and Windows


def create_run_summary(run_dir, run_id: str, started_at: str,
                        dataset_info: dict = None, embedding_model_info: dict = None) -> None:
    summary = {
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "dataset_info": dataset_info,
        "embedding_model_info": embedding_model_info,
        "models": {},
        "warnings": [],
    }
    _atomic_write_json(_summary_path(run_dir), summary)


def update_run_summary(run_dir, model_name: str, model_result: dict) -> None:
    """Atomic read-modify-write — safe to call once per trainer as each
    completes; the file is always valid JSON if read mid-run."""
    path = _summary_path(run_dir)
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    summary["models"][model_name] = model_result
    for w in model_result.get("warnings", []):
        summary["warnings"].append(f"{model_name}: {w}")
    _atomic_write_json(path, summary)


def finalize_run_summary(run_dir, finished_at: str, status: str) -> None:
    """status: 'completed' | 'completed_with_warnings' | 'failed'."""
    path = _summary_path(run_dir)
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    summary["finished_at"] = finished_at
    summary["status"] = status
    _atomic_write_json(path, summary)


# ---------------------------------------------------------------------------
# Plotting (single canonical implementation, replacing per-trainer duplicates
# for the 3 families touched in this slice only)
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_roc(fpr, tpr, auc, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_pr(recall, precision, ap, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Probability contract
# ---------------------------------------------------------------------------

def _normalize_proba(y_proba, class_order):
    """
    Normalizes y_proba to shape (n_samples, len(class_order)), columns
    ordered per class_order. Accepts:
      - (n_samples,)               binary positive-class probability only
                                    (e.g. a raw XGBoost Booster.predict()
                                    output for a binary objective) — only
                                    valid when class_order has length 2;
                                    class_order[1] is treated as positive.
      - (n_samples, n_classes)      standard predict_proba() output.
    """
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        if len(class_order) != 2:
            raise ValueError("1-D y_proba is only valid for a binary class_order")
        return np.column_stack([1 - y_proba, y_proba])
    if y_proba.ndim == 2 and y_proba.shape[1] == len(class_order):
        return y_proba
    raise ValueError(f"y_proba shape {y_proba.shape} incompatible with class_order length {len(class_order)}")


# ---------------------------------------------------------------------------
# Evaluation artifacts (confusion matrix / report / ROC / PR)
# ---------------------------------------------------------------------------

def save_evaluation_artifacts(
    run_dir,
    model_name: str,
    y_true_display,
    y_pred_display,
    display_labels: list,
    y_proba,
    proba_class_order: list,
    y_true_for_curves=None,
    positive_label=None,
) -> dict:
    """
    y_true_display/y_pred_display/display_labels drive the confusion matrix
    and classification report — whatever scale/labels are meaningful for a
    human to read (e.g. original domain IDs, or friendly names).

    y_proba/proba_class_order drive the ROC/PR curves and MUST be in the
    same encoding the model actually produced probabilities in (pass
    model.classes_.tolist(), never inferred/assumed). y_true_for_curves lets
    a caller supply true labels in that same encoding when it differs from
    the display encoding (e.g. an ordinal model fit on shifted labels);
    defaults to y_true_display when the two coincide.

    positive_label is required when proba_class_order has length 2 — the
    class explicitly treated as "positive" for the binary ROC/PR curve.

    Any class with no true instances (or only true instances, i.e. no
    negative examples) in the test split has its curve skipped with a
    recorded warning rather than a crash or a fabricated value.
    """
    run_dir = Path(run_dir)
    warnings = []
    manifest = []

    if y_true_for_curves is None:
        y_true_for_curves = y_true_display
    y_true_for_curves = np.asarray(y_true_for_curves)
    y_proba_norm = _normalize_proba(y_proba, proba_class_order)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true_display, y_pred_display, labels=display_labels)
    cm_path = run_dir / f"{model_name}_confusion_matrix.png"
    _plot_confusion_matrix(cm, display_labels, f"{model_name} Confusion Matrix", cm_path)
    manifest.append(_manifest_entry(cm_path, run_dir, "confusion_matrix"))

    # --- Classification report ---
    report_text = classification_report(y_true_display, y_pred_display, labels=display_labels, zero_division=0)
    report_path = run_dir / f"{model_name}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    manifest.append(_manifest_entry(report_path, run_dir, "classification_report"))

    # --- ROC / PR ---
    # Whether this is "binary" is a property of the model's intended design,
    # not an accident of how many classes happen to be present in a given
    # run's data — a 3-class ordinal model that only saw 2 classes this run
    # is still fundamentally multiclass, not binary. The caller signals
    # binary-ness explicitly by passing positive_label; its absence always
    # means one-vs-rest, even when only 2 classes are present.
    is_binary = positive_label is not None
    roc_results = {}

    if is_binary:
        if len(proba_class_order) != 2:
            raise ValueError(f"positive_label was given but proba_class_order has {len(proba_class_order)} classes, expected 2")
        pos_idx = proba_class_order.index(positive_label)
        y_true_binary = (y_true_for_curves == positive_label).astype(int)

        if len(set(y_true_binary.tolist())) < 2:
            warnings.append(f"positive class {positive_label!r} has no both-way examples in the test split; ROC/PR skipped")
        else:
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba_norm[:, pos_idx])
            auc = roc_auc_score(y_true_binary, y_proba_norm[:, pos_idx])
            precision, recall, _ = precision_recall_curve(y_true_binary, y_proba_norm[:, pos_idx])
            ap = average_precision_score(y_true_binary, y_proba_norm[:, pos_idx])

            roc_path = run_dir / f"{model_name}_roc_curve.png"
            _plot_roc(fpr, tpr, auc, f"{model_name} ROC Curve (positive={positive_label})", roc_path)
            manifest.append(_manifest_entry(roc_path, run_dir, "roc_curve"))

            pr_path = run_dir / f"{model_name}_pr_curve.png"
            _plot_pr(recall, precision, ap, f"{model_name} PR Curve (positive={positive_label})", pr_path)
            manifest.append(_manifest_entry(pr_path, run_dir, "pr_curve"))

            roc_results = {"positive_label": positive_label, "auc": float(auc), "average_precision": float(ap)}
    else:
        per_class_auc, per_class_ap, valid_aucs = {}, {}, []
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))

        for idx, cls in enumerate(proba_class_order):
            y_true_binary = (y_true_for_curves == cls).astype(int)
            if len(set(y_true_binary.tolist())) < 2:
                warnings.append(f"class {cls!r} skipped: no both-way true instances in the test split")
                continue
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba_norm[:, idx])
            auc = roc_auc_score(y_true_binary, y_proba_norm[:, idx])
            precision, recall, _ = precision_recall_curve(y_true_binary, y_proba_norm[:, idx])
            ap = average_precision_score(y_true_binary, y_proba_norm[:, idx])
            per_class_auc[str(cls)] = float(auc)
            per_class_ap[str(cls)] = float(ap)
            valid_aucs.append(auc)
            ax_roc.plot(fpr, tpr, label=f"class {cls} (AUC={auc:.3f})")
            ax_pr.plot(recall, precision, label=f"class {cls} (AP={ap:.3f})")

        valid_classes = [c for c in proba_class_order if str(c) in per_class_auc]
        if valid_classes:
            valid_idx = [proba_class_order.index(c) for c in valid_classes]
            y_true_bin_matrix = np.column_stack([(y_true_for_curves == c).astype(int) for c in valid_classes])
            micro_auc = float(roc_auc_score(y_true_bin_matrix, y_proba_norm[:, valid_idx], average="micro"))
            macro_auc = float(np.mean(valid_aucs))
        else:
            micro_auc = None
            macro_auc = None
            warnings.append("no class had both-way true instances; multiclass ROC/AUC skipped entirely")

        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"{model_name} ROC Curves (One-vs-Rest)")
        ax_roc.legend(fontsize=7)
        roc_path = run_dir / f"{model_name}_roc_curve.png"
        fig_roc.tight_layout()
        fig_roc.savefig(roc_path)
        plt.close(fig_roc)
        manifest.append(_manifest_entry(roc_path, run_dir, "roc_curve"))

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"{model_name} PR Curves (One-vs-Rest)")
        ax_pr.legend(fontsize=7)
        pr_path = run_dir / f"{model_name}_pr_curve.png"
        fig_pr.tight_layout()
        fig_pr.savefig(pr_path)
        plt.close(fig_pr)
        manifest.append(_manifest_entry(pr_path, run_dir, "pr_curve"))

        roc_results = {
            "per_class_auc": per_class_auc,
            "per_class_average_precision": per_class_ap,
            "micro_auc": micro_auc,
            "macro_auc": macro_auc,
            "skipped_classes": [c for c in proba_class_order if str(c) not in per_class_auc],
        }

    return {"roc_pr": roc_results, "warnings": warnings, "artifacts": manifest}


# ---------------------------------------------------------------------------
# Model serialization (kept separate from evaluation — each family uses its
# own correct mechanism, never forced through a single one)
# ---------------------------------------------------------------------------

def register_model_artifact(run_dir, model_name: str, model_obj, serializer: str) -> dict:
    """serializer: 'joblib' (sklearn/mord estimators) or 'xgboost_native'
    (XGBoost's own save_model(), matching what these trainers already do
    today for XGBoost — never forced through joblib)."""
    run_dir = Path(run_dir)
    if serializer == "joblib":
        import joblib
        path = run_dir / f"{model_name}.pkl"
        joblib.dump(model_obj, path)
    elif serializer == "xgboost_native":
        path = run_dir / f"{model_name}.json"
        model_obj.save_model(str(path))
    else:
        raise ValueError(f"Unknown serializer: {serializer!r}")
    return _manifest_entry(path, run_dir, "model_artifact")


# ---------------------------------------------------------------------------
# Explicit, deliberate promotion — NEVER called automatically by training.
# ---------------------------------------------------------------------------

_MAINTAINANCE_DIR = Path(__file__).resolve().parent
_CLASSIFICATION_MODELS_DIR = _MAINTAINANCE_DIR.parent

# Small, explicit registry (not auto-discovered) so promotion always targets
# a known, reviewed path. "domain" is deliberately absent: its live-model
# selection mechanism (which of the 3 saved candidates package_models.py
# actually loads) needs further investigation before an automated promotion
# path can be defined for it — see plan notes, not guessed at here.
_PROMOTABLE_FAMILIES = {
    "harm_binary": {
        "run_artifact": "harm_binary.pkl",
        "live_path": _CLASSIFICATION_MODELS_DIR / "Harm_level" / "harm_binary_model.pkl",
    },
    "feedback_type": {
        "run_artifact": "feedback_type.pkl",
        "live_path": _CLASSIFICATION_MODELS_DIR / "feedback_type" / "FeedbackType_OrdinalModel.pkl",
    },
}


def promote_run(run_id: str, family_names: list) -> dict:
    """
    Explicit, deliberate action: copies a specific run's model artifact(s)
    onto the fixed live-inference path(s) for the named families. Nothing
    in this module or in train_all.py calls this automatically — per
    ML_ARCHITECTURE_DECISION_RECORD.md, a training run must never silently
    swap out what live inference uses.
    """
    validate_run_id(run_id)
    run_dir = RUNS_ROOT / run_id
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")

    results = {}
    for family in family_names:
        if family == "domain":
            raise NotImplementedError(
                "Promotion for 'domain' is not implemented in Slice A: its live-model "
                "selection mechanism (which of the 3 saved candidates package_models.py "
                "actually loads) needs investigation before an automated promotion path "
                "can be defined."
            )
        config = _PROMOTABLE_FAMILIES.get(family)
        if config is None:
            raise ValueError(f"Family {family!r} is not registered for promotion")
        src = run_dir / config["run_artifact"]
        if not src.exists():
            raise FileNotFoundError(f"Run {run_id} has no artifact for family {family!r} at {src}")
        shutil.copy2(src, config["live_path"])
        results[family] = {"source": str(src), "promoted_to": str(config["live_path"])}
    return results
