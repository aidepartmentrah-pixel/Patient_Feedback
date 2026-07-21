import datetime
import json
import os
import sys
import traceback

# Add workspace root to Python path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from models_directory.Classification_Models.Maintainance import run_versioning

# All 18 model families now accept a run_dir kwarg and write versioned
# evaluation/model artifacts (Stage 9 Slice A: Domain/Harm_Binary/
# Feedback_Type; Slice B: the remaining 15).
_VERSIONED_MODEL_NAMES = {
    "Domain_Model",
    "Category_Domain1", "Category_Domain2", "Category_Domain3",
    "Subcategory_Cat1", "Subcategory_Cat2", "Subcategory_Cat3",
    "Subcategory_Cat4", "Subcategory_Cat5", "Subcategory_Cat6", "Subcategory_Cat7",
    "Harm_Binary", "Harm_Ordinal_High", "Harm_Ordinal_Low",
    "Severity_Model",
    "Feedback_Type", "Improvement_Opportunity_Type", "Classification_En",
}

#Statistically Acceptable - Needs Refactoring
from models_directory.Classification_Models.Hierarchical_Classification_Model.domain.train_domain_model import train_domain_models
from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_1.train_category_domain1 import train_category_domain1
from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_2.train_category_domain2 import train_category_domain2
from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_3.train_category_domain3 import train_category_domain3
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_1.train_subcategory_category1 import train_subcategory_cat1
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_2.train_subcategory_category2 import train_subcategory_cat2
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_3.train_subcategory_category3 import train_subcategory_cat3
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_4.train_subcategory_category4 import train_subcategory_cat4
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_5.train_subcategory_category5 import train_subcategory_cat5
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_6.train_subcategory_category6 import train_subcategory_cat6
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_7.train_subcategory_category7 import train_subcategory_cat7
from models_directory.Classification_Models.Harm_level.train_harm_binary import train_harm_binary

#Not Added Yet
from models_directory.Classification_Models.feedback_type.train_feedback_type_model import train_feedback_type_model
from models_directory.Classification_Models.improvement_opportunity_type.train_improvement_model import train_improvement_model
from models_directory.Classification_Models.Classification_En.train_classification_En_model import train_classification_en_model

#Not Acceptable - Needs Adjustments
from models_directory.Classification_Models.Harm_level.train_harm_ordinal_high import train_harm_ordinal_high
from models_directory.Classification_Models.Harm_level.train_harm_ordinal_low import train_harm_ordinal_low

from models_directory.Classification_Models.Severity_level.train_severity_model import train_severity_model


def save_training_report(all_metrics: dict, save_path: str):
    """
    Legacy, best-effort report — keyed only by date, so a same-day rerun
    overwrites it. Kept unchanged for backward compatibility. From Stage 9
    Slice A onward, the versioned run folder + run_summary.json (see
    run_versioning.py) is the authoritative, collision-free run record.
    """

    today = datetime.datetime.now().strftime("%d_%m_%Y")
    filename = f"classification_training_report_{today}.txt"
    full_path = os.path.join(save_path, filename)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  CLASSIFICATION TRAINING PERFORMANCE REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {today}\n")
        f.write("=" * 70 + "\n\n")

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

        f.write("=" * 70 + "\n")
        f.write("Summary Statistics\n")
        f.write("=" * 70 + "\n")
        
        total_models = len(all_metrics)
        avg_f1 = sum(m.get("f1", 0) for m in all_metrics.values()) / max(total_models, 1)
        
        f.write(f"Total Models Trained: {total_models}\n")
        f.write(f"Average F1-Score: {avg_f1:.6f}\n")

    print(f"\n Training report saved: {full_path}\n")


def _load_dataset_info() -> dict:
    """Reads models_directory/split_metadata.json (written by
    split_data_from_sql_server(), see Stage 8) for the run summary's
    dataset_info — reused rather than recomputed."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "split_metadata.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not load split_metadata.json: {e}")
        return None


def _load_embedding_model_info() -> dict:
    """ml_model_version_db.get_active_model_version() — the same DB source
    Stage 6's worker registers against, so the run summary always reflects
    the model that actually produced the embeddings being trained on."""
    try:
        from backend.core.database import get_connection
        from backend.api.db_layer import ml_model_version_db
        conn = get_connection()
        try:
            cursor = conn.cursor()
            return ml_model_version_db.get_active_model_version(cursor)
        finally:
            conn.close()
    except Exception as e:
        print(f"[WARNING] Could not load active embedding model version: {e}")
        return None


def train_all(run_id: str = None):
    """Runs ALL training steps, writes versioned run artifacts for the
    Stage 9 Slice A trainers (Domain, Harm_Binary, Feedback_Type), and
    generates the legacy unified training report."""

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    if run_id is None:
        run_id = run_versioning.generate_run_id()
    run_dir = run_versioning.get_run_dir(run_id)
    started_at = datetime.datetime.now().isoformat()

    run_versioning.create_run_summary(
        run_dir, run_id, started_at,
        dataset_info=_load_dataset_info(),
        embedding_model_info=_load_embedding_model_info(),
    )
    print(f"\n[RUN {run_id}] Versioned artifacts will be written to: {run_dir}\n")

    all_metrics = {}
    all_models = {}
    run_had_warnings = {"value": False}

    # Track step number for progress reporting
    current_step = 0

    def run_training(model_name, func):
        """Train a model and collect standardized metrics."""
        nonlocal current_step
        current_step += 1

        # Report progress to training service
        try:
            from backend.api.services.training_service import update_training_progress
            update_training_progress(model_name, current_step)
        except Exception as e:
            print(f"[WARNING] Could not update progress: {e}")

        print(f"\n{'='*70}")
        print(f"Training: {model_name} (Step {current_step}/18)")
        print(f"{'='*70}")

        try:
            if model_name in _VERSIONED_MODEL_NAMES:
                model, metrics = func(run_dir=run_dir)
            else:
                model, metrics = func()
        except Exception as e:
            # Some trainers (e.g. feedback_type) re-raise on failure instead
            # of returning (None, None) — normalize both contracts here so
            # one failing trainer never aborts the rest of the run.
            print(f"\n[ERROR] {model_name} raised an exception: {e}")
            traceback.print_exc()
            model, metrics = None, None

        # Handle failed training (returns None, None)
        if metrics is None:
            print(f"\n[WARNING] {model_name} FAILED - returned None metrics")
            run_had_warnings["value"] = True
            metrics = {
                "model_name": model_name,
                "num_records": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "error": "Training returned None"
            }
        elif metrics.get("warnings"):
            run_had_warnings["value"] = True

        print(f"\n {model_name} Complete:")
        print(f"  Records: {metrics.get('num_records', 0)}")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.6f}")
        print(f"  Precision: {metrics.get('precision', 0):.6f}")
        print(f"  Recall:    {metrics.get('recall', 0):.6f}")
        print(f"  F1-Score:  {metrics.get('f1', 0):.6f}")

        all_models[model_name] = model
        all_metrics[model_name] = metrics
        run_versioning.update_run_summary(run_dir, model_name, metrics)

    try:
        # DOMAIN MODEL
        print("\n" + "="*70)
        print("DOMAIN LEVEL")
        print("="*70)
        run_training("Domain_Model", train_domain_models)

        # CATEGORY (Inside each domain)
        print("\n" + "="*70)
        print("CATEGORY LEVEL")
        print("="*70)
        run_training("Category_Domain1", train_category_domain1)
        run_training("Category_Domain2", train_category_domain2)
        run_training("Category_Domain3", train_category_domain3)

        # SUBCATEGORIES
        print("\n" + "="*70)
        print("SUBCATEGORY LEVEL")
        print("="*70)
        run_training("Subcategory_Cat1", train_subcategory_cat1)
        run_training("Subcategory_Cat2", train_subcategory_cat2)
        run_training("Subcategory_Cat3", train_subcategory_cat3)
        run_training("Subcategory_Cat4", train_subcategory_cat4)
        run_training("Subcategory_Cat5", train_subcategory_cat5)
        run_training("Subcategory_Cat6", train_subcategory_cat6)
        run_training("Subcategory_Cat7", train_subcategory_cat7)

        # HARM LEVEL MODELS
        print("\n" + "="*70)
        print("HARM LEVEL")
        print("="*70)
        run_training("Harm_Binary", train_harm_binary)
        run_training("Harm_Ordinal_High", train_harm_ordinal_high)
        run_training("Harm_Ordinal_Low", train_harm_ordinal_low)

        # SEVERITY
        print("\n" + "="*70)
        print("SEVERITY LEVEL")
        print("="*70)
        run_training("Severity_Model", train_severity_model)

        # ADDITIONAL CLASSIFICATION MODELS
        print("\n" + "="*70)
        print("ADDITIONAL CLASSIFICATION MODELS")
        print("="*70)
        run_training("Feedback_Type", train_feedback_type_model)
        run_training("Improvement_Opportunity_Type", train_improvement_model)
        run_training("Classification_En", train_classification_en_model)

        # SAVE REPORT (legacy, date-keyed — see save_training_report docstring)
        print("\n" + "="*70)
        print("GENERATING REPORT")
        print("="*70)
        REPORT_DIR = os.path.join(SCRIPT_DIR, "Performance_Reporting")
        save_training_report(all_metrics, REPORT_DIR)

        print("\n" + "="*70)
        print(" TRAINING COMPLETE")
        print("="*70)
        print(f"Total models trained: {len(all_metrics)}")

        avg_f1 = sum(m.get("f1", 0) for m in all_metrics.values()) / max(len(all_metrics), 1)
        print(f"Average F1-Score: {avg_f1:.6f}\n")

        # Convert metrics dict to list format expected by API
        models_list = []
        for model_name, metrics in all_metrics.items():
            models_list.append({
                "model_name": model_name,
                "num_records": metrics.get("num_records", 0),
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0)
            })

        status = "completed_with_warnings" if run_had_warnings["value"] else "completed"
        run_versioning.finalize_run_summary(run_dir, datetime.datetime.now().isoformat(), status)
        print(f"[RUN {run_id}] Finalized with status={status}. Artifacts: {run_dir}")

        return {
            "run_id": run_id,
            "models": models_list,
            "summary": {
                "total_models": len(all_metrics),
                "average_f1": avg_f1
            }
        }
    except Exception:
        run_versioning.finalize_run_summary(run_dir, datetime.datetime.now().isoformat(), "failed")
        raise

if __name__ == "__main__":
    train_all()