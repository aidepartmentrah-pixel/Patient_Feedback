import datetime
import json
import os

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

#Not Acceptable - Needs Adjustments
from models_directory.Classification_Models.Harm_level.train_harm_ordinal_high import train_harm_ordinal_high
from models_directory.Classification_Models.Harm_level.train_harm_ordinal_low import train_harm_ordinal_low

from models_directory.Classification_Models.Severity_level.train_severity_model import train_severity_model


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

    print(f"\n📄 Training report saved: {full_path}\n")



def train_all():
    """Runs ALL training steps and generates a unified training report."""

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    all_metrics = {}
    all_models = {}

    def run_training(model_name, func):
        """Train a model and collect standardized metrics."""
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"{'='*70}")
        model, metrics = func()
        
        print(f"\n✔ {model_name} Complete:")
        print(f"  Records: {metrics.get('num_records', 0)}")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.6f}")
        print(f"  Precision: {metrics.get('precision', 0):.6f}")
        print(f"  Recall:    {metrics.get('recall', 0):.6f}")
        print(f"  F1-Score:  {metrics.get('f1', 0):.6f}")

        all_models[model_name] = model
        all_metrics[model_name] = metrics

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

    # SAVE REPORT
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)
    REPORT_DIR = os.path.join(SCRIPT_DIR, "Performance_Reporting")
    save_training_report(all_metrics, REPORT_DIR)

    print("\n" + "="*70)
    print("✔ TRAINING COMPLETE")
    print("="*70)
    print(f"Total models trained: {len(all_metrics)}")
    
    avg_f1 = sum(m.get("f1", 0) for m in all_metrics.values()) / max(len(all_metrics), 1)
    print(f"Average F1-Score: {avg_f1:.6f}\n")

    return all_models, all_metrics



if __name__ == "__main__":
    train_all()