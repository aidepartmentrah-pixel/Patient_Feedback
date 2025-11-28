import datetime
import json
import os

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
from models_directory.Classification_Models.Harm_level.train_harm_ordinal_high import train_harm_ordinal_high
from models_directory.Classification_Models.Harm_level.train_harm_ordinal_low import train_harm_ordinal_low
from models_directory.Classification_Models.Severity_level.train_severity_model import train_severity_model


def save_training_report(all_metrics: dict, save_path: str):
    """Save all training metrics to a TXT report."""

    today = datetime.datetime.now().strftime("%d_%m_%Y")
    filename = f"classification_training_report_{today}.txt"
    full_path = os.path.join(save_path, filename)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write("=== CLASSIFICATION TRAINING PERFORMANCE REPORT ===\n")
        f.write(f"Generated on: {today}\n")
        f.write("=================================================\n\n")

        for model_name, metrics in all_metrics.items():

            num_records = metrics.get("num_records", "N/A")

            f.write(f"--- {model_name} ---\n")
            f.write(f"Training Records: {num_records}\n")

            f.write(f"Accuracy : {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall   : {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-score : {metrics.get('f1', 0):.4f}\n")
            f.write(f"mAP      : {metrics.get('mAP', 0):.4f}\n")

            f.write("\n---------------------------------------------\n\n")

    print(f"\n📄 Training report saved: {full_path}\n")



def train_all():
    """Runs ALL training steps and generates a unified training report."""

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    all_metrics = {}
    all_models = {}

    def run_training(model_name, func):
        model, metrics = func()
        print(json.dumps(metrics, indent=4))

        all_models[model_name] = model
        all_metrics[model_name] = metrics

    # CATEGORY (Inside each domain)
    run_training("Category_Domain1", train_category_domain1)
    run_training("Category_Domain2", train_category_domain2)
    run_training("Category_Domain3", train_category_domain3)

    # DOMAIN MODEL
    run_training("Domain_Model", train_domain_models)

    # SUBCATEGORIES
    run_training("Subcategory_Cat1", train_subcategory_cat1)
    run_training("Subcategory_Cat2", train_subcategory_cat2)
    run_training("Subcategory_Cat3", train_subcategory_cat3)
    run_training("Subcategory_Cat4", train_subcategory_cat4)
    run_training("Subcategory_Cat5", train_subcategory_cat5)
    run_training("Subcategory_Cat6", train_subcategory_cat6)
    run_training("Subcategory_Cat7", train_subcategory_cat7)

    # HARM LEVEL MODELS
    run_training("Harm_Binary", train_harm_binary)
    run_training("Harm_Ordinal_High", train_harm_ordinal_high)
    run_training("Harm_Ordinal_Low", train_harm_ordinal_low)

    # SEVERITY
    run_training("Severity_Model", train_severity_model)

    # OPTIONAL STAGE
    # run_training("Stage_Model", train_stage)

    # SAVE REPORT
    save_training_report(all_metrics, SCRIPT_DIR)

    return all_models, all_metrics




if __name__ == "__main__":
    train_all()