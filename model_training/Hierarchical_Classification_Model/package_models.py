"""
package_models.py

Loads domain-category-subcategory hierarchical models.
Provides:
    predict_text(text)
    predict_embedding(embedding)
"""

import os
import joblib
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

# ============================================================
# CONFIG
# ============================================================

MODEL_ROOT = Path(__file__).resolve().parent
MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Dynamic model selection based on your description
CATEGORY_ALGO = {
    0: "logreg",
    1: "rf",
    2: "logreg",
}

SUBCATEGORY_ALGO = {
    0: "xgb",
    1: "logreg",
    2: "logreg",
    3: "logreg",
    4: "logreg",
    5: "logreg",
    6: "logreg",
}

CATEGORY_SUBCAT_ALG = {
    0: "xgb",
    1: "logreg",
    2: "logreg",
    3: "logreg",
    4: "logreg",
    5: "logreg",
    6: "logreg"
}

DOMAIN_CATEGORY_ALG = CATEGORY_ALGO


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading tokenizer + MPNET model...")
tokenizer = AutoTokenizer.from_pretrained(MPNET)
embed_model = AutoModel.from_pretrained(MPNET)
embed_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model.to(device)
print(f"Device = {device}")


# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================

def get_embedding(text: str):
    """Generate embedding for a single text."""
    if not text or not text.strip():
        raise ValueError("Empty text provided.")

    enc = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = embed_model(**enc)

    emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
    return emb  # (1,768)


# ============================================================
# MODEL LOADER — MATCHING YOUR FILE NAMES EXACTLY
# ============================================================

def load_node_model(level: str, node_id: str, alg: str):
    """
    level: 'domain', 'category', 'sub_category'
    node_id:
        domain      -> folder: models/domain/
        domain_0    -> folder: models/category/domain_0/
        cat_0       -> folder: models/sub_category/cat_0/

    alg: 'logreg', 'rf', 'xgb', ...
    """
    level_dir = MODEL_ROOT / level

    # For nested domain/category
    node_dir = level_dir / node_id

    # Domain has no nested subfolder → fix:
    if level == "domain":
        node_dir = level_dir  # use models/domain/

    # Determine correct file prefix
    if level == "domain":
        prefix = "domain"
    elif level == "category":
        # node_id = "domain_1" → extract "1"
        dom_val = node_id.split("_")[1]
        prefix = f"category_d{dom_val}"
    elif level == "sub_category":
        # node_id = "cat_2" → extract "2"
        cat_val = node_id.split("_")[1]
        prefix = f"subcat_c{cat_val}"
    else:
        raise ValueError(f"Invalid level: {level}")

    model_path = node_dir / f"{prefix}_{alg}.pkl"
    le_path = node_dir / f"{prefix}_label_encoder.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    if not le_path.exists():
        raise FileNotFoundError(f"Missing label encoder: {le_path}")

    model = joblib.load(model_path)
    le = joblib.load(le_path)
    return model, le


# ============================================================
# MAIN CASCADE PREDICTOR
# ============================================================

def _run_cascade(emb):
    # 1. DOMAIN
    domain_model, le_domain = load_node_model("domain", "domain", "logreg")
    dom_pred = domain_model.predict(emb)[0]
    dom_label = le_domain.inverse_transform([dom_pred])[0]

    # 2. CATEGORY
    cat_alg = DOMAIN_CATEGORY_ALG[dom_pred]  # your mapping dict
    cat_node = f"domain_{dom_pred}"
    cat_model, le_cat = load_node_model("category", cat_node, cat_alg)
    cat_pred = cat_model.predict(emb)[0]
    cat_label = le_cat.inverse_transform([cat_pred])[0]

    # 3. SUBCATEGORY
    sub_alg = CATEGORY_SUBCAT_ALG[cat_pred]  # your mapping dict
    sub_node = f"cat_{cat_pred}"
    sub_model, le_sub = load_node_model("sub_category", sub_node, sub_alg)
    sub_pred = sub_model.predict(emb)[0]
    sub_label = le_sub.inverse_transform([sub_pred])[0]

    return {
        "domain": dom_label,
        "category": cat_label,
        "sub_category": sub_label
    }



# ============================================================
# PUBLIC PREDICT FUNCTIONS
# ============================================================

def predict_text(text: str):
    """Predict domain/category/subcategory from raw Arabic text."""
    emb = get_embedding(text)
    return _run_cascade(emb)


def predict_embedding(emb: np.ndarray):
    """Predict from a pre-generated embedding (shape 768)."""
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return _run_cascade(emb)


# ============================================================
# MANUAL TESTING
# ============================================================

if __name__ == "__main__":
    sample = ("اعترض والد المريض مهدي خليل حرقوص أن إبنه يتعالج في مستشفانا منذ حوالي 8 سنوات وخلال فترة الحرب تم نقله إلى مستشفى الحريري وكان لدى المرافق ملاحظات حول ما حصل معه هناك أنه خلال الحرب تم نقل  إبنه المريض إلى م. الحريري(وكان يتم متابعته من قبل فريق مستشفى الرسول الأعظم (ص)) وكان من المفترض وضعه في غرفة منفردة, وتم وضعه بجانب مريض لديه salmonella ما أدى إلى إرتفاع حرارة المريض وبقي يأخذ دواء إلتهابات لفترة  ذكر أنه كان يريد إجراء suction  للمريض فطلب من التمريض إحضار tubes لسحب البلغ والدم للمربض فأخبره الأخ محمود فرحات أنه لا يوجد tubes وأن بإمكانه إستعمال الtube عدة مرات ما أثار إستياء المرافق, فذهب واشترى tubes من الشركة وقام بالإجراء لإبنه   ذكر المرافق ان الbottle suction  بقيت نفسها خلال فترة تواجد المريض في مستشفى الحريري ولم يتم تغييرها وذلك بسبب عدم القدرة على الذهاب إلى مستشفانا بحسب ما تم إخبار المرافق "
        )
    print("\nSample:", sample)

    try:
        out = predict_text(sample)
        print("\nResult:")
        print(out)
    except Exception as e:
        print("ERROR:", e)
