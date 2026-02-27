import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python3
"""
Unified Hierarchical Prediction:
Domain → Category → Subcategory
"""

from models_directory.Classification_Models.Stage.modular_functions import get_embedding
from collections import Counter
import numpy as np
import json

# ==========================================
# IMPORT ALL MODEL PREDICTORS
# ==========================================

# ---- DOMAIN ----
from models_directory.Classification_Models.Hierarchical_Classification_Model.domain.predict_domain import predict_from_text as predict_domain_text
from models_directory.Classification_Models.Hierarchical_Classification_Model.domain.predict_domain import predict_domain_from_embedding as predict_domain_embedding

# ---- CATEGORY ----
from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_1.predict_category_domain1 import predict_from_text as predict_category_text_domain1
from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_2.predict_category_domain2 import predict_from_text as predict_category_text_domain2
from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_3.predict_category_domain3 import predict_from_text as predict_category_text_domain3

from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_1.predict_category_domain1 import predict_from_embedding as predict_category_embedding_domain1
from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_2.predict_category_domain2 import predict_from_embedding as predict_category_embedding_domain2
from models_directory.Classification_Models.Hierarchical_Classification_Model.category.domain_3.predict_category_domain3 import predict_from_embedding as predict_category_embedding_domain3

# ---- SUBCATEGORY ----
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_1.predict_subcategory_category1 import predict_from_text as predict_subcategory_text_category1
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_2.predict_subcategory_category2 import predict_from_text as predict_subcategory_text_category2
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_3.predict_subcategory_category3 import predict_from_text as predict_subcategory_text_category3
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_4.predict_subcategory_category4 import predict_from_text as predict_subcategory_text_category4
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_5.predict_subcategory_category5 import predict_from_text as predict_subcategory_text_category5
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_6.predict_subcategory_category6 import predict_from_text as predict_subcategory_text_category6
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_7.predict_subcategory_category7 import predict_from_text as predict_subcategory_text_category7

from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_1.predict_subcategory_category1 import predict_from_embedding as predict_subcategory_embedding_category1
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_2.predict_subcategory_category2 import predict_from_embedding as predict_subcategory_embedding_category2
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_3.predict_subcategory_category3 import predict_from_embedding as predict_subcategory_embedding_category3
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_4.predict_subcategory_category4 import predict_from_embedding as predict_subcategory_embedding_category4
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_5.predict_subcategory_category5 import predict_from_embedding as predict_subcategory_embedding_category5
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_6.predict_subcategory_category6 import predict_from_embedding as predict_subcategory_embedding_category6
from models_directory.Classification_Models.Hierarchical_Classification_Model.sub_category.category_7.predict_subcategory_category7 import predict_from_embedding as predict_subcategory_embedding_category7


# ==========================================
# MAPPINGS
# ==========================================

# Domains → list of categories
DOMAIN_TO_CATEGORIES = {
    1: [6, 7],
    2: [4, 5],
    3: [1, 2, 3]
}

# Categories → list of subcategories
CATEGORY_TO_SUBCATEGORIES = {
    1: [1, 2, 3, 4],
    2: [5, 6],
    3: [7, 8],
    4: [9, 10, 11, 12],
    5: [13, 14, 15, 16, 17, 18],
    6: [19, 20, 21],
    7: [22, 24, 25, 26, 27]
}

# CATEGORY PREDICTORS
CATEGORY_PREDICTORS_TEXT = {
    1: predict_category_text_domain1,
    2: predict_category_text_domain2,
    3: predict_category_text_domain3
}

CATEGORY_PREDICTORS_EMBEDDING = {
    1: predict_category_embedding_domain1,
    2: predict_category_embedding_domain2,
    3: predict_category_embedding_domain3
}

# SUBCATEGORY PREDICTORS
SUBCATEGORY_PREDICTORS_TEXT = {
    1: predict_subcategory_text_category1,
    2: predict_subcategory_text_category2,
    3: predict_subcategory_text_category3,
    4: predict_subcategory_text_category4,
    5: predict_subcategory_text_category5,
    6: predict_subcategory_text_category6,
    7: predict_subcategory_text_category7
}

# SUBCATEGORY PREDICTORS
SUBCATEGORY_PREDICTORS_EMBEDDING = {
    1: predict_subcategory_embedding_category1,
    2: predict_subcategory_embedding_category2,
    3: predict_subcategory_embedding_category3,
    4: predict_subcategory_embedding_category4,
    5: predict_subcategory_embedding_category5,
    6: predict_subcategory_embedding_category6,
    7: predict_subcategory_embedding_category7
}

# ==========================================
# HIERARCHICAL PREDICT FUNCTION
# ==========================================

def hierarchical_predict_text(text: str):
    output = {}

    # -------------------------
    # 1) DOMAIN
    # -------------------------
    domain_preds = predict_domain_text(text)
    domain_final = domain_preds["xgboost"]  # choose XGBoost as final
    output["domain"] = domain_final

    # -------------------------
    # 2) CATEGORY (depends on domain)
    # -------------------------
    category_predictor = CATEGORY_PREDICTORS_TEXT.get(domain_final)
    if not category_predictor:
        raise ValueError(f"No category predictor for domain={domain_final}")

    category_preds = category_predictor(text)

    # Majority voting for category
    votes_category = [
        category_preds["logistic_regression"],
        category_preds["random_forest"],
        category_preds["xgboost"]
    ]
    category_final = Counter(votes_category).most_common(1)[0][0]

    # Validate category
    if category_final not in DOMAIN_TO_CATEGORIES[domain_final]:
        category_final = DOMAIN_TO_CATEGORIES[domain_final][0]

    output["category"] = category_final

    # -------------------------
    # 3) SUBCATEGORY (depends on category)
    # -------------------------
    subcategory_predictor = SUBCATEGORY_PREDICTORS_TEXT.get(category_final)
    if not subcategory_predictor:
        # fallback to first allowed subcategory
        subcategory_final = CATEGORY_TO_SUBCATEGORIES[category_final][0]
    else:
        subcategory_preds = subcategory_predictor(text)

        votes_subcategory = [
            subcategory_preds["logistic_regression"],
            subcategory_preds["random_forest"],
            subcategory_preds["xgboost"]
        ]
        subcategory_final = Counter(votes_subcategory).most_common(1)[0][0]

        # Validate subcategory
        if subcategory_final not in CATEGORY_TO_SUBCATEGORIES[category_final]:
            subcategory_final = CATEGORY_TO_SUBCATEGORIES[category_final][0]

    output["subcategory"] = subcategory_final

    return output

def hierarchical_predict_embeddings(embedding: np.ndarray):
    output = {}

    print(f"\n[HIER_PRED] === HIERARCHICAL PREDICTION START ===")
    print(f"[HIER_PRED] Embedding shape: {embedding.shape}")

    # -------------------------
    # 1) DOMAIN
    # -------------------------
    print(f"[HIER_PRED] Step 1: Predicting DOMAIN...")
    domain_preds = predict_domain_embedding(embedding)
    print(f"[HIER_PRED] Domain predictions: {domain_preds}")
    domain_final = domain_preds["xgboost"]  # choose XGBoost as final
    print(f"[HIER_PRED] Domain final (xgboost): {domain_final}")
    output["domain"] = domain_final

    # -------------------------
    # 2) CATEGORY (depends on domain)
    # -------------------------
    print(f"[HIER_PRED] Step 2: Predicting CATEGORY for domain={domain_final}...")
    category_predictor = CATEGORY_PREDICTORS_EMBEDDING.get(domain_final)
    if not category_predictor:
        raise ValueError(f"No category embedding predictor for domain={domain_final}")

    print(f"[HIER_PRED] Calling category predictor for domain {domain_final}...")
    category_preds = category_predictor(embedding)
    print(f"[HIER_PRED] Category predictions: {category_preds}")

    # Majority voting for category
    votes_category = [
        category_preds["logistic_regression"],
        category_preds["random_forest"],
        category_preds["xgboost"]
    ]
    print(f"[HIER_PRED] Category votes: {votes_category}")
    category_final = Counter(votes_category).most_common(1)[0][0]
    print(f"[HIER_PRED] Category majority vote: {category_final}")

    # Validate category
    valid_cats = DOMAIN_TO_CATEGORIES.get(domain_final, [])
    print(f"[HIER_PRED] Valid categories for domain {domain_final}: {valid_cats}")
    if category_final not in valid_cats:
        print(f"[HIER_PRED] WARNING: Category {category_final} not in valid list, using fallback")
        category_final = DOMAIN_TO_CATEGORIES[domain_final][0]
        print(f"[HIER_PRED] Category after fallback: {category_final}")

    output["category"] = category_final

    # -------------------------
    # 3) SUBCATEGORY (depends on category)
    # -------------------------
    print(f"[HIER_PRED] Step 3: Predicting SUBCATEGORY for category={category_final}...")
    subcategory_predictor = SUBCATEGORY_PREDICTORS_EMBEDDING.get(category_final)

    if not subcategory_predictor:
        # fallback to first valid subcategory
        print(f"[HIER_PRED] WARNING: No subcategory predictor for category={category_final}, using fallback")
        subcategory_final = CATEGORY_TO_SUBCATEGORIES[category_final][0]
    else:
        print(f"[HIER_PRED] Calling subcategory predictor for category {category_final}...")
        subcategory_preds = subcategory_predictor(embedding)
        print(f"[HIER_PRED] Subcategory predictions: {subcategory_preds}")

        # Majority voting for subcategory
        votes_subcategory = [
            subcategory_preds["logistic_regression"],
            subcategory_preds["random_forest"],
            subcategory_preds["xgboost"]
        ]
        print(f"[HIER_PRED] Subcategory votes: {votes_subcategory}")
        subcategory_final = Counter(votes_subcategory).most_common(1)[0][0]
        print(f"[HIER_PRED] Subcategory majority vote: {subcategory_final}")

        # Validate
        valid_subcats = CATEGORY_TO_SUBCATEGORIES.get(category_final, [])
        if subcategory_final not in valid_subcats:
            print(f"[HIER_PRED] WARNING: Subcategory {subcategory_final} not in valid list {valid_subcats}, using fallback")
            subcategory_final = CATEGORY_TO_SUBCATEGORIES[category_final][0]

    output["subcategory"] = subcategory_final
    print(f"[HIER_PRED] === FINAL OUTPUT: {output} ===")

    return output



# ==========================================
# TEST EXAMPLE
# ==========================================

if __name__ == "__main__":
    print("Hierarchical predictor")
    example_text = "The nurse did not respond quickly and follow-up was poor."
    raw = get_embedding(example_text)
    embedding = np.frombuffer(raw, dtype=np.float32)
    result_text = hierarchical_predict_text(text=example_text)
    result_embedding = hierarchical_predict_embeddings(embedding)
    print(json.dumps(result_embedding, indent=4))
    print(json.dumps(result_text, indent=4))



