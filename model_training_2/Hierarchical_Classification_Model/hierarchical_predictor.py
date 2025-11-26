import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python3
"""
Unified Hierarchical Prediction:
Domain → Category → Subcategory
"""
from model_training_2.Stage.modular_functions import get_embedding
from collections import Counter
import numpy as np
import json

# ==========================================
# IMPORT ALL MODEL PREDICTORS
# ==========================================

# ---- DOMAIN ----
from model_training_2.Hierarchical_Classification_Model.domain.predict_domain import predict_from_text as predict_domain_text
from model_training_2.Hierarchical_Classification_Model.domain.predict_domain import predict_domain_from_embedding as predict_domain_embedding

# ---- CATEGORY ----
from model_training_2.Hierarchical_Classification_Model.category.domain_1.predict_category_domain1 import predict_from_text as predict_category_text_domain1
from model_training_2.Hierarchical_Classification_Model.category.domain_2.predict_category_domain2 import predict_from_text as predict_category_text_domain2
from model_training_2.Hierarchical_Classification_Model.category.domain_3.predict_category_domain3 import predict_from_text as predict_category_text_domain3

from model_training_2.Hierarchical_Classification_Model.category.domain_1.predict_category_domain1 import predict_from_embedding as predict_category_embedding_domain1
from model_training_2.Hierarchical_Classification_Model.category.domain_2.predict_category_domain2 import predict_from_embedding as predict_category_embedding_domain2
from model_training_2.Hierarchical_Classification_Model.category.domain_3.predict_category_domain3 import predict_from_embedding as predict_category_embedding_domain3

# ---- SUBCATEGORY ----
from model_training_2.Hierarchical_Classification_Model.sub_category.category_1.predict_subcategory_category1 import predict_from_text as predict_subcategory_text_category1
from model_training_2.Hierarchical_Classification_Model.sub_category.category_2.predict_subcategory_category2 import predict_from_text as predict_subcategory_text_category2
from model_training_2.Hierarchical_Classification_Model.sub_category.category_3.predict_subcategory_category3 import predict_from_text as predict_subcategory_text_category3
from model_training_2.Hierarchical_Classification_Model.sub_category.category_4.predict_subcategory_category4 import predict_from_text as predict_subcategory_text_category4
from model_training_2.Hierarchical_Classification_Model.sub_category.category_5.predict_subcategory_category5 import predict_from_text as predict_subcategory_text_category5
from model_training_2.Hierarchical_Classification_Model.sub_category.category_6.predict_subcategory_category6 import predict_from_text as predict_subcategory_text_category6
from model_training_2.Hierarchical_Classification_Model.sub_category.category_7.predict_subcategory_category7 import predict_from_text as predict_subcategory_text_category7

from model_training_2.Hierarchical_Classification_Model.sub_category.category_1.predict_subcategory_category1 import predict_from_embedding as predict_subcategory_embedding_category1
from model_training_2.Hierarchical_Classification_Model.sub_category.category_2.predict_subcategory_category2 import predict_from_embedding as predict_subcategory_embedding_category2
from model_training_2.Hierarchical_Classification_Model.sub_category.category_3.predict_subcategory_category3 import predict_from_embedding as predict_subcategory_embedding_category3
from model_training_2.Hierarchical_Classification_Model.sub_category.category_4.predict_subcategory_category4 import predict_from_embedding as predict_subcategory_embedding_category4
from model_training_2.Hierarchical_Classification_Model.sub_category.category_5.predict_subcategory_category5 import predict_from_embedding as predict_subcategory_embedding_category5
from model_training_2.Hierarchical_Classification_Model.sub_category.category_6.predict_subcategory_category6 import predict_from_embedding as predict_subcategory_embedding_category6
from model_training_2.Hierarchical_Classification_Model.sub_category.category_7.predict_subcategory_category7 import predict_from_embedding as predict_subcategory_embedding_category7


# ==========================================
# MAPPINGS
# ==========================================

# Domains → list of categories
DOMAIN_TO_CATEGORIES = {
    1: [5, 7],
    2: [2, 3],
    3: [1, 4, 6]
}

# Categories → list of subcategories
CATEGORY_TO_SUBCATEGORIES = {
    1: [2, 10, 21, 24],
    2: [3, 14, 28, 31],
    3: [4, 6, 8, 9, 13, 30],
    4: [11, 23],
    5: [1, 19, 26],
    6: [12, 27],
    7: [5, 15, 16, 18, 22, 29]
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

    # -------------------------
    # 1) DOMAIN
    # -------------------------
    domain_preds = predict_domain_embedding(embedding)
    domain_final = domain_preds["xgboost"]  # choose XGBoost as final
    output["domain"] = domain_final

    # -------------------------
    # 2) CATEGORY (depends on domain)
    # -------------------------
    category_predictor = CATEGORY_PREDICTORS_EMBEDDING.get(domain_final)
    if not category_predictor:
        raise ValueError(f"No category embedding predictor for domain={domain_final}")

    category_preds = category_predictor(embedding)

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
    subcategory_predictor = SUBCATEGORY_PREDICTORS_EMBEDDING.get(category_final)

    if not subcategory_predictor:
        # fallback to first valid subcategory
        subcategory_final = CATEGORY_TO_SUBCATEGORIES[category_final][0]
    else:
        subcategory_preds = subcategory_predictor(embedding)

        # Majority voting for subcategory
        votes_subcategory = [
            subcategory_preds["logistic_regression"],
            subcategory_preds["random_forest"],
            subcategory_preds["xgboost"]
        ]
        subcategory_final = Counter(votes_subcategory).most_common(1)[0][0]

        # Validate
        if subcategory_final not in CATEGORY_TO_SUBCATEGORIES[category_final]:
            subcategory_final = CATEGORY_TO_SUBCATEGORIES[category_final][0]

    output["subcategory"] = subcategory_final

    return output



# ==========================================
# TEST EXAMPLE
# ==========================================

if __name__ == "__main__":
    example_text = "The nurse did not respond quickly and follow-up was poor."
    raw = get_embedding(example_text)
    embedding = np.frombuffer(raw, dtype=np.float32)


    result_text = hierarchical_predict_text(example_text)
    result_embedding = hierarchical_predict_embeddings(embedding)
    print(json.dumps(result_text, indent=4))
    print(json.dumps(result_embedding, indent=4))



