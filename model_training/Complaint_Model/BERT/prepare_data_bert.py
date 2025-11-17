import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import os

# ======================================
# ✅ CONFIG
# ======================================
DB_PATH = r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\model_training\patient_feedback_ml.db"
TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"
TEXT_COL = "complaint_text"  # or whatever column contains the Arabic text

target_cols = ["domain", "category", "sub_category", "classification_ar"]

# ======================================
# ✅ DATA LOADING FUNCTION
# ======================================
import sqlite3

def load_table(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    print(f"✅ Loaded {len(df)} rows from {table_name}")
    return df


# ======================================
# ✅ CLEAN FUNCTION (ADAPTED)
# ======================================
def clean_and_encode_data(train_df, test_df, target_cols, text_col):
    print("🧹 Cleaning text and targets...")

    # Drop missing or invalid text rows
    train_df = train_df.dropna(subset=[text_col])
    test_df = test_df.dropna(subset=[text_col])

    # Remove whitespace and empty strings
    train_df[text_col] = train_df[text_col].astype(str).str.strip()
    test_df[text_col] = test_df[text_col].astype(str).str.strip()
    train_df = train_df[train_df[text_col] != ""]
    test_df = test_df[test_df[text_col] != ""]

    # Label encode each target column
    encoders = {}
    for col in target_cols:
        le = LabelEncoder()
        train_df = train_df.dropna(subset=[col])
        test_df = test_df.dropna(subset=[col])

        # Fit only on train set
        le.fit(train_df[col].astype(str))
        train_df[col] = le.transform(train_df[col].astype(str))

        # Apply encoder to test set safely
        test_df[col] = test_df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        test_df = test_df[test_df[col] != -1]  # drop unseen labels

        encoders[col] = le
        print(f"🧩 Encoded column '{col}' with {len(le.classes_)} classes.")

    print("✅ Cleaning and encoding complete.")
    return train_df, test_df, encoders


# ======================================
# ✅ MAIN CLEAN + PREP EXECUTION
# ======================================
df_train = load_table(DB_PATH, TRAIN_TABLE)
df_test = load_table(DB_PATH, TEST_TABLE)

df_train, df_test, encoders = clean_and_encode_data(df_train, df_test, target_cols, TEXT_COL)

print("\n📊 After cleaning:")
print(f"Train: {df_train.shape}, Test: {df_test.shape}")
print("Target sample counts:")
for col in target_cols:
    print(f"  {col}: {df_train[col].nunique()} classes")

# ======================================
# ✅ Example integration with Dataset class
# ======================================
from torch.utils.data import Dataset

class FeedbackDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, target_cols):
        self.texts = df[TEXT_COL].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_cols = target_cols

        self.targets = torch.tensor(df[target_cols].values, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = self.targets[idx]
        return item


print("\n✅ Dataset class ready for tokenizer use.")
