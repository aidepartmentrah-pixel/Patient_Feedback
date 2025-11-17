import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import joblib
from tqdm import tqdm
import pandas as pd
from prepare_data_bert import df_train, df_test, encoders, FeedbackDataset, target_cols

# ======================================
# ✅ CONFIG
# ======================================
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2  # Keep small for CPU
LR = 2e-5

SAVE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SAVE_DIR, "multihead_bert_classifier.pt")
ENCODERS_PATH = os.path.join(SAVE_DIR, "label_encoders.pkl")
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer")

DEVICE = torch.device("cpu")

# ======================================
# ✅ LOAD TOKENIZER AND MODEL
# ======================================
print("🔠 Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(TOKENIZER_PATH)
print("✅ Tokenizer saved.")

# ======================================
# ✅ DATASETS & LOADERS
# ======================================
train_dataset = FeedbackDataset(df_train, tokenizer, MAX_LEN, target_cols)
test_dataset = FeedbackDataset(df_test, tokenizer, MAX_LEN, target_cols)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ======================================
# ✅ MULTI-HEAD BERT MODEL
# ======================================
class MultiHeadBERT(nn.Module):
    def __init__(self, bert, num_labels_dict):
        super().__init__()
        self.bert = bert
        hidden_size = bert.config.hidden_size

        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, n_classes)
            for name, n_classes in num_labels_dict.items()
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        logits = {name: head(pooled_output) for name, head in self.heads.items()}
        return logits

num_labels = {col: len(encoders[col].classes_) for col in target_cols}
model = MultiHeadBERT(bert_model, num_labels).to(DEVICE)

# ======================================
# ✅ OPTIMIZER & LOSS
# ======================================
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
criterion = nn.CrossEntropyLoss()

# ======================================
# ✅ TRAIN LOOP
# ======================================
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        loss = sum(criterion(logits[col], labels[:, i]) for i, col in enumerate(target_cols))
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = {col: [] for col in target_cols}, {col: [] for col in target_cols}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            for i, col in enumerate(target_cols):
                preds = torch.argmax(logits[col], dim=1).cpu().numpy()
                all_preds[col].extend(preds)
                all_labels[col].extend(labels[:, i].cpu().numpy())

    results = {}
    for col in target_cols:
        acc = accuracy_score(all_labels[col], all_preds[col])
        f1 = f1_score(all_labels[col], all_preds[col], average="macro", zero_division=0)
        results[col] = {"accuracy": acc, "f1_macro": f1}
    return results

# ======================================
# ✅ MAIN TRAINING
# ======================================
print("\n🚀 Starting training...")
for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader)
    metrics = evaluate(model, test_loader)

    print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")
    for col, m in metrics.items():
        print(f"  {col:<18} Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}")

# ======================================
# ✅ SAVE MODEL AND ENCODERS
# ======================================
torch.save(model.state_dict(), MODEL_PATH)
joblib.dump(encoders, ENCODERS_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")
print(f"✅ Label encoders saved to: {ENCODERS_PATH}")
print("✅ Training complete!")


