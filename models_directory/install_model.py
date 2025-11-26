import os
from transformers import AutoTokenizer, AutoModel
import torch

# --------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Choose your local save path
local_model_path = r"Classification_Models/model_storage/mpnet_embeddings"

os.makedirs(local_model_path, exist_ok=True)

# --------------------------------------------
# 2. DOWNLOAD + SAVE MODEL LOCALLY (first run)
# --------------------------------------------
print("⬇️ Downloading model & tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("💾 Saving model locally...")
tokenizer.save_pretrained(local_model_path)
model.save_pretrained(local_model_path)

print("✅ Model saved successfully at:")
print(local_model_path)

# --------------------------------------------
# 3. LOAD MODEL OFFLINE (future runs)
# --------------------------------------------
print("\n📦 Loading model offline...")
tokenizer_offline = AutoTokenizer.from_pretrained(local_model_path)
model_offline = AutoModel.from_pretrained(local_model_path)

print("✅ Offline model loaded successfully!")

# --------------------------------------------
# 4. TEST: Generate an embedding
# --------------------------------------------
text = "مرحبا، هذا اختبار بسيط باستخدام نموذج MPNet متعدد اللغات."

inputs = tokenizer_offline(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True
)

with torch.no_grad():
    outputs = model_offline(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy
