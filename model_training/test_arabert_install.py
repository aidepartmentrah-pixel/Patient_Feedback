from transformers import AutoTokenizer, AutoModel
import torch

model_name = "aubmindlab/bert-base-arabertv2"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("✅ Model and tokenizer loaded successfully!")

text = "مرحبا، هذا اختبار بسيط."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

print("Embedding shape:", embedding.shape)
