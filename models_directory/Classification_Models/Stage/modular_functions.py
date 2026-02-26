import numpy as np
import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
import re

# Global cache (initially empty)
_TOKENIZER = None
_MODEL = None


def metric_projection_score(metric_vec, sentence_vec):
    """
    Computes how much the sentence talks about the metric.
    Both vectors must already be L2-normalized.

    Returns:
        float in range [0, 1]
            1.0 → Perfect alignment
            0.5 → Neutral / unrelated
            0.0 → Opposite meaning
    """
    raw = float(np.dot(metric_vec, sentence_vec))   # [-1, 1]
    return (raw + 1) / 2                            # [0, 1]

def generate_metric_embedding(metric_name: str,vocab_list: list,storage_file: str = "metric_embeddings.json"):
    """
    Takes a list of words/phrases, embeds each with MPNet,
    computes their centroid, and saves it under 'metric_name'
    in a JSON file.
    """
    if os.path.exists(storage_file):
        with open(storage_file, "r", encoding="utf-8") as f:
            metric_store = json.load(f)
    else:
        metric_store = {}
    vectors = []
    for word in vocab_list:
        vec = get_embedding(word)  # Uses MPNet function defined earlier
        vectors.append(vec)

    vectors = np.vstack(vectors)  # shape (K, D)
    centroid = np.mean(vectors, axis=0)
    metric_store[metric_name] = centroid.tolist()

    with open(storage_file, "w", encoding="utf-8") as f:
        json.dump(metric_store, f, indent=2, ensure_ascii=False)

    print(f"[OK] Metric '{metric_name}' saved to: {storage_file}")
    return centroid

def load_metric_embedding(metric_name: str,
                          storage_file: str = "metric_embeddings.json") -> np.ndarray:

    if not os.path.exists(storage_file):
        raise FileNotFoundError(f"No metric file found: {storage_file}")

    with open(storage_file, "r", encoding="utf-8") as f:
        metric_store = json.load(f)

    if metric_name not in metric_store:
        raise KeyError(f"Metric '{metric_name}' not found in file.")

    return np.array(metric_store[metric_name])


def generate_metric_embedding_from_file(metric_name: str,vocab_folder: str = "vocab",
        storage_file: str = "metric_embeddings.json" ):
    """
    Loads a list of vocab terms from:
        ./vocab/<metric_name>.json

    Embeds them using MPNet (via embed_text function),
    computes the centroid,
    and saves it under `metric_name` inside metric_embeddings.json.
    """

    # ------------------------------------
    # 1. Load vocab JSON for this metric
    # ------------------------------------
    vocab_path = os.path.join(vocab_folder, f"{metric_name}.json")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"[ERROR] Vocab file not found: {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_list = json.load(f)

    if not isinstance(vocab_list, list):
        raise ValueError(f"[ERROR] Vocab file must contain a list. File: {vocab_path}")

    # ------------------------------------
    # 2. Embed each term
    # ------------------------------------
    vectors = []
    for term in vocab_list:
        vec = get_embedding(term)  # Must be defined elsewhere
        vectors.append(vec)

    vectors = np.vstack(vectors)  # (K, D)
    centroid = np.mean(vectors, axis=0)

    # ------------------------------------
    # 3. Load existing metric embedding storage
    # ------------------------------------
    if os.path.exists(storage_file):
        with open(storage_file, "r", encoding="utf-8") as f:
            metric_store = json.load(f)
    else:
        metric_store = {}

    # ------------------------------------
    # 4. Save/update centroid
    # ------------------------------------
    metric_store[metric_name] = centroid.tolist()

    with open(storage_file, "w", encoding="utf-8") as f:
        json.dump(metric_store, f, indent=2, ensure_ascii=False)
    return centroid

def clean_arabic(text: str) -> str:
    if text is None:
        return ""

    # Convert to string
    text = str(text)

    # Remove Arabic diacritics (Tashkeel)
    diacritics_pattern = re.compile(r"[\u064B-\u065F]")
    text = re.sub(diacritics_pattern, "", text)

    # Normalize characters
    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove non-Arabic letters and common punctuation
    text = re.sub(r"[^؀-ۿ0-9\s]+", " ", text)

    # Collapse extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text



#The 3 splitting functions. (They need systematics approach testing)
def split_arabic_text_into_sentences(text, max_sentences=6):
    import re

    if not text or not isinstance(text, str):
        return []

    # ================================
    # 1. Main separators (strong → weak)
    # ================================
    separators = [
        r"[.!؟\?]",  # sentence punctuation
        r"؛",
        r"،",
        r"\bثم\b",
        r"\sلكن\s",
        r"\bوبعد ذلك\b",
        r"\bبعد ذلك\b",
        r"\bوبعدها\b",
        r"\bوبالتالي\b",
        r"\bومن ثم\b",
        r"\bوعليه\b",
    ]

    combined = "(" + "|".join(separators) + ")"

    # ================================
    # 2. First pass split
    # ================================
    parts = re.split(combined, text)
    sentences = []
    temp = ""

    for part in parts:
        if re.match(combined, part):
            # part is a separator → attach to previous segment
            temp += part
        else:
            # part is new chunk
            if temp:
                sentences.append(temp.strip())
            temp = part.strip()

    if temp:
        sentences.append(temp.strip())

    # ================================
    # 3. Helper to remove useless fragments
    #    Short subsentences (≤ 8 chars) are considered unhelpful
    # ================================
    def is_valid(s):
        s = s.strip()

        # Remove tiny fragments
        if len(s) <= 8:
            return False

        # Remove punctuation-only
        banned = {".", ")", "(", ".)", ").", ". )", ") .", "\"", "''"}
        if s in banned:
            return False

        return True

    sentences = [s for s in sentences if is_valid(s)]

    # ================================
    # 4. Handle overly-long sentences
    # ================================
    SMART_SPLIT = r"(?:،| و |؛)"
    MAX_LEN = 220  # If above this length → force split

    final_sentences = []
    for s in sentences:
        if len(s) > MAX_LEN:
            # Force additional splitting
            extra = re.split(SMART_SPLIT, s)
            extra = [p.strip() for p in extra if is_valid(p)]
            final_sentences.extend(extra)
        else:
            final_sentences.append(s)

    sentences = final_sentences

    # ================================
    # 5. Reduce to max_sentences if needed
    # ================================
    if len(sentences) <= max_sentences:
        return sentences

    # Merge smallest neighbors until reduced to target
    while len(sentences) > max_sentences:
        shortest_idx = min(
            range(len(sentences) - 1),
            key=lambda i: len(sentences[i]) + len(sentences[i + 1])
        )

        sentences[shortest_idx] = (
                sentences[shortest_idx] + " " + sentences[shortest_idx + 1]
        )
        del sentences[shortest_idx + 1]

    return sentences

text = "هذا نص تجريبي. يحتوي على عدة جمل، وبعضها طويل جدًا بحيث يحتاج إلى تقسيم إضافي، مثل هذه الجملة التي تستمر لفترة طويلة جدًا بدون توقف، مما يجعل من الصعب قراءتها وفهمها بشكل صحيح. لذلك، نحتاج إلى التأكد من أن عملية التقسيم تعمل بشكل جيد!"
sentences = split_arabic_text_into_sentences(text)
for s in sentences:
    print(s)


def split_arabic_text_into_sentences_ml(text, max_sentences=6):
    """
    ML-based Arabic sentence splitter.
    Uses Stanza Arabic pipeline for sentence segmentation,
    then applies the same post-processing logic as the regex version.

    Requirements:
        pip install stanza

    And once per machine:
        import stanza
        stanza.download("ar")
    """
    import re
    import logging
    logging.getLogger("stanza").setLevel(logging.ERROR)
    # Lazy import (faster if not used)
    import stanza

    # Initialize Stanza Arabic model
    # (Normally you would move this outside the function so it loads only once)
    nlp = stanza.Pipeline(
        lang="ar",
        processors="tokenize",
        tokenize_no_ssplit=False
    )

    # ================================
    # 0. Safety check
    # ================================
    if not text or not isinstance(text, str):
        return []

    # ================================
    # 1. Run sentence segmentation
    # ================================
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sentences]

    # ================================
    # 2. Cleanup function for filtering
    # ================================
    def is_valid(s):
        s = s.strip()
        if len(s) <= 8:
            return False

        # Remove pure punctuation garbage
        banned = {".", ")", "(", ".)", ").", ". )", ") .", "\"", "''"}
        if s in banned:
            return False

        return True

    # Filter out useless short fragments
    sentences = [s for s in sentences if is_valid(s)]

    # ================================
    # 3. Smart forced re-splitting of very long sentences
    # ================================
    SMART_SPLIT = r"(?:،| و |؛)"
    MAX_LEN = 220

    final_sentences = []
    for s in sentences:
        if len(s) > MAX_LEN:
            # Split further on logical joiners
            extra = re.split(SMART_SPLIT, s)
            extra = [p.strip() for p in extra if is_valid(p)]
            final_sentences.extend(extra)
        else:
            final_sentences.append(s)

    sentences = final_sentences

    # ================================
    # 4. If already short enough → done
    # ================================
    if len(sentences) <= max_sentences:
        return sentences

    # ================================
    # 5. Merge smallest neighbors until we reach max_sentences
    # ================================
    while len(sentences) > max_sentences:
        # Find pair of neighbors with smallest combined length
        shortest_idx = min(
            range(len(sentences) - 1),
            key=lambda i: len(sentences[i]) + len(sentences[i + 1])
        )

        # Merge the two
        sentences[shortest_idx] = (
            sentences[shortest_idx] + " " + sentences[shortest_idx + 1]
        )
        del sentences[shortest_idx + 1]

    return sentences

def split_arabic_text_into_sentences_rules(text):
    # ==========================================================
    # 1) PROTECT PARENTHESIS SECTIONS FIRST
    # ==========================================================
    protected = []
    placeholder = []
    temp = text

    # Extract everything like (...) so we don't split inside it later
    for i, match in enumerate(re.findall(r'\(.*?\)', text)):
        key = f"@@P{i}@@"
        protected.append(match)
        placeholder.append(key)
        temp = temp.replace(match, key)

    # ==========================================================
    # 2) BASIC SPLITTING BASED ON PUNCTUATION
    # ==========================================================
    parts = re.split(r'[\.!\?…]+', temp)

    advanced = []

    # ==========================================================
    # 3) ADVANCED CLAUSE SPLITTING – SMART RULES
    # ==========================================================
    for p in parts:

        # ----------------------------------------------
        # RULE: Do not split dates like: 12-3-2024
        # Replace with a placeholder temporarily
        # ----------------------------------------------
        dates = re.findall(r'\d{1,2}-\d{1,2}-\d{2,4}', p)
        date_map = {}
        for i, d in enumerate(dates):
            dk = f"@@D{i}@@"
            date_map[dk] = d
            p = p.replace(d, dk)

        # ----------------------------------------------
        # Step A: First split by major separators
        # ----------------------------------------------
        sub = re.split(
            r'(?:،|؛| لكن | ثم | كما | حيث | وأن | وأنه )',
            p
        )

        # ----------------------------------------------
        # Step B: Smart split on "و" **only when it is starting a new clause**
        # i.e. A space before and after, next word starts with pronoun/verb
        # ----------------------------------------------
        refined = []
        for s in sub:
            chunks = re.split(
                r'(?<=\s)و\s+(?=[اأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي])',
                s
            )
            refined.extend(chunks)

        # ----------------------------------------------
        # Restore dates
        # ----------------------------------------------
        restored = []
        for x in refined:
            for dk, d in date_map.items():
                x = x.replace(dk, d)
            restored.append(x)

        advanced.extend(restored)

    # ==========================================================
    # 4) RESTORE PARENTHESIS
    # ==========================================================
    final = []
    for a in advanced:
        for key, org in zip(placeholder, protected):
            a = a.replace(key, org)
        final.append(a)

    # ==========================================================
    # 5) CLEAN + RETURN
    # ==========================================================
    return [x.strip() for x in final if len(x.strip()) > 0]



def get_model_path(Troubleshoot=False):
    """Finds the local MPNet model folder in the repo."""
    cwd = os.getcwd()
    for _ in range(10):  # go up max 10 levels
        candidate = os.path.join(
            cwd,
            "models_directory",
            "Classification_Models",
            "model_storage",
            "mpnet_embeddings"
        )
        if os.path.exists(candidate):
            if Troubleshoot:
                print("Using model path:", candidate)
            return candidate
        cwd = os.path.dirname(cwd)
    raise FileNotFoundError("MPNet embeddings folder not found in any parent directories")

def get_embedding(text: str, Troubleshoot=False):
    """
    Load MPNet model from local directory (only once).
    Resolves path relative to project root, works from any script.
    Returns embedding as bytes (float32).
    """

    global _TOKENIZER, _MODEL

    # -----------------------------
    # Determine project root
    # -----------------------------
    try:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    except NameError:
        # fallback if __file__ is not defined (e.g., interactive)
        PROJECT_ROOT = os.getcwd()

    MODEL_PATH = os.path.join(
        PROJECT_ROOT, "models_directory", "Classification_Models", "model_storage", "mpnet_embeddings"
    )
    MODEL_PATH = os.path.abspath(MODEL_PATH)
    MODEL_PATH = MODEL_PATH.replace("\\", "/")  # critical for HF local load on Windows

    if Troubleshoot:
        print(f"📦 Trying to load model from: {MODEL_PATH}")
        print("Directory exists:", os.path.exists(MODEL_PATH))
        if os.path.exists(os.path.dirname(MODEL_PATH)):
            print("Contents:", os.listdir(os.path.dirname(MODEL_PATH)))

    # -----------------------------
    # Load model only once
    # -----------------------------
    if _TOKENIZER is None or _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"MPNet model folder not found at {MODEL_PATH}")
        if Troubleshoot:
            print(f"📦 Loading MPNet model from:\n{MODEL_PATH}")

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        _MODEL = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
        _MODEL.eval()

    # -----------------------------
    # Handle empty input
    # -----------------------------
    if text is None:
        text = ""

    inputs = _TOKENIZER(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = _MODEL(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return emb.astype(np.float32).tobytes()

def get_embedding_list(texts, Troubleshoot=False):
    """
    Convert a list of texts into embeddings using the same MPNet model as get_embedding.

    Args:
        texts (List[str]): List of sentences.
        Troubleshoot (bool): If True, prints tokenization shapes and timing.

    Returns:
        List[bytes]: List of embeddings as bytes (float32), same as get_embedding.
    """
    global _TOKENIZER, _MODEL

    if not isinstance(texts, list):
        raise ValueError("Input must be a list of strings")

    # -----------------------------
    # Ensure model is loaded
    # -----------------------------
    if _TOKENIZER is None or _MODEL is None:
        # Attempt to load using same logic as get_embedding
        try:
            PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        except NameError:
            PROJECT_ROOT = os.getcwd()

        MODEL_PATH = os.path.join(
            PROJECT_ROOT, "models_directory", "Classification_Models", "model_storage", "mpnet_embeddings"
        )
        MODEL_PATH = os.path.abspath(MODEL_PATH).replace("\\", "/")

        if Troubleshoot:
            print(f"📦 Loading MPNet model from: {MODEL_PATH}")
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        _MODEL = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
        _MODEL.eval()

    # -----------------------------
    # Replace None entries with ""
    # -----------------------------
    texts = [t if t is not None else "" for t in texts]

    # -----------------------------
    # Tokenize all texts at once
    # -----------------------------
    inputs = _TOKENIZER(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    if Troubleshoot:
        print(f"[DEBUG] Tokenized {len(texts)} texts, input_ids shape: {inputs['input_ids'].shape}")

    # -----------------------------
    # Forward pass
    # -----------------------------
    with torch.no_grad():
        outputs = _MODEL(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        embeddings = embeddings.cpu().numpy().astype(np.float32)

    # -----------------------------
    # Return as list of bytes (same as single get_embedding)
    # -----------------------------
    return [emb.tobytes() for emb in embeddings]


def l2_normalize(vec):
    if vec is None:
        return None

    # If the embedding is stored as raw bytes → convert back to float32
    if isinstance(vec, (bytes, bytearray)):
        vec = np.frombuffer(vec, dtype=np.float32)

    v = np.array(vec, dtype=float)
    n = np.linalg.norm(v)

    return v / n if n > 0 else v



if __name__ == "__main__":
    print(get_embedding("Samee7", True))