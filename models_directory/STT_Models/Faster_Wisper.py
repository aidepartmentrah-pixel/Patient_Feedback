import time
from faster_whisper import WhisperModel

MODEL_SIZE = "medium"
DEVICE = "cpu"

# Toggle Arabic correction pass after transcription
USE_ARABIC_CORRECTION = True

CORRECTION_MODEL_NAME = "CAMeL-Lab/arabart-qalb15-gec-ged-13"

# Lazy load models to support Windows multiprocessing
_model = None
_correction_tokenizer = None
_correction_model = None


def get_whisper_model():
    """Lazy load Whisper model on first use."""
    global _model
    if _model is None:
        print(f"[STT] Loading Faster-Whisper model: {MODEL_SIZE}")
        _model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type="int8",
            cpu_threads=8,
            num_workers=1
        )
    return _model


def get_correction_model():
    """Lazy load the Arabic correction model on first use."""
    global _correction_tokenizer, _correction_model
    if _correction_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        print(f"[Correction] Loading correction model: {CORRECTION_MODEL_NAME}")
        _correction_tokenizer = AutoTokenizer.from_pretrained(CORRECTION_MODEL_NAME)
        _correction_model = AutoModelForSeq2SeqLM.from_pretrained(CORRECTION_MODEL_NAME)
    return _correction_tokenizer, _correction_model


def correct_arabic_text(text: str) -> str:
    """Run raw Arabic text through the correction model and return corrected text."""
    tokenizer, model = get_correction_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    output_ids = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + 64)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def transcribe_arabic(audio_path: str) -> str:
    start = time.time()
    model = get_whisper_model()
    segments, info = model.transcribe(
        audio_path,
        language="ar",
        beam_size=1
    )
    raw_text = " ".join(segment.text for segment in segments)
    elapsed = time.time() - start
    print(f"[STT] Transcription time: {elapsed:.3f} sec")

    print(f"[Correction] Arabic correction enabled: {USE_ARABIC_CORRECTION}")
    if not USE_ARABIC_CORRECTION:
        return raw_text

    try:
        correction_start = time.time()
        corrected_text = correct_arabic_text(raw_text)
        correction_elapsed = time.time() - correction_start
        print(f"[Correction] Correction time: {correction_elapsed:.3f} sec")
        return corrected_text
    except Exception as e:
        print(f"[Correction] Correction failed, returning raw transcription: {e}")
        return raw_text
