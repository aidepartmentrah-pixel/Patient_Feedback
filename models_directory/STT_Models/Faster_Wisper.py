from faster_whisper import WhisperModel

MODEL_SIZE = "medium"
DEVICE = "cpu"

# Lazy load model to support Windows multiprocessing
_model = None

def get_whisper_model():
    """Lazy load Whisper model on first use."""
    global _model
    if _model is None:
        _model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type="int8"
        )
    return _model

def transcribe_arabic(audio_path: str) -> str:
    model = get_whisper_model()
    segments, info = model.transcribe(
        audio_path,
        language="ar",
        beam_size=5
    )

    return " ".join(segment.text for segment in segments)
