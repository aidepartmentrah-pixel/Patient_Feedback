import time
from faster_whisper import WhisperModel

MODEL_SIZE = "small"
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
            compute_type="int8",
            cpu_threads=4,
            num_workers=1
        )
    return _model

def transcribe_arabic(audio_path: str) -> str:
    start = time.time()
    model = get_whisper_model()
    segments, info = model.transcribe(
        audio_path,
        language="ar",
        beam_size=1
    )
    result = " ".join(segment.text for segment in segments)
    elapsed = time.time() - start
    print(f"[STT] Transcription time: {elapsed:.3f} sec")
    return result
