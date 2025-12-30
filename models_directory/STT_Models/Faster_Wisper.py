from faster_whisper import WhisperModel

MODEL_SIZE = "medium"
DEVICE = "cpu"

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type="int8"
)

def transcribe_arabic(audio_path: str) -> str:
    segments, info = model.transcribe(
        audio_path,
        language="ar",
        beam_size=5
    )

    return " ".join(segment.text for segment in segments)
