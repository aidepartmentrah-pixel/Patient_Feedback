from faster_whisper import WhisperModel

MODEL_SIZE = "medium"  # or "small"
DEVICE = "cpu"         # change to "cuda" if GPU exists

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

    text = ""
    for segment in segments:
        text += segment.text + " "

    return text.strip()
