from models_directory.STT_Models.Faster_Wisper import transcribe_arabic


def transcribe_audio_to_text(audio_path: str) -> dict:
    """
    Independent STT service.
    Converts audio file to text.
    """

    if not audio_path:
        return {
            "success": False,
            "error": "NO_AUDIO",
            "message": "Audio path is required",
        }

    try:
        text = transcribe_arabic(audio_path)

        return {
            "success": True,
            "text": text,
        }

    except Exception as e:
        return {
            "success": False,
            "error": "STT_FAILED",
            "message": str(e),
        }
