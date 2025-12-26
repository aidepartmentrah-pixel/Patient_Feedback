"""
STT (Speech-to-Text) Service
Converts Arabic audio files to text using Faster Whisper model.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add workspace root to path to import models_directory
# From: backend/api/services/stt_service.py
# To: Patient_Feedback/ (4 levels up)
workspace_root = Path(__file__).resolve().parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from models_directory.STT_Models.Faster_Wisper import transcribe_arabic


def transcribe_audio_to_text(audio_path: str) -> dict:
    """
    Independent STT service.
    Converts audio file to text.
    
    Args:
        audio_path: Path to audio file (mp3, wav, m4a, etc.)
        
    Returns:
        Dictionary with transcribed text
    """

    if not audio_path:
        return {
            "success": False,
            "error": "NO_AUDIO",
            "message": "Audio path is required",
            "message_ar": "مسار الصوت مطلوب"
        }
    
    if not os.path.exists(audio_path):
        return {
            "success": False,
            "error": "FILE_NOT_FOUND",
            "message": f"Audio file not found: {audio_path}",
            "message_ar": f"ملف الصوت غير موجود: {audio_path}"
        }

    try:
        text = transcribe_arabic(audio_path)

        return {
            "success": True,
            "text": text,
            "audio_path": audio_path
        }

    except Exception as e:
        return {
            "success": False,
            "error": "STT_FAILED",
            "message": f"Transcription failed: {str(e)}",
            "message_ar": f"فشل تحويل الصوت إلى نص: {str(e)}"
        }


def transcribe_audio_bytes(audio_bytes: bytes, filename: str) -> dict:
    """
    Transcribe audio from bytes (for file uploads).
    
    Args:
        audio_bytes: Audio file bytes
        filename: Original filename
        
    Returns:
        Dictionary with transcribed text
    """
    
    if not audio_bytes:
        return {
            "success": False,
            "error": "NO_AUDIO",
            "message": "Audio data is required",
            "message_ar": "بيانات الصوت مطلوبة"
        }
    
    # Save to temporary file
    temp_file = None
    try:
        # Get file extension
        ext = Path(filename).suffix if filename else ".wav"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        # Transcribe
        result = transcribe_audio_to_text(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result
        
    except Exception as e:
        # Clean up on error
        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        
        return {
            "success": False,
            "error": "STT_FAILED",
            "message": f"Transcription failed: {str(e)}",
            "message_ar": f"فشل تحويل الصوت إلى نص: {str(e)}"
        }
