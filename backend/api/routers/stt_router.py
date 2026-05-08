"""
STT (Speech-to-Text) Router
API endpoints for converting Arabic audio to text.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import os

from ..services.stt_service import transcribe_audio_to_text, transcribe_audio_bytes


router = APIRouter(prefix="/api/stt", tags=["STT"])


# ==================== ENDPOINTS ====================

@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file (mp3, wav, m4a, etc.)")
):
    """
    Convert Arabic audio file to text using Faster Whisper model.
    
    **Supported formats:**
    - MP3
    - WAV
    - M4A
    - OGG
    - FLAC
    
    **Example Request:**
    ```
    POST /api/stt/transcribe
    Content-Type: multipart/form-data
    
    audio: [audio file]
    ```
    
    **Returns:**
    - Transcribed Arabic text
    - Original filename
    """
    
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "NO_FILENAME",
                    "message": "Audio file must have a filename",
                    "message_ar": "يجب أن يحتوي ملف الصوت على اسم"
                }
            )
        
        # Check file extension
        allowed_extensions = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm"]
        file_ext = os.path.splitext(audio.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_FORMAT",
                    "message": f"Unsupported audio format: {file_ext}. Allowed: {', '.join(allowed_extensions)}",
                    "message_ar": f"صيغة صوت غير مدعومة: {file_ext}"
                }
            )
        
        # Read audio bytes
        audio_bytes = await audio.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "EMPTY_FILE",
                    "message": "Audio file is empty",
                    "message_ar": "ملف الصوت فارغ"
                }
            )
        
        # Check file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(audio_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "FILE_TOO_LARGE",
                    "message": f"Audio file too large. Maximum size: 50MB",
                    "message_ar": "ملف الصوت كبير جداً. الحد الأقصى: 50 ميجابايت"
                }
            )
        
        # Transcribe
        result = transcribe_audio_bytes(audio_bytes, audio.filename)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail={
                    "error": result.get("error", "STT_FAILED"),
                    "message": result.get("message", "Transcription failed"),
                    "message_ar": result.get("message_ar", "فشل تحويل الصوت إلى نص")
                }
            )
        
        return {
            "success": True,
            "text": result.get("text", ""),
            "filename": audio.filename,
            "size_bytes": len(audio_bytes)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}",
                "message_ar": f"حدث خطأ: {str(e)}"
            }
        )


@router.post("/transcribe-path")
async def transcribe_audio_from_path(
    audio_path: str = Form(..., description="Path to audio file on server")
):
    """
    Transcribe audio from a file path on the server.
    
    **Note:** This endpoint is for internal use when audio is already on the server.
    
    **Example Request:**
    ```json
    {
      "audio_path": "/path/to/audio.mp3"
    }
    ```
    
    **Returns:**
    - Transcribed Arabic text
    """
    
    try:
        result = transcribe_audio_to_text(audio_path)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": result.get("error", "STT_FAILED"),
                    "message": result.get("message", "Transcription failed"),
                    "message_ar": result.get("message_ar", "فشل تحويل الصوت إلى نص")
                }
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}",
                "message_ar": f"حدث خطأ: {str(e)}"
            }
        )


@router.get("/test")
async def test_stt():
    """
    Test endpoint to verify STT service is working.
    """
    
    return {
        "status": "operational",
        "service": "stt",
        "supported_formats": ["mp3", "wav", "m4a", "ogg", "flac", "aac"],
        "max_file_size": "50MB",
        "message": "Upload audio file to /api/stt/transcribe endpoint"
    }
