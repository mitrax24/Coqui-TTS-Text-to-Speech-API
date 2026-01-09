from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import tempfile
from TTS.api import TTS

# Initialize FastAPI app
app = FastAPI(title="Coqui TTS API")

# Load model once at startup (Tacotron2-DDC is the lightest)
print("Loading TTS model (Tacotron2-DDC)...")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False, verbose=False)
print("Model loaded successfully!")

# Hard limit for text input (memory safe)
MAX_TEXT_LENGTH = 200


@app.get("/tts")
def synthesize_speech(text: str):
    """
    Synthesize speech from text.
    
    Args:
        text: Input text to convert to speech (max 200 characters)
    
    Returns:
        WAV audio file
    """
    # Validate and sanitize input
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters"
        )
    
    try:
        # Generate audio file in /tmp
        output_path = os.path.join(tempfile.gettempdir(), "output.wav")
        tts.tts_to_file(text=text, file_path=output_path)
        
        # Return audio file
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename="speech.wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass


@app.get("/")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "model": "tacotron2-DDC"}
