import os
import torch
import torchaudio
import runpod
import base64
from io import BytesIO
import numpy as np
from scipy.io import wavfile
from scipy import signal
import traceback

# --- Global Variables & Model Loading with Error Catching ---
INIT_ERROR_FILE = "/tmp/init_error.log"
model = None

try:
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)
        
    print("Loading MusicGen large model...")
    from audiocraft.models import MusicGen
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = MusicGen.get_pretrained("facebook/musicgen-large", device=device)
    print("✅ Model loaded successfully.")

except Exception as e:
    tb_str = traceback.format_exc()
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(f"Failed to initialize model: {tb_str}")
    model = None

# --- Helper Functions ---
def upsample_audio(input_wav_bytes, target_sr=48000):
    try:
        with BytesIO(input_wav_bytes) as in_io:
            sr, audio = wavfile.read(in_io)

        up_factor = target_sr / sr
        upsampled_audio = signal.resample(audio, int(len(audio) * up_factor))
        if audio.dtype == np.int16:
            upsampled_audio = upsampled_audio.astype(np.int16)

        with BytesIO() as out_io:
            wavfile.write(out_io, target_sr, upsampled_audio)
            return out_io.getvalue()
    except Exception:
        return input_wav_bytes

# --- Runpod Handler ---
def handler(event):
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            return {"error": f"Worker initialization failed: {f.read()}"}

    job_input = event.get("input", {})
    text = job_input.get("text")
    
    if not text:
        return {"error": "No text prompt provided."}
    
    try:
        duration = job_input.get("duration", 120)
        sample_rate = job_input.get("sample_rate", 32000)
        
        # Generate audio synchronously
        model.set_generation_params(duration=duration)
        res = model.generate([text])
        audio_tensor = res[0].cpu()
        
        buffer = BytesIO()
        torchaudio.save(buffer, audio_tensor, model.sample_rate, format="wav")
        raw_wav_bytes = buffer.getvalue()
        
        final_wav_bytes = raw_wav_bytes
        if sample_rate == 48000:
            final_wav_bytes = upsample_audio(raw_wav_bytes, target_sr=48000)
            
        audio_base64 = base64.b64encode(final_wav_bytes).decode('utf-8')
        
        return {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "format": "wav"
        }
        
    except Exception as e:
        return {"error": traceback.format_exc()}

# --- Start Serverless Worker ---
runpod.serverless.start({"handler": handler})
