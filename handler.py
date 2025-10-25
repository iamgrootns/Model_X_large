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
import requests  # ✅ ADD THIS IMPORT
import urllib.parse

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

def upload_to_gcs(signed_url, audio_bytes, content_type="audio/wav"):
    """Upload audio to Google Cloud Storage using signed URL"""
    try:
        response = requests.put(
            signed_url,
            data=audio_bytes,
            headers={"Content-Type": content_type},
            timeout=300  # 5 minute timeout for large files
        )
        response.raise_for_status()
        print(f"✅ Uploaded to GCS: {signed_url[:100]}...")
        return True
    except Exception as e:
        print(f"❌ GCS upload failed: {e}")
        return False

def notify_backend(callback_url, status, error_message=None):
    """Send webhook notification to backend"""
    try:
        # Add status and error_message as query parameters
        parsed = urllib.parse.urlparse(callback_url)
        params = urllib.parse.parse_qs(parsed.query)
        params['status'] = [status]
        if error_message:
            params['error_message'] = [error_message]
        
        # Reconstruct URL with new parameters
        new_query = urllib.parse.urlencode(params, doseq=True)
        webhook_url = urllib.parse.urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, new_query, parsed.fragment
        ))
        
        print(f"🔔 Calling webhook: {webhook_url}")
        response = requests.post(webhook_url, timeout=30)
        response.raise_for_status()
        print(f"✅ Backend notified: {status}")
        return True
    except Exception as e:
        print(f"❌ Webhook notification failed: {e}")
        return False

# --- Runpod Handler ---
def handler(event):
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            error_msg = f"Worker initialization failed: {f.read()}"
            return {"error": error_msg}

    job_input = event.get("input", {})
    text = job_input.get("text")
    callback_url = job_input.get("callback_url")
    upload_urls = job_input.get("upload_urls", {})
    
    if not text:
        error_msg = "No text prompt provided."
        if callback_url:
            notify_backend(callback_url, "failed", error_msg)
        return {"error": error_msg}
    
    try:
        duration = job_input.get("duration", 120)
        sample_rate = job_input.get("sample_rate", 32000)
        
        print(f"🎵 Generating audio: prompt='{text}', duration={duration}s")
        
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
        
        print(f"✅ Audio generated: {len(final_wav_bytes)} bytes")
        
        # Upload to GCS if signed URLs provided
        if upload_urls:
            wav_url = upload_urls.get("wav_url")
            if wav_url:
                upload_success = upload_to_gcs(wav_url, final_wav_bytes)
                if not upload_success:
                    raise Exception("Failed to upload WAV to GCS")
        
        # Notify backend of completion
        if callback_url:
            notify_backend(callback_url, "completed")
        
        # Also return the audio in response for backwards compatibility
        audio_base64 = base64.b64encode(final_wav_bytes).decode('utf-8')
        
        return {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "format": "wav",
            "status": "completed"
        }
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"❌ Error: {error_msg}")
        
        # Notify backend of failure
        if callback_url:
            notify_backend(callback_url, "failed", str(e))
        
        return {"error": error_msg, "status": "failed"}

# --- Start Serverless Worker ---
runpod.serverless.start({"handler": handler})
