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
import uuid
import time
from concurrent.futures import ThreadPoolExecutor

# --- Global Variables & Model Loading with Error Catching ---
INIT_ERROR_FILE = "/tmp/init_error.log"
tasks = {}
model = None
executor = ThreadPoolExecutor(max_workers=1)

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

def generate_audio_sync(task_id, text, duration, sample_rate):
    try:
        tasks[task_id]["status"] = "processing"
        
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
        
        tasks[task_id].update({
            "status": "completed",
            "output": {
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "format": "wav"
            }
        })
    except Exception as e:
        tasks[task_id].update({
            "status": "failed",
            "error": traceback.format_exc()
        })

# --- Runpod Handler ---
def handler(event):
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            return {"error": f"Worker initialization failed: {f.read()}"}

    job_input = event.get("input", {})
    task_id = job_input.get("task_id")

    # If no task_id, this is a new request
    if not task_id:
        text = job_input.get("text")
        if not text:
            return {"error": "No text prompt provided."}
            
        new_task_id = str(uuid.uuid4())
        tasks[new_task_id] = {"status": "pending"}
        
        duration = job_input.get("duration", 120)
        sample_rate = job_input.get("sample_rate", 32000)
        
        executor.submit(generate_audio_sync, new_task_id, text, duration, sample_rate)
        
        return {"task_id": new_task_id, "status": "pending"}

    # If task_id is provided, check the status
    else:
        task = tasks.get(task_id)
        if not task:
            return {"error": "Task not found."}
        
        # If completed, return the full result and remove from memory
        if task["status"] == "completed":
            result = {"task_id": task_id, "status": "completed", "output": task["output"]}
            del tasks[task_id]
            return result
            
        # If failed, return the error and remove from memory
        elif task["status"] == "failed":
            result = {"task_id": task_id, "status": "failed", "error": task["error"]}
            del tasks[task_id]
            return result
            
        # Otherwise, just return the current status
        else:
            return {"task_id": task_id, "status": task["status"]}

# --- Start Serverless Worker ---
runpod.serverless.start({"handler": handler})
