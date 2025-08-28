import whisperx
from whisperx.diarize import DiarizationPipeline
import torch
import os

HF_TOKEN = "hf_your_actual_token_here"

print("🔎 Checking device availability...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Using device: {device}")

print("\n🔎 Verifying WhisperX model load...")
try:
    model = whisperx.load_model("tiny", device, compute_type="float16")
    print("✅ WhisperX transcription model loaded successfully.")
except Exception as e:
    print("❌ WhisperX model load failed:", e)

print("\n🔎 Verifying diarization model access...")
try:
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
    print("✅ Diarization model is accessible.")
except Exception as e:
    print("❌ Diarization model load failed:", e)
