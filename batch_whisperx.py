import whisperx
from whisperx.diarize import DiarizationPipeline
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
import time
import datetime
import shutil
import torch
torch.set_num_threads(1)

# CONFIGURATION
INPUT_DIR = "input_audios"
OUTPUT_DIR = "output_transcripts"
PROCESSED_DIR = "processed_audios"
MODEL_SIZE = "medium"
DEVICE = "cpu"  # Or "mps" on Apple Silicon
COMPUTE_TYPE = "float32"  # "float32" if float16 errors occur
LANGUAGE = "en"
HF_TOKEN = "xxxxxxxxxxxxxxxxxxxxx"  # Replace with your real token

def transcribe_file(file_path):
    try:
        start_time = time.time()  # ⏱️ Start timing
        file_path = Path(file_path)
        output_path = Path(OUTPUT_DIR) / f"{file_path.stem}_transcript.txt"

        print(f"🔄 {file_path.name}: Loading model...")
        model = whisperx.load_model(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)

        print(f"🎧 {file_path.name}: Loading audio...")
        audio = whisperx.load_audio(file_path)

        print(f"📝 {file_path.name}: Transcribing...")
        result = model.transcribe(audio)

        # Show transcript live (pre-diarization)
        for seg in result["segments"]:
            print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text'].strip()}")

        print(f"📐 {file_path.name}: Aligning words...")
        model_a, metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device=DEVICE, return_char_alignments=False)

        print(f"🔊 {file_path.name}: Audio shape: {audio.shape}")
        duration = audio.shape[0] / 16000  # Assuming 16kHz sample rate
        print(f"⏱️ {file_path.name}: Duration = {duration:.2f} seconds")
        print(f"🔊 {file_path.name}: Diarizing...")
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        diarize_segments = diarize_model(audio)
        print(diarize_segments)

        print(f"👥 {file_path.name}: Assigning speakers...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        speaker_map = {}
        for seg in result["segments"]:
            spk = seg.get("speaker", "unknown")
            if spk not in speaker_map:
                speaker_map[spk] = f"Person {chr(65 + len(speaker_map))}"  # 65 = 'A'
            seg["speaker_label"] = speaker_map[spk]

        print(f"💾 {file_path.name}: Saving transcript to {output_path}")
        with open(output_path, "w") as f:
            for seg in result["segments"]:
                text = seg["text"].strip()
                f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['speaker_label']}: {text}\n")

        end_time = time.time()
        elapsed = end_time - start_time
        formatted = str(datetime.timedelta(seconds=int(elapsed)))
        print(f"✅ Done: {file_path.name} (⏱️ Time taken: {formatted})")

        # Move file after successful processing
        processed_dir = Path(PROCESSED_DIR)
        processed_dir.mkdir(exist_ok=True)
        shutil.move(str(file_path), processed_dir / file_path.name)
        print(f"📦 {file_path.name}: Moved to {processed_dir}")

        return True

    except Exception as e:
        print(f"❌ Failed {file_path.name}: {e}")
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    audio_files = sorted(
    [
        Path(INPUT_DIR) / f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".mp3", ".wav", ".m4a"))
    ],
    key=lambda f: f.stat().st_size
    )
    audio_files = [str(f) for f in audio_files]

    print(f"📁 Found {len(audio_files)} audio files to process...\n")

    with Pool(processes=min(cpu_count(), 3)) as pool:
        for _ in tqdm(pool.imap_unordered(transcribe_file, audio_files), total=len(audio_files), desc="Batch Progress"):
            pass

if __name__ == "__main__":
    total_start = time.time()
    main()
    total_end = time.time()
    total_elapsed = str(datetime.timedelta(seconds=int(total_end - total_start)))
    print(f"\n🧾 Total batch processing time: {total_elapsed}")