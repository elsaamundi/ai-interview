import os
import torch
import soundfile as sf
import math
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from utils.video_audio_utils import extract_audio


# =======================
# LOAD MODEL LOKAL
# =======================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "whisper-large-v2-en")

processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# =======================
# TRANSCRIBE VIDEO
# =======================

def transcribe_video(video_path, prompt=""):

    audio_path = extract_audio(video_path)
    audio, sr = sf.read(audio_path)

    if sr != 16000:
        raise ValueError(f"Audio sample rate harus 16000Hz, dapat {sr}")

    chunk_size = sr * 30   # 30 detik per chunk
    overlap = 2 * sr   # overlap 2 detik
    total_samples = len(audio)
    num_chunks = math.ceil(total_samples / chunk_size)

    texts = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_samples)

        chunk = audio[start:end]

        inputs = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                task="transcribe",
                language="en"
            )

        text = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        texts.append(text.strip())

    return " ".join(texts)

# ============================= alternatif
'''
import os
import torch
import soundfile as sf
from transformers import pipeline
from utils.video_audio_utils import extract_audio

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "whisper-large-v2-en")

device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    feature_extractor=MODEL_DIR,
    chunk_length_s=30,         # sangat penting untuk audio panjang
    return_timestamps=True,
    device=device
)

def transcribe_video(video_path, prompt=""):

    audio_path = extract_audio(video_path)

    result = asr(
        audio_path,
        generate_kwargs={
            "task": "transcribe",
            "language": "en",
            "prompt": prompt,
        },
        return_timestamps=True
    )

    # Hasil full text
    full_text = result["text"]

    # Optional — segments
    segments = result.get("chunks", [])

    formatted_segments = []
    for seg in segments:
        start, end = seg["timestamp"]
        text = seg["text"].strip()
        formatted_segments.append(
            f"[{start:6.2f}s - {end:6.2f}s] {text}"
        )

    return {
        "text": full_text,
        "segments": "\n".join(formatted_segments)
    }
'''

