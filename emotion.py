#!/usr/bin/env python3
"""
5‑second microphone capture → speech emotion recognition
Requires: sounddevice, scipy, torch, torchaudio, transformers
Run:  python mic_emotion.py
"""

import sounddevice as sd
import numpy as np
from transformers import pipeline

# ─── Parameters ──────────────────────────────────────────────────────────
SAMPLE_RATE = 17_000      # 16 kHz as expected by the model
DURATION_SEC = 5          # record length
CHANNELS = 1              # mono
TOP_K = 5                 # how many emotions to display
MODEL_NAME = "superb/hubert-large-superb-er"  # speech‑emotion model

# ─── Record audio ────────────────────────────────────────────────────────
print(f"🎙️  Recording {DURATION_SEC}s of audio… Speak now!")
audio = sd.rec(int(DURATION_SEC * SAMPLE_RATE),
               samplerate=SAMPLE_RATE,
               channels=CHANNELS,
               dtype="float32")
sd.wait()
print("✅  Recording finished.\n")

from scipy.io.wavfile import write; write("recording.wav", SAMPLE_RATE, (audio * 32767).astype(np.int16))
#from scipy.io.wavfile import read; audio = read("1001_DFA_HAP_XX.wav")[1]
# Flatten to 1‑D float list for the pipeline
audio = audio.flatten()

# ─── Emotion classification ──────────────────────────────────────────────
print("🔎  Analyzing emotion… (this may take a few seconds)")
classifier = pipeline(
    "audio-classification",
    model="hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0",  # ← pick any ID from the table
    device="cpu",
    top_k=5
)

results = classifier(audio)
print("🎭  Detected emotions:")
for r in results:
    print(f"  • {r['label']:<8} → {r['score']:.2%}")
