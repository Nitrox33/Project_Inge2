#────────────────────────────────────────────────────────IMPORT ALL THE LIBRARIES ────────────────────────────────────────────────────────
import vosk
import sys
import os
import vosk
import json
import sounddevice as sd
import numpy as np
from transformers import pipeline
import warnings; warnings.filterwarnings("ignore", message=".*input name `inputs` is deprecated.*")

# ──────────────────────────────────────────────────────── PARAMETERS ────────────────────────────────────────────────────────
voice_to_text_model_path = "vosk-model-small-fr-0.22"
# Initialize the model with model-path
VTTmodel = vosk.Model(voice_to_text_model_path)
EMOTION_MODEL = "hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0"  # works well for French


RECORD = True


# ──────────────────────────────────────────────────────── FUNCTIONS ────────────────────────────────────────────────────────
def get_audio_from_file(file_path):
    # Load the audio file
    from scipy.io.wavfile import read; audio = read(file_path)[1]
    print("✅  Using pre-recorded audio file.")
    return audio

def get_audio_from_microphone(duration_sec=10, sample_rate=17000, channels=1):
    # Record audio from the microphone
    print(f"🎙️  Recording {duration_sec}s of audio… Speak now!")
    audio = sd.rec(int(duration_sec * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype="float32")
    sd.wait()
    from scipy.io.wavfile import write; write("recording.wav", sample_rate, (audio * 32767).astype(np.int16))
    print("✅  Recording finished.\n")
    return audio

# def transcribe_audio(audio, sample_rate=17000):
#     recognizer = vosk.KaldiRecognizer(VTTmodel, sample_rate)
#     text = ""
#     for chunk in np.array_split(audio, len(audio) // (sample_rate // 10)):
#         if recognizer.AcceptWaveform(chunk.tobytes()):
#             text += recognizer.Result()
#     text += recognizer.FinalResult()
#     return text

def transcribe_audio(audio, sample_rate=17000):
    audio = audio.flatten()
    asr = pipeline("automatic-speech-recognition",
               model="eustlb/distil-large-v3-fr")               # rapide & précis :contentReference[oaicite:1]{index=1}
    text = asr(audio)["text"]
    return text

def classify_audio(audio, model_name=EMOTION_MODEL):
    # Flatten to 1‑D float list for the pipeline
    audio = audio.flatten()

    # ─── Emotion classification ──────────────────────────────────────────────
    print("🔎  Analyzing emotion… (this may take a few seconds)")
    classifier = pipeline(
        "audio-classification",
        model=model_name,
        device="cpu",
        top_k=5
    )

    results = classifier(audio)
    print("🎭  Detected emotions:")
    for r in results:
        print(f"  • {r['label']:<8} → {r['score']:.2%}")
    return results

def classify_text(text, model_name="AgentPublic/camembert-base-toxic-fr-user-prompts"):
    # Initialize the classifier
    classifier = pipeline(
        "text-classification",
        model=model_name,
        device="cpu",
        top_k=None
    )

    # Classify the text
    results = classifier(text)[0]
    print("🎭  Detected emotions from text:")
    for r in results:
        print(f"  • {r['label']:<8} → {r['score']:.2%}")
    return results

# ──────────────────────────────────────────────────────── MAIN FUNCTION ────────────────────────────────────────────────────────
def main():
    # Check if the model is available
    if not os.path.exists(voice_to_text_model_path):
        print(f"Model not found at {voice_to_text_model_path}. Please download it.")
        sys.exit(1)

    # Get audio input from either file or microphone
    
    if RECORD:
        audio = get_audio_from_microphone()
    else:
        audio = get_audio_from_file("recording.wav")
    
    transcribed_text = transcribe_audio(audio)
    print("📝 Transcribed Text:", transcribed_text)
    
    emotion_results = classify_audio(audio)
    aggression_results = classify_audio(audio, model_name="Hemg/violence-audio-Recognition-1111")
    text_results = classify_text(transcribed_text, model_name="AgentPublic/camembert-base-toxic-fr-user-prompts")
    
    
    
    

# ──────────────────────────────────────────────────────── MAIN CALL ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()
