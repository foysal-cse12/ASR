import streamlit as st
import sounddevice as sd
import torch
import numpy as np
import json
import threading
import time
from scipy.signal import resample
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from flashtext import KeywordProcessor
from rapidfuzz import process
import warnings
warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
#MODEL_PATH = "primeline/whisper-large-v3-turbo-german"
MODEL_PATH = "D:\\ASR_Model\\whisper-large-v3-turbo-german"
VOCAB_PATH = "custom_vocab.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DURATION_CHUNK = 4.0
INPUT_RATE = 44100
MODEL_RATE = 16000


# =========================
# Load ASR Model & Processor
# =========================
@st.cache_resource
def load_model_and_processor():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH, torch_dtype=TORCH_DTYPE, use_safetensors=True
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


# =========================
# Load Custom Vocabulary
# =========================
@st.cache_data
def load_custom_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab


# =========================
# Initialize Keyword & Fuzzy Matching
# =========================
def init_processors(custom_vocab):
    keyword_processor = KeywordProcessor()
    dialect_map = {}
    for correct_term, variants in custom_vocab.items():
        for variant in variants:
            keyword_processor.add_keyword(variant.lower(), correct_term)
            dialect_map[variant.lower()] = correct_term
    return keyword_processor, dialect_map, list(dialect_map.keys())


def fuzzy_correction(text, variant_list, dialect_map, threshold=95):
    words = text.split()
    corrected = []
    for word in words:
        if word in variant_list:
            corrected.append(dialect_map[word])
        else:
            match = process.extractOne(word, variant_list)
            if match and match[1] >= threshold:
                corrected.append(dialect_map[match[0]])
            else:
                corrected.append(word)
    return " ".join(corrected)


# =========================
# Real-Time ASR Worker
# =========================
class ASRWorker:
    def __init__(self, pipe, keyword_processor, variant_list, dialect_map):
        self.pipe = pipe
        self.keyword_processor = keyword_processor
        self.variant_list = variant_list
        self.dialect_map = dialect_map
        self.running = False
        self.buffer = []
        self.transcript = ""
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.process_audio)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def reset(self):
        self.buffer = []
        self.transcript = ""

    def process_audio(self):
        def callback(indata, frames, time_info, status):
            if status:
                print("Status:", status)
            audio = indata[:, 0]
            resampled = resample(audio, int(len(audio) * MODEL_RATE / INPUT_RATE)).astype(np.float32)
            result = self.pipe(resampled)
            text = result.get("text", "").strip()
            if text:
                flashtext_corrected = self.keyword_processor.replace_keywords(text) #.lower()
                final_corrected = fuzzy_correction(flashtext_corrected, self.variant_list, self.dialect_map)
                self.buffer.append(final_corrected)
                self.transcript = " ".join(self.buffer)

        with sd.InputStream(callback=callback, channels=1, samplerate=INPUT_RATE,
                            blocksize=int(DURATION_CHUNK * INPUT_RATE), dtype='float32'):
            while self.running:
                time.sleep(0.1)

    def get_transcript(self):
        return self.transcript


# =========================
# Streamlit App
# =========================
def main():
    st.title("🎤 Real-Time German Transcription")

    model, processor = load_model_and_processor()
    custom_vocab = load_custom_vocab()
    keyword_processor, dialect_map, variant_list = init_processors(custom_vocab)

    # ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
        generate_kwargs={"language": "german", "task": "transcribe"},
    )

    if "worker" not in st.session_state:
        st.session_state.worker = ASRWorker(pipe, keyword_processor, variant_list, dialect_map)

    col1, col2 = st.columns(2)
    with col1:
        start = st.button("▶️ Start Recording")
    with col2:
        stop = st.button("⏹ Stop Recording")

    if start:
        st.session_state.worker.reset()  # Clear previous session
        st.session_state.worker.start()
        st.success("Recording started. Speak into the mic...")

    if stop:
        st.session_state.worker.stop()
        st.success("Recording stopped.")

    st.subheader("📝 Live Transcript:")
    transcript_placeholder = st.empty()

    while st.session_state.worker.running:
        transcript = st.session_state.worker.get_transcript()
        transcript_placeholder.markdown(f"**{transcript}**")
        time.sleep(0.5)

    # Final display after stopping
    #transcript = st.session_state.worker.get_transcript()
    #transcript_placeholder.markdown(f"**{transcript}**")

    # Final display after stopping
    if not st.session_state.worker.running and transcript_placeholder:
        final_transcript = st.session_state.worker.get_transcript()
        st.subheader("✅ Final Transcript:")
        st.markdown(f"**{final_transcript}**")


if __name__ == "__main__":
    main()
