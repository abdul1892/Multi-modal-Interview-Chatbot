"""
Speech-to-Text engine with two interchangeable backends:

  * "groq"  -> Groq-hosted Whisper large-v3-turbo  (online: fast + accurate)
  * "local" -> local OpenAI Whisper "medium"        (offline: no network needed)

HOW TO SWITCH BACKEND
---------------------
Comment / uncomment exactly ONE line in the selector block below.
Everything else (the rest of the app) stays the same — app.py only ever calls
stt_engine.transcribe_audio(path), which works identically for both backends.

Both implementations live in this file on purpose: the engine genuinely
supports local and hosted transcription, you just pick which one is active.
"""
import os
import warnings
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Suppress the harmless CPU FP16 notice emitted by local Whisper.
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


# ===========================================================================
#  STT BACKEND SELECTOR  —  comment / uncomment ONE line
# ===========================================================================
STT_BACKEND = "groq"      # ONLINE : Groq API, model whisper-large-v3-turbo
#STT_BACKEND = "local"   # OFFLINE: local Whisper "medium" model
# ===========================================================================
#
# (Optional) Auto-pick by connectivity instead of toggling by hand — uncomment
# the four lines below and they will override the manual choice above:
#
# import socket
# def _online(host="api.groq.com", port=443, timeout=2.0):
#     try: socket.create_connection((host, port), timeout=timeout); return True
#     except OSError: return False
# STT_BACKEND = "groq" if _online() else "local"
# ===========================================================================

# Local Whisper model size used when STT_BACKEND == "local".
LOCAL_WHISPER_MODEL = "medium"

# Short context prompt that nudges Whisper toward technical spellings. Used by
# the Groq backend (max ~224 tokens). Keep it generic to the interview domain.
_JARGON_PROMPT = (
    "Technical interview answer. May mention terms like Python, Java, SQL, "
    "Kubernetes, Docker, API, REST, polymorphism, idempotent, latency, "
    "recursion, big-O, hashmap, microservices, TCP, OAuth."
)


class SpeechToTextEngine:
    def __init__(self):
        self.backend = STT_BACKEND
        self.model = None       # local Whisper model handle
        self.client = None      # Groq client handle

        if self.backend == "local":
            # ---- OFFLINE: local Whisper -------------------------------------
            print(f"[SYSTEM] Loading Local STT Model (Whisper {LOCAL_WHISPER_MODEL})...")
            import whisper  # lazy import so the Groq path doesn't need it
            self.model = whisper.load_model(LOCAL_WHISPER_MODEL)
            print("[EVALUATOR] Speech-to-Text Stream Initialized (LOCAL Whisper).")
        else:
            # ---- ONLINE: Groq-hosted Whisper large-v3-turbo -----------------
            from groq import Groq  # lazy import so the local path doesn't need it
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("[WARNING] GROQ_API_KEY not found — Groq STT will fail. "
                      "Set the key or switch STT_BACKEND to 'local'.")
                self.client = None
            else:
                self.client = Groq(api_key=api_key)
            print("[EVALUATOR] Speech-to-Text Stream Initialized (GROQ whisper-large-v3-turbo).")

    # -----------------------------------------------------------------------
    #  Public API — identical signature for both backends.
    # -----------------------------------------------------------------------
    def transcribe_audio(self, audio_file_path):
        """Convert the candidate's recorded audio file to text. Returns '' on failure."""
        print(f"\n[STT] Transcribing candidate audio via {self.backend.upper()} backend...")
        if self.backend == "local":
            return self._transcribe_local(audio_file_path)
        return self._transcribe_groq(audio_file_path)

    # -----------------------------------------------------------------------
    #  OFFLINE backend: local Whisper medium
    #  (decodes webm/opus via ffmpeg, which Whisper calls internally)
    # -----------------------------------------------------------------------
    def _transcribe_local(self, audio_file_path):
        try:
            result = self.model.transcribe(audio_file_path, language="en")
            transcription = result["text"].strip()
            print(f"[STT] Transcription complete: '{transcription}'")
            return transcription
        except Exception as e:
            print(f"[ERROR] Local Whisper transcription failed: {e}")
            return ""

    # -----------------------------------------------------------------------
    #  ONLINE backend: Groq whisper-large-v3-turbo
    #  Accepts webm directly — no transcode needed.
    # -----------------------------------------------------------------------
    def _transcribe_groq(self, audio_file_path):
        if not self.client:
            print("[ERROR] Groq STT client not initialized (missing API key).")
            return ""
        try:
            with open(audio_file_path, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_file_path), f.read()),
                    model="whisper-large-v3-turbo",
                    prompt=_JARGON_PROMPT,
                    response_format="text",   # returns the plain text string
                    language="en",
                    temperature=0.0,
                )
            # With response_format="text" the SDK returns the string directly;
            # guard for the object form just in case.
            text = transcription if isinstance(transcription, str) else getattr(transcription, "text", "")
            text = (text or "").strip()
            print(f"[STT] Transcription complete: '{text}'")
            return text
        except Exception as e:
            print(f"[ERROR] Groq Whisper transcription failed: {e}")
            return ""
