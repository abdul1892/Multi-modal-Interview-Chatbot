import os
import json
import PyPDF2
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB cap


class ResumeParser:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[PARSER] Resume Ingestion Engine Initialized.")

    @staticmethod
    def validate_pdf(pdf_path):
        """Return (ok, error_message). Caller should reject upload on failure."""
        if not os.path.exists(pdf_path):
            return False, "File does not exist."
        size = os.path.getsize(pdf_path)
        if size == 0:
            return False, "File is empty."
        if size > MAX_PDF_BYTES:
            return False, f"File too large ({size} bytes). Max is {MAX_PDF_BYTES}."

        # PDF magic-number check
        try:
            with open(pdf_path, 'rb') as f:
                header = f.read(5)
            if header[:4] != b'%PDF':
                return False, "Not a valid PDF (missing %PDF header)."
        except Exception as e:
            return False, f"Could not read file: {e}"

        return True, None

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"[ERROR] Failed to read PDF {pdf_path}: {e}")
            return None

    def parse_skills(self, pdf_path):
        print(f"\n[SYSTEM] Extracting dynamic skills from {os.path.basename(pdf_path)}...")

        ok, err = self.validate_pdf(pdf_path)
        if not ok:
            print(f"[WARNING] PDF validation failed: {err}. Using fallback skills.")
            return {"Python": 0.6, "Problem Solving": 0.5}

        resume_text = self.extract_text_from_pdf(pdf_path)
        if not resume_text:
            print("[WARNING] Resume text is empty. Using fallback skills.")
            return {"Python": 0.6, "Problem Solving": 0.5}

        prompt = f"""
        You are an expert Technical Engineering Manager.
        Read the following resume text and extract the candidate's top 3 to 4 core technical skills.
        Assign a confidence weight between 0.1 and 1.0 for each skill based on demonstrated experience.

        CRITICAL CONSTRAINT:
        Respond ONLY with a valid JSON dictionary where keys are skill names (strings) and values are weights (floats).
        Do NOT wrap in markdown blocks. No conversational text.

        Resume Text:
        {resume_text[:3000]}
        """

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            output = response.choices[0].message.content.strip()

            if output.startswith("```json"):
                output = output[7:-3].strip()
            elif output.startswith("```"):
                output = output[3:-3].strip()

            skills_dict = json.loads(output)
            print(f"[PARSER] HRL Matrix dynamically loaded with: {skills_dict}")
            return skills_dict
        except Exception as e:
            print(f"[ERROR] Groq Skill Parsing Failed: {e}")
            return {"Python": 0.8, "Data Structures": 0.7}
