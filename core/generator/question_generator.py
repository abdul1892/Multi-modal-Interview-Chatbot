import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))


class QuestionGenerator:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[GENERATOR] 'The Mouth' Initialized. Groq API connected.")

    def _call_with_retry(self, prompt, temperature=0.6, max_retries=3):
        """Exponential backoff for Groq rate limits / transient errors."""
        delay = 1.0
        last_exc = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_exc = e
                print(f"[WARN] Groq call attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
        raise last_exc

    def _parse_q_and_ideal(self, text, topic):
        q_part = text.split("QUESTION:")[1].split("IDEAL:")[0].strip() if "QUESTION:" in text else text
        i_part = text.split("IDEAL:")[1].strip() if "IDEAL:" in text else f"An accurate verbal explanation of {topic} concepts."
        return {"question": q_part, "ideal_answer": i_part}

    def generate_question(self, topic, level):
        print(f"[SYSTEM] Requesting Level {level} {topic} question from Groq...")

        prompt = f"""
        You are an expert Senior Technical Interviewer conducting a Voice-Only technical interview.
        Your task is to generate a single interview question AND a brief ideal answer.

        Context:
        - Topic: {topic}
        - Difficulty Level: {level} (scale 1 to 10)

        CRITICAL CONSTRAINTS:
        1. VOICE ONLY: Do NOT ask them to write functions, syntax, or code blocks.
        2. CONCISENESS: Keep the question direct. Do NOT ask multi-part questions.
        3. FORMAT: You MUST return the output in this exact format:
        QUESTION: [Your short, single-part question here]
        IDEAL: [A hyper-concise, 1-sentence ideal answer. Strip out ALL conversational fluff.
                Focus STRICTLY on the core technical keywords. DO NOT include extra explanation.]
        """
        try:
            text = self._call_with_retry(prompt, temperature=0.6)
            return self._parse_q_and_ideal(text, topic)
        except Exception as e:
            print(f"[ERROR] Groq API Call Failed after retries: {e}")
            return {
                "question": f"Could you please explain your experience with {topic}?",
                "ideal_answer": f"The candidate should accurately explain {topic}."
            }

    def generate_clarification(self, topic, original_q, candidate_a, ideal_a):
        print(f"[SYSTEM] Requesting Clarification nudge from Groq for {topic}...")

        prompt = f"""
        You are an expert Senior Technical Interviewer conducting a Voice-Only technical interview.

        Context:
        - Topic: {topic}
        - Original Question: "{original_q}"
        - Candidate's Partial Answer: "{candidate_a}"
        - What they missed (Ideal Answer): "{ideal_a}"

        The candidate got partial credit. Ask a brief, 1-sentence follow-up question
        to nudge them to provide the missing logic.

        CRITICAL CONSTRAINTS:
        1. VOICE ONLY.
        2. FORMAT: You MUST return the output in this exact format:
        QUESTION: [Your brief follow-up question]
        IDEAL: [A hyper-concise, 1-sentence ideal answer for the follow-up. FIRST-PERSON ("I would...").]
        """
        try:
            text = self._call_with_retry(prompt, temperature=0.6)
            return self._parse_q_and_ideal(text, topic)
        except Exception:
            return {
                "question": "Can you elaborate slightly on the specific logic or conditions for that approach?",
                "ideal_answer": "I would use specific logic to complete the operation."
            }

    def generate_final_report(self, full_timeline_prompt):
        print("[SYSTEM] Requesting Comprehensive Final Executive Report from Groq...")
        try:
            return self._call_with_retry(full_timeline_prompt, temperature=0.3)
        except Exception as e:
            print(f"[ERROR] Groq Report Generation Failed: {e}")
            return "Final Evaluation Generation Failed. Please check API limits or network connection."
