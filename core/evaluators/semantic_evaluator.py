import os
import re
import json
import math
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Optional terminal-plotting dependency. We import lazily inside the render
# method so a missing plotext install can't break the entire evaluator.
try:
    import plotext as _plt
    _PLOTEXT_AVAILABLE = True
except ImportError:
    _PLOTEXT_AVAILABLE = False


class SemanticEvaluator:
    """
    Hybrid technical evaluator.

    Cosine similarity rewards topical overlap, not correctness — a vague answer
    that uses the right vocabulary can score as high as a correct one. To fix
    that, we use Groq with a strict rubric prompt as the PRIMARY scorer, and
    keep cosine similarity as a sanity-check signal returned alongside.

    The cosine projection is rendered in the terminal as a plotext braille
    chart right before each Q-table refresh, so the operator sees the geometric
    relationship between the ideal and candidate embedding vectors at a glance.
    Only Q1 is plotted — for non-negative similarities the candidate vector
    mathematically cannot land outside the unit box [0, 1] × [0, 1].
    """

    def __init__(self):
        print("[SYSTEM] Loading Local NLP Model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        groq_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=groq_key) if groq_key else None
        if not self.client:
            print("[WARNING] GROQ_API_KEY missing — falling back to cosine-only scoring.")

        if not _PLOTEXT_AVAILABLE:
            print("[WARNING] plotext not installed — terminal cosine chart disabled. "
                  "Install with: pip install plotext")

        print("[EVALUATOR] Semantic Stream Initialized.")

    def _cosine_score(self, ideal_answer, candidate_answer):
        embeddings = self.model.encode([ideal_answer, candidate_answer])
        ideal_vector = embeddings[0].reshape(1, -1)
        candidate_vector = embeddings[1].reshape(1, -1)
        similarity_score = float(cosine_similarity(ideal_vector, candidate_vector)[0][0])
        normalized = max(1.0, min(10.0, similarity_score * 10))
        return similarity_score, normalized

    def _rubric_score(self, ideal_answer, candidate_answer):
        """Ask Groq to grade against the ideal with a strict numeric rubric."""
        prompt = f"""You are a strict technical interview grader.

[IDEAL ANSWER]
{ideal_answer}

[CANDIDATE ANSWER]
{candidate_answer}

Grade the candidate answer on a scale of 1.0 to 10.0 based ONLY on technical correctness
against the ideal. Reward correct identification of the core concept. Penalize answers
that use the right vocabulary but state something factually wrong. Penalize hallucinated
or irrelevant content. A completely empty / off-topic answer is 1.0. A fully correct
answer with the key concept stated is 9.0–10.0. Partial credit (correct topic, missing
the key mechanism) is 4.0–6.0.

Respond with ONLY a JSON object of the form:
{{"score": <float 1.0-10.0>, "reason": "<one short sentence>"}}
Do not wrap in markdown.
"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if Groq adds them despite instructions
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()

            data = json.loads(raw)
            score = float(data.get("score", 5.0))
            score = max(1.0, min(10.0, score))
            return score, data.get("reason", "")
        except Exception as e:
            print(f"[WARNING] Rubric scoring failed, falling back to cosine: {e}")
            return None, None

    def _print_cosine_quadrant(self, similarity, tech_score):
        """
        Render the cosine projection as a plotext braille chart of Q1.

        Two vectors are drawn from the origin:
            * Ideal     — endpoint at (1, 0), cyan
            * Candidate — endpoint at (cos θ, sin θ), magenta

        Called at the end of evaluate_answer() so the chart prints in the
        terminal immediately before micro_agent.display_q_table().
        """
        sim = max(0.0, min(1.0, similarity))
        angle_rad = math.acos(sim)
        angle_deg = math.degrees(angle_rad)
        cand_x = math.cos(angle_rad)   # mathematically equal to sim
        cand_y = math.sin(angle_rad)

        print()
        print("=" * 62)
        print("  COSINE SIMILARITY PROJECTION  (Q1 quadrant)")
        print("-" * 62)
        print(f"  Similarity : {sim:.4f}        Angle : {angle_deg:6.2f}°")
        print(f"  Ideal      : (1.000, 0.000)   [cyan]")
        print(f"  Candidate  : ({cand_x:.3f}, {cand_y:.3f})   [magenta]")
        print(f"  Tech score : {tech_score}/10")

        if not _PLOTEXT_AVAILABLE:
            print("  [plotext not installed — install with: pip install plotext]")
            print("=" * 62)
            return

        try:
            _plt.clear_figure()
            _plt.theme("dark")
            _plt.plot_size(58, 18)

            # Vector lines from origin to each endpoint
            _plt.plot([0.0, 1.0],    [0.0, 0.0],    color="cyan",     label="ideal")
            _plt.plot([0.0, cand_x], [0.0, cand_y], color="magenta+", label="candidate")

            # Endpoint markers (drawn on top of the lines for visibility)
            _plt.scatter([1.0],    [0.0],    color="cyan",     marker="hd")
            _plt.scatter([cand_x], [cand_y], color="magenta+", marker="hd")

            _plt.title(f"sim={sim:.4f}  angle={angle_deg:.2f}°")
            _plt.xlim(-0.05, 1.05)
            _plt.ylim(-0.05, 1.05)
            _plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            _plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])

            _plt.show()
        except Exception as e:
            # Never let a chart-rendering bug break the actual evaluation.
            print(f"[WARNING] plotext render failed: {e}")

        print("=" * 62)

    def evaluate_answer(self, ideal_answer, candidate_answer):
        # Handle empty / whitespace answers up front
        if not candidate_answer or not candidate_answer.strip():
            return {
                "similarity_raw": 0.0,
                "tech_score": 1.0,
                "rubric_reason": "No answer provided."
            }

        similarity_raw, cosine_normalized = self._cosine_score(ideal_answer, candidate_answer)

        if self.client:
            rubric_score, rubric_reason = self._rubric_score(ideal_answer, candidate_answer)
            if rubric_score is not None:
                final_score = round(rubric_score, 1)
                # Print the Q1 projection right before returning, so the
                # subsequent micro_agent.display_q_table() prints under it.
                self._print_cosine_quadrant(similarity_raw, final_score)
                return {
                    "similarity_raw": round(similarity_raw, 4),
                    "tech_score": final_score,
                    "rubric_reason": rubric_reason
                }

        # Fallback path: no Groq available
        final_score = round(cosine_normalized, 1)
        self._print_cosine_quadrant(similarity_raw, final_score)
        return {
            "similarity_raw": round(similarity_raw, 4),
            "tech_score": final_score,
            "rubric_reason": "Cosine-only fallback."
        }