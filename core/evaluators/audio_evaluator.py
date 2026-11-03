import os
import time
from dotenv import load_dotenv
from hume import HumeClient
from hume.expression_measurement.batch import Prosody, Models
from hume.expression_measurement.batch.types import InferenceBaseRequest

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))


class AudioConfidenceEvaluator:
    def __init__(self):
        self.api_key = os.getenv("HUME_API_KEY")
        if not self.api_key:
            print("[WARNING] HUME_API_KEY not found in .env! Audio evaluation will fail.")
            self.client = None
        else:
            self.client = HumeClient(api_key=self.api_key)
            print("[EVALUATOR] Deep Learning Audio Engine (Hume AI) Initialized.")

    def _neutral_result(self):
        """Single source of truth for fallback values so downstream keys always exist."""
        return {
            'confidence_score': 5.0,
            'determination': 0.0,
            'doubt': 0.0,
            'anxiety': 0.0,
            'tempo_bpm': 0,
            'fluency_ratio': 0.0
        }

    def evaluate_audio(self, audio_path):
        if not self.client:
            return self._neutral_result()

        print(f"\n[SYSTEM] Sending {os.path.basename(audio_path)} to Hume AI Prosody Model...")

        try:
            prosody_config = Prosody()
            models_chosen = Models(prosody=prosody_config)
            configs = InferenceBaseRequest(models=models_chosen)

            with open(audio_path, mode="rb") as file:
                job_id = self.client.expression_measurement.batch.start_inference_job_from_local_file(
                    json=configs,
                    file=[file]
                )

            print(f"[SYSTEM] Hume Job ID: {job_id}. Polling for completion...")

            # Poll with a max-wait so a stuck job doesn't freeze the interview
            max_wait_seconds = 60
            elapsed = 0
            while True:
                details = self.client.expression_measurement.batch.get_job_details(id=job_id)
                status = str(details.state.status)

                if "COMPLETED" in status:
                    break
                elif "FAILED" in status:
                    raise Exception("Job failed on Hume's cloud servers.")

                if elapsed >= max_wait_seconds:
                    raise Exception(f"Hume job timed out after {max_wait_seconds}s")

                time.sleep(1.0)
                elapsed += 1

            full_predictions = self.client.expression_measurement.batch.get_job_predictions(id=job_id)

            if not full_predictions or not full_predictions[0].results.predictions:
                print("[WARNING] Hume returned empty predictions. Was the audio silent or too short?")
                return self._neutral_result()

            file_predictions = full_predictions[0].results.predictions[0]
            if (not file_predictions.models
                    or not file_predictions.models.prosody
                    or not file_predictions.models.prosody.grouped_predictions):
                print("[WARNING] No vocal prosody detected in the audio file. Might be dead air.")
                return self._neutral_result()

            predictions = file_predictions.models.prosody.grouped_predictions[0].predictions

            confidence_scores = []
            hesitation_scores = []
            anxiety_scores = []

            for chunk in predictions:
                emotions = {e.name: e.score for e in chunk.emotions}
                confidence_scores.append(emotions.get('Determination', 0))
                hesitation_scores.append(emotions.get('Doubt', 0))
                anxiety_scores.append(emotions.get('Anxiety', 0))

            avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            avg_hes = sum(hesitation_scores) / len(hesitation_scores) if hesitation_scores else 0
            avg_anx = sum(anxiety_scores) / len(anxiety_scores) if anxiety_scores else 0

            print(f"[HUME XAI] Averages -> Determination: {avg_conf:.2f} | Doubt: {avg_hes:.2f} | Anxiety: {avg_anx:.2f}")

            # Calibrated grading matrix
            final_score = 5.0 + (avg_conf * 25.0) - (avg_hes * 15.0) - (avg_anx * 10.0)
            final_score = max(1.0, min(10.0, final_score))

            # FIX: return all the metrics the downstream code expects.
            return {
                'confidence_score': round(final_score, 1),
                'determination': round(float(avg_conf), 4),
                'doubt': round(float(avg_hes), 4),
                'anxiety': round(float(avg_anx), 4),
                'tempo_bpm': 0,
                'fluency_ratio': 0.0
            }

        except Exception as e:
            print(f"[ERROR] Hume AI Evaluation Failed: {e}")
            return self._neutral_result()
