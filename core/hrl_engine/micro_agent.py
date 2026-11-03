import math


class MicroAgent:
    """
    Per-skill ability tracker — one (rating, uncertainty) pair per skill.

    NOTE ON THE NAME: the class is still called MicroAgent so nothing else in
    the app has to change its imports, but it is no longer a Q-learning agent.
    It is now a Glicko-style rating tracker (the intuitive cousin of Item
    Response Theory / adaptive testing).

    WHAT IT DOES
    ------------
    Think of each skill like a chess rating. The candidate has a rating per
    skill on the same 1-10 scale the question generator already uses, plus an
    uncertainty band around it:

        * answer well above the question's difficulty  -> rating goes UP
        * answer below it                              -> rating goes DOWN
        * the size of the move depends on how surprising the result was
        * every answer SHRINKS the uncertainty band

    We always ask the next question at the candidate's current rating, because
    a question they have a roughly 50/50 shot at is the most informative one.
    Once the uncertainty band is tight enough we are confident in the estimate
    and the topic can be retired.

    This replaces the old difficulty ladder, which stepped up on a pass and,
    after two failures, PERMANENTLY LOCKED the topic at a low level. A single
    nervous stumble could cap a candidate forever. Here a bad answer is just a
    small nudge that the estimate self-corrects from.
    """

    # Logistic scale. A 4-point rating gap means a ~91% expected score, so the
    # number 4 plays the role Elo's "400" does. Smaller = steeper curve.
    SCALE = 4.0

    # Starting assumptions for a brand-new skill.
    INITIAL_RATING = 4.0     # mid-low: the high levels have to be earned
    INITIAL_SIGMA = 3.0      # a wide band — at the start we know nothing

    # Once the uncertainty band is at/below this, we call the estimate solid.
    CONFIDENT_SIGMA = 1.0

    # The rating can never leave the usable difficulty range.
    MIN_RATING = 1.0
    MAX_RATING = 10.0

    def __init__(self, topics):
        self.ratings = {t: self.INITIAL_RATING for t in topics}
        self.sigmas = {t: self.INITIAL_SIGMA for t in topics}
        # (difficulty_asked, score_fraction, expected_score) per answer.
        self.history = {t: [] for t in topics}
        print(f"[RATING-TRACKER] Initialized {len(topics)} skill ratings "
              f"(start {self.INITIAL_RATING}±{self.INITIAL_SIGMA}, "
              f"confident at ±{self.CONFIDENT_SIGMA} or tighter).")

    # ------------------------------------------------------------------
    #  Core model: expected score for ability theta vs difficulty d.
    #  Same logistic curve IRT/Elo use; returns a probability in (0, 1).
    # ------------------------------------------------------------------
    def _expected(self, theta, difficulty):
        return 1.0 / (1.0 + 10 ** ((difficulty - theta) / self.SCALE))

    def _clamp(self, value):
        return max(self.MIN_RATING, min(self.MAX_RATING, value))

    # ------------------------------------------------------------------
    #  Read-only queries used by the rest of the app.
    # ------------------------------------------------------------------
    def get_current_level(self, topic):
        """Difficulty to ask next = the candidate's current rating, rounded.
        That is the most informative question (≈50% expected success)."""
        return int(round(self._clamp(self.ratings[topic])))

    def get_estimate(self, topic):
        """(rating, sigma) for the report — e.g. 'Python: 7.2 ± 0.4'."""
        return round(self.ratings[topic], 1), round(self.sigmas[topic], 2)

    def get_all_estimates(self):
        return {t: self.get_estimate(t) for t in self.ratings}

    def is_confident(self, topic):
        return self.sigmas[topic] <= self.CONFIDENT_SIGMA

    def get_uncertainty(self, topic):
        """
        0-100 figure for the UI and topic selection.
        100 = we know nothing yet, 0 = fully confident. Linear in sigma
        between the starting band and the confidence threshold.
        """
        span = self.INITIAL_SIGMA - self.CONFIDENT_SIGMA
        if span <= 0:
            return 0.0
        pct = (self.sigmas[topic] - self.CONFIDENT_SIGMA) / span * 100.0
        return round(max(0.0, min(100.0, pct)), 1)

    # ------------------------------------------------------------------
    #  The update — one "rated match" per answer.
    # ------------------------------------------------------------------
    def update(self, topic, difficulty, score_fraction):
        """
        Glicko-style single-observation update.

        difficulty      : the level we asked at (1-10).
        score_fraction  : rubric score / 10, in [0, 1]. The rich score is used
                          DIRECTLY as a partial win (a 7/10 = 0.7), instead of
                          being flattened to pass/fail.
        """
        if topic not in self.ratings:
            return

        theta = self.ratings[topic]
        sigma = self.sigmas[topic]
        variance = sigma * sigma

        # q converts between rating points and the logistic scale.
        q = math.log(10) / self.SCALE
        expected = self._expected(theta, difficulty)

        # Fisher information from this answer. Biggest when expected ≈ 0.5,
        # i.e. a question pitched right at the candidate's level teaches us
        # the most — which is exactly why we ask there.
        info = q * q * expected * (1.0 - expected)
        info = max(info, 1e-6)   # guard against a zero-information divide

        new_variance = 1.0 / (1.0 / variance + info)
        new_theta = self._clamp(theta + new_variance * q * (score_fraction - expected))
        new_sigma = math.sqrt(new_variance)

        self.ratings[topic] = new_theta
        self.sigmas[topic] = new_sigma
        self.history[topic].append((difficulty, round(score_fraction, 2), round(expected, 2)))

        print(f"\n[RATING] {topic}: asked@L{difficulty}  "
              f"scored {score_fraction * 10:.1f}/10  (expected {expected * 100:.0f}%)")
        print(f"[RATING] {topic}: rating {theta:.2f} -> {new_theta:.2f}   "
              f"uncertainty ±{sigma:.2f} -> ±{new_sigma:.2f}"
              f"{'   [CONFIDENT]' if self.is_confident(topic) else ''}")
        self.display_table(topic)

    # ------------------------------------------------------------------
    #  Terminal readout (keeps the old "live matrix" feel in the logs).
    # ------------------------------------------------------------------
    def display_table(self, topic):
        theta, sigma = self.get_estimate(topic)
        mark = int(round(theta))
        number_line = "".join("●" if i == mark else "·" for i in range(1, 11))
        print(f"\n=== [{topic.upper()}] LIVE ABILITY ESTIMATE ===")
        print(f"  Rating       : {theta}/10  (±{sigma})")
        print(f"  Uncertainty  : {self.get_uncertainty(topic):.0f}%"
              f"{'   ✓ locked-in' if self.is_confident(topic) else ''}")
        print(f"  Scale 1..10  : {number_line}")
        print(f"  Answers seen : {len(self.history[topic])}")
        print("-" * 50)
