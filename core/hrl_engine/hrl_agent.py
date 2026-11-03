class MacroAgent:
    """
    Topic-selection agent (depth-first).

    Decides which skill to probe next. A skill stays "active" until EITHER the
    rating tracker is statistically confident in it OR we hit the per-topic
    question cap (a safety valve). Among the active skills we prefer the most
    resume-important one, breaking ties toward the one we are still least sure
    about.

    WHAT CHANGED
    ------------
    The old version kept its own "uncertainty" number that started at 100 and
    was drained by hand-picked amounts (−30 on a good answer, −15 on a bad one,
    and so on). Those numbers were invented and meant nothing.

    Now uncertainty comes straight from the rating tracker's confidence band,
    so "this skill still has uncertainty" reflects how unsure we genuinely are
    about the candidate's ability — not an arbitrary timer. The agent no longer
    owns or mutates any uncertainty itself; it just reads it.
    """

    def __init__(self, skill_matrix, rating_tracker, max_questions_per_topic=5):
        self.skill_weights = skill_matrix
        self.rating = rating_tracker          # the MicroAgent rating tracker
        self.questions_asked = {skill: 0 for skill in skill_matrix.keys()}
        self.max_questions_per_topic = max_questions_per_topic
        self.current_focus = None
        print(f"[MACRO-AGENT] Initialized topic selector over {len(skill_matrix)} skills "
              f"(cap {max_questions_per_topic} questions/topic).")

    def _is_exhausted(self, topic):
        """A topic is finished when we're confident in it OR we've asked enough."""
        return (self.rating.is_confident(topic)
                or self.questions_asked[topic] >= self.max_questions_per_topic)

    def select_next_topic(self):
        # Depth-first: stay on the current topic while it's still worth probing.
        if self.current_focus and not self._is_exhausted(self.current_focus):
            print(f"[MACRO-AGENT] Staying on '{self.current_focus}' "
                  f"(uncertainty {self.rating.get_uncertainty(self.current_focus):.0f}%, "
                  f"asked {self.questions_asked[self.current_focus]}).")
            return self.current_focus

        if self.current_focus and self._is_exhausted(self.current_focus):
            reason = ("confident in rating" if self.rating.is_confident(self.current_focus)
                      else "question cap reached")
            print(f"[MACRO-AGENT] Done with '{self.current_focus}' ({reason}). Pivoting.")

        # Among not-yet-exhausted skills: highest resume weight first, then the
        # one we're still least sure about.
        candidates = [t for t in self.skill_weights if not self._is_exhausted(t)]
        if not candidates:
            return None

        candidates.sort(
            key=lambda t: (self.skill_weights[t], self.rating.get_uncertainty(t)),
            reverse=True
        )
        self.current_focus = candidates[0]
        print(f"[MACRO-AGENT] New target: '{self.current_focus}' "
              f"(weight {self.skill_weights[self.current_focus]}, "
              f"uncertainty {self.rating.get_uncertainty(self.current_focus):.0f}%).")
        return self.current_focus

    def register_question_asked(self, topic):
        """Caller invokes this whenever a question (incl. a clarification) is delivered."""
        if topic in self.questions_asked:
            self.questions_asked[topic] += 1
