import unittest

from services.interview.conversation_manager import (
    analyze_answer,
    choose_focus_keyword,
    fallback_question,
    question_is_duplicate,
    select_action,
    validate_generated_question,
    validate_candidate_answer,
)


class ConversationDecisionTests(unittest.TestCase):
    def decide(self, answer, *, probes=0, turns=1, main=1, category="technical"):
        analysis = analyze_answer("Describe your design", answer, category)
        action = select_action(analysis, probes, 2, turns, 12, main, 5, category)
        return analysis, action

    def test_project_detail_drives_grounded_followup_focus(self):
        answer = "For SplitApp, I built a FastAPI backend with PostgreSQL and JWT authentication."
        analysis, action = self.decide(answer)
        self.assertIn(action, {"DEEPEN", "CHALLENGE"})
        self.assertEqual(choose_focus_keyword(answer, analysis), "JWT")

    def test_weak_answer_requests_clarification(self):
        _, action = self.decide("I do not know")
        self.assertEqual(action, "CLARIFICATION")

    def test_explicit_skill_gap_moves_to_another_topic(self):
        analysis = analyze_answer(
            "How have you used Java?", "I don't know Java, but I use Python for projects.", "technical"
        )
        action = select_action(analysis, 0, 2, 2, 12, 2, 5, "technical")
        self.assertTrue(analysis["explicit_knowledge_gap"])
        self.assertIn("Java", analysis["knowledge_gap_topics"])
        self.assertNotIn("Python", analysis["knowledge_gap_topics"])
        self.assertEqual(action, "CHANGE_TOPIC")

    def test_generated_question_cannot_return_to_declared_gap(self):
        with self.assertRaisesRegex(ValueError, "knowledge gap"):
            validate_generated_question(
                "How would you use Java for this service?", "skill-based", [],
                "CHANGE_TOPIC", current_category="technical", excluded_topics=["Java"],
            )

    def test_unanswered_skip_moves_on(self):
        _, action = self.decide("(skipped)")
        self.assertEqual(action, "MOVE_ON")

    def test_contradictory_claim_is_clarified(self):
        analysis, action = self.decide("It always works, however it never works under load.")
        self.assertTrue(analysis["contradiction_signals"])
        self.assertEqual(action, "CLARIFICATION")

    def test_questionable_jwt_claim_is_clarified(self):
        analysis, action = self.decide("A JWT is encrypted, so its payload is always secret.")
        self.assertTrue(analysis["technically_questionable"])
        self.assertEqual(action, "CLARIFICATION")

    def test_strong_answer_receives_challenge(self):
        answer = (
            "I designed a reliable FastAPI service using PostgreSQL transactions and Redis caching. "
            "I chose the cache after profiling repeated reads, measured a 40 percent latency reduction, "
            "and kept writes transactional to avoid stale authorization decisions."
        )
        analysis, action = self.decide(answer)
        self.assertGreaterEqual(analysis["answer_quality"], 0.72)
        self.assertEqual(action, "CHALLENGE")

    def test_behavioral_answer_missing_outcome_gets_probe(self):
        _, action = self.decide("I worked with my teammate to resolve a disagreement.", category="behavioral")
        self.assertEqual(action, "BEHAVIORAL_PROBE")

    def test_topic_changes_after_probe_limit(self):
        analysis = analyze_answer("Explain the design", "I used JWT for access control.", "technical")
        action = select_action(analysis, 2, 2, 4, 12, 2, 5, "technical")
        self.assertEqual(action, "CHANGE_TOPIC")

    def test_turn_limit_ends_interview(self):
        analysis = analyze_answer("Explain the design", "JWT secures protected endpoints.", "technical")
        action = select_action(analysis, 0, 2, 12, 12, 4, 5, "technical")
        self.assertEqual(action, "END_INTERVIEW")

    def test_skip_on_last_main_question_cannot_exceed_main_limit(self):
        analysis = analyze_answer("Explain the design", "(skipped)", "technical")
        action = select_action(analysis, 0, 2, 5, 12, 5, 5, "technical")
        self.assertEqual(action, "END_INTERVIEW")

    def test_duplicate_detector_rejects_near_paraphrase(self):
        old = "Why did you choose FastAPI for this service?"
        new = "What made you select FastAPI for this service?"
        self.assertTrue(question_is_duplicate(new, [old]))

    def test_fallback_question_uses_candidate_keyword(self):
        question = fallback_question("DEEPEN", "JWT", [])
        self.assertIn("JWT", question)

    def test_generated_question_requires_answer_grounding(self):
        with self.assertRaises(ValueError):
            validate_generated_question(
                "What is your approach to authentication?", "technical", [], "DEEPEN", "JWT", "technical"
            )

    def test_generated_question_requires_new_category_for_topic_change(self):
        with self.assertRaises(ValueError):
            validate_generated_question(
                "How would you scale this service?", "technical", [], "CHANGE_TOPIC", current_category="technical"
            )

    def test_generated_question_rejects_multiple_questions(self):
        with self.assertRaises(ValueError):
            validate_generated_question(
                "How did FastAPI help your project? What else did you consider?",
                "project-specific", [], "MOVE_ON", current_category="technical"
            )

    def test_audience_noun_is_not_a_candidate_technology(self):
        answer = "Customers needed a faster checkout, so we changed the flow."
        analysis = analyze_answer("Describe a project", answer, "technical")
        self.assertNotIn("Customers", analysis["concepts_mentioned"])
        self.assertNotEqual(choose_focus_keyword(answer, analysis), "Customers")

    def test_unknown_capitalized_asr_word_is_not_a_technical_focus(self):
        question = "Describe your project."
        answer = "I used Netaji and it helped the team."
        analysis = analyze_answer(question, answer, "technical")
        self.assertNotIn("Netaji", analysis["concepts_mentioned"])
        self.assertEqual(choose_focus_keyword(answer, analysis, question), "your approach")

    def test_indirect_answer_uses_real_concept_from_current_question(self):
        question = "How did you use Java in the service?"
        answer = "I used it for the API layer."
        analysis = analyze_answer(question, answer, "technical")
        self.assertEqual(choose_focus_keyword(answer, analysis, question), "Java")

    def test_invalid_audience_and_unnamed_tradeoff_questions_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_generated_question(
                "Why did you apply customers?", "technical", [], "CHALLENGE", "customers", "technical"
            )
        with self.assertRaises(ValueError):
            validate_generated_question(
                "What drawbacks did you see in the flow?", "technical", [], "CHALLENGE", "flow", "technical"
            )

    def test_generated_question_rejects_multi_question_or_invalid_json_fields(self):
        with self.assertRaises(ValueError):
            validate_generated_question(
                "How did you do it? Why?", "technical", [], "DEEPEN", "JWT", "technical"
            )

    def test_empty_transcript_is_rejected_but_explicit_skip_is_recordable(self):
        with self.assertRaises(ValueError):
            validate_candidate_answer("   ")
        self.assertEqual(validate_candidate_answer("", skip=True), "(skipped)")


if __name__ == "__main__":
    unittest.main()
