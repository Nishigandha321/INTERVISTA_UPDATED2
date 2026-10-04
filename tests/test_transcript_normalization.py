import unittest

from speech.transcription import normalize_candidate_transcript, normalize_transcript


TECH_QUESTION = "What technologies did you use in your project?"


class TranscriptNormalizationTests(unittest.TestCase):
    def normalize(self, text, context=TECH_QUESTION, interview_type="technical"):
        return normalize_candidate_transcript(text, interview_type, context)

    def test_env_spelling_is_normalized_only_in_technical_context(self):
        result = self.normalize("We kept secrets in D-O-T E-N-V during local development.")
        self.assertIn(".env", result["normalized_transcript"])
        self.assertTrue(result["normalization_applied"])
        self.assertIn("D-O-T E-N-V", result["raw_transcript"])

    def test_common_technical_aliases(self):
        self.assertIn("C++", self.normalize("I used C plus plus.")["normalized_transcript"])
        self.assertIn("Node.js", self.normalize("We used node JS.")["normalized_transcript"])
        self.assertIn("PostgreSQL", self.normalize("We used post gres.")["normalized_transcript"])

    def test_technology_and_existing_product_names_are_not_rewritten(self):
        self.assertEqual(self.normalize("Technology was the main constraint.")["normalized_transcript"], "Technology was the main constraint.")
        self.assertIn("FastAPI", self.normalize("I used FastAPI.")["normalized_transcript"])

    def test_unfamiliar_entity_is_flagged_for_clarification_not_rewritten(self):
        result = self.normalize("I used Netaji.")
        self.assertEqual(result["suspicious_term"], "Netaji")
        self.assertEqual(result["normalized_transcript"], "I used Netaji.")

    def test_hr_answer_is_not_technically_rewritten_or_flagged(self):
        answer = "I used C plus plus when I was nervous, but I learned to ask for help."
        result = self.normalize(answer, "Tell me about a time you faced a challenge.", "hr")
        self.assertEqual(result["normalized_transcript"], answer)
        self.assertFalse(result["normalization_applied"])
        self.assertEqual(result["suspicious_term"], "")

    def test_legacy_cleanup_preserves_answer_case(self):
        self.assertEqual(normalize_transcript("I use FastAPI."), "I use FastAPI.")


if __name__ == "__main__":
    unittest.main()
