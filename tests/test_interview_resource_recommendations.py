import unittest
from urllib.parse import parse_qs, urlparse

from services.learning_resources.fetchers.geeksforgeeks_fetcher import GeeksForGeeksFetcher
from services.learning_resources.query_generator import SearchQueryGenerator


def immediate_result(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("Expected this deterministic async helper to return without awaiting")


class InterviewResourceRecommendationTests(unittest.TestCase):
    def test_gfg_fallback_is_a_useful_topic_search_without_network(self):
        fetcher = GeeksForGeeksFetcher()
        results = immediate_result(fetcher._slug_fallback("Java fundamentals", 3))
        self.assertEqual(len(results), 1)
        self.assertIn("geeksforgeeks.org/search/", results[0]["url"])
        self.assertEqual(parse_qs(urlparse(results[0]["url"]).query)["q"], ["Java fundamentals"])

    def test_resource_queries_are_focused_and_need_no_llm_call(self):
        generator = SearchQueryGenerator(None, None)
        queries = immediate_result(generator.generate_queries(
            "Java basics", role="PDI Analyst", count=4, company="Deloitte", difficulty="beginner"
        ))
        self.assertEqual(queries[0], "Java basics fundamentals")
        self.assertEqual(len(queries), 4)
        self.assertTrue(all("Deloitte" not in query and "PDI Analyst" not in query for query in queries))


if __name__ == "__main__":
    unittest.main()
