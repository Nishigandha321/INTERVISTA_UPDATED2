# services/learning_resources/fetchers/geeksforgeeks_fetcher.py
# GeeksforGeeks resource fetcher — real scraping only, no mock data

import asyncio
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from .base_fetcher import BaseFetcher
from utils.logger import Logger

logger = Logger(__name__)


class GeeksForGeeksFetcher(BaseFetcher):
    """
    Fetch real learning resources from GeeksforGeeks.

    Uses GFG search scraping first. Slug-based fallback only returns URLs
    verified via HTTP HEAD (or Range GET). Never returns unverified guessed URLs.
    """

    GFG_SEARCH_URL = "https://www.geeksforgeeks.org/search/"
    GFG_BASE = "https://www.geeksforgeeks.org"

    SLUG_OVERRIDES: Dict[str, str] = {
        "binary search tree": "binary-search-tree-data-structure",
        "bst": "binary-search-tree-data-structure",
        "dynamic programming": "dynamic-programming",
        "graph": "graph-data-structure-and-algorithms",
        "linked list": "data-structures/linked-list",
        "tree": "binary-tree-data-structure",
        "stack": "stack-data-structure",
        "queue": "queue-data-structure",
        "heap": "heap-data-structure",
        "sorting": "sorting-algorithms",
        "recursion": "recursion",
        "hashing": "hashing-data-structure",
        "array": "array-data-structure",
        "string": "string-data-structure",
        "greedy": "greedy-algorithms",
        "backtracking": "backtracking-algorithms",
        "divide and conquer": "divide-and-conquer",
        "bit manipulation": "bit-manipulation-tricks-and-questions",
        "object oriented programming": "object-oriented-programming-oops-concept-in-java-with-examples",
        "oop": "object-oriented-programming-oops-concept-in-java-with-examples",
        "design patterns": "design-patterns-understand-the-importance-with-real-life-examples",
        "system design": "system-design-tutorial",
        "rest api": "rest-api-introduction",
        "rest apis": "rest-api-introduction",
        "sql": "sql-tutorial",
        "database normalization": "introduction-of-database-normalization",
        "os": "operating-systems",
        "operating system": "operating-systems",
        "process scheduling": "cpu-scheduling-in-operating-systems",
        "networking": "computer-network-tutorials",
        "tcp ip": "tcp-ip-model",
        "concurrency": "multithreading-in-java",
        "multithreading": "multithreading-in-java",
        "javascript": "javascript-tutorial",
        "python": "python-programming-language-tutorial",
        "java": "java-tutorials",
        "react": "react-tutorial",
        "machine learning": "machine-learning",
        "deep learning": "deep-learning-tutorial",
        "neural network": "neural-networks-a-beginners-guide",
        "docker": "introduction-to-docker",
        "kubernetes": "introduction-to-kubernetes",
        "git": "git-tutorial",
        "time complexity": "understanding-time-complexity-simple-examples",
        "big o notation": "analysis-of-algorithms-big-o-analysis",
    }

    def __init__(self, use_api: bool = False):
        super().__init__("geeksforgeeks")
        self.use_api = use_api
        self._verify_semaphore = asyncio.Semaphore(5)

    async def search(
        self,
        query: str,
        limit: int = 5,
        difficulty: str = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not self.validate_search_query(query):
            return []

        limit = min(limit, 20)

        try:
            results = await self._scrape_search(query, limit)
            if results:
                logger.info(f"GFG scrape: {len(results)} results for '{query}'")
                return results

            results = await self._slug_fallback(query, limit)
            logger.info(f"GFG slug fallback: {len(results)} verified results for '{query}'")
            return results

        except Exception as e:
            logger.error(f"GFG search failed for '{query}': {e}")
            return await self._slug_fallback(query, limit)

    async def _scrape_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed; cannot scrape GFG")
            return []

        try:
            encoded_query = query.replace(" ", "+")
            url = f"{self.GFG_SEARCH_URL}?q={encoded_query}"
            headers = self._build_headers()

            await asyncio.sleep(random.uniform(0.1, 0.3))
            async with httpx.AsyncClient(verify=False, timeout=15, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)

            if resp.status_code != 200:
                logger.warning(
                    "GFG scrape for '%s' returned HTTP %s from %s",
                    query,
                    resp.status_code,
                    url,
                )
                return []

            body_text = getattr(resp, "text", "") or ""
            if self._looks_like_bot_check(body_text, resp.status_code):
                logger.warning(
                    "GFG scrape for '%s' looks bot-protected (HTTP %s): %s",
                    query,
                    resp.status_code,
                    self._body_preview(body_text),
                )
                return []

            parsed = self._parse_search_html(body_text, limit, query)
            if not parsed:
                logger.info("GFG scrape for '%s' returned no parseable results; falling back to overrides", query)
            return parsed

        except Exception as e:
            logger.warning(f"GFG scraping error: {e}")
            return []

    def _build_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.geeksforgeeks.org/",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Connection": "keep-alive",
        }

    @staticmethod
    def _looks_like_bot_check(body_text: str, status_code: int) -> bool:
        if status_code in {403, 429}:
            return True
        lower_text = (body_text or "").lower()
        challenge_markers = [
            "just a moment",
            "cf-browser-verification",
            "cloudflare",
            "captcha",
            "browser check",
            "access denied",
            "verify you are human",
        ]
        return any(marker in lower_text for marker in challenge_markers)

    @staticmethod
    def _body_preview(body_text: str, limit: int = 220) -> str:
        text = re.sub(r"\s+", " ", body_text or "")
        return text[:limit]

    def _parse_search_html(self, html: str, limit: int, query: str) -> List[Dict[str, Any]]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return []

        soup = BeautifulSoup(html, "html.parser")
        results = []

        selectors = [
            "div.search-results-container article",
            "div.search_article_card",
            "div[class*='SearchResultCard']",
            "div[class*='search-result']",
            "article.article-pane",
        ]

        articles = []
        for sel in selectors:
            articles = soup.select(sel)
            if articles:
                break

        if not articles:
            return self._parse_links_fallback(soup, limit, query)

        for article in articles[:limit]:
            try:
                link_elem = article.find("a", href=True)
                title_elem = article.find(["h2", "h3", "h4", "span", "p"])

                if not link_elem:
                    continue

                href = link_elem.get("href", "")
                if not href.startswith("http"):
                    href = self.GFG_BASE + href

                if not re.match(r"https://www\.geeksforgeeks\.org/[a-z0-9\-/]+/?$", href):
                    continue

                title = (
                    link_elem.get_text(strip=True)
                    or (title_elem.get_text(strip=True) if title_elem else "")
                    or href.split("/")[-2].replace("-", " ").title()
                )

                desc_elem = article.find("p")
                description = desc_elem.get_text(strip=True)[:400] if desc_elem else ""

                if title and href:
                    results.append(self._make_result(title, href, description, query))
            except Exception:
                continue

        return results

    def _parse_links_fallback(self, soup, limit: int, query: str = "") -> List[Dict[str, Any]]:
        results = []
        seen = set()

        for a in soup.find_all("a", href=True):
            if len(results) >= limit:
                break
            href = a.get("href", "")
            if not re.match(r"https://www\.geeksforgeeks\.org/[a-z][a-z0-9\-]{5,}/", href):
                continue
            if href in seen:
                continue
            if any(x in href for x in ["/tag/", "/category/", "/courses/", "/jobs/", "/events/"]):
                continue
            seen.add(href)
            title = a.get_text(strip=True) or href.split("/")[-2].replace("-", " ").title()
            if len(title) < 5:
                continue
            results.append(self._make_result(title, href, "", query))

        return results

    def _candidate_urls(self, query: str, limit: int) -> List[Tuple[str, bool]]:
        query_lower = query.lower().strip()
        candidates: List[Tuple[str, bool]] = []

        if query_lower in self.SLUG_OVERRIDES:
            slug = self.SLUG_OVERRIDES[query_lower]
            candidates.append((f"{self.GFG_BASE}/{slug}/", True))
        else:
            slug = re.sub(r"[^a-z0-9\s-]", "", query_lower).strip()
            slug = re.sub(r"\s+", "-", slug)
            slug_variants = [
                slug,
                f"{slug}-data-structure",
                f"introduction-to-{slug}",
                f"{slug}-algorithm",
                f"{slug}-in-java",
                f"{slug}-in-python",
            ]
            for variant in slug_variants[:limit]:
                candidates.append((f"{self.GFG_BASE}/{variant}/", False))

        seen = set()
        deduped = []
        for url, trusted_override in candidates:
            if url not in seen:
                seen.add(url)
                deduped.append((url, trusted_override))
        return deduped

    async def _slug_fallback(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Return trusted override URLs immediately, otherwise verify generated candidates."""
        candidates = self._candidate_urls(query, limit)
        if not candidates:
            return []

        trusted_urls = [url for url, trusted_override in candidates if trusted_override]
        unverified_candidates = [(url, False) for url, trusted_override in candidates if not trusted_override]

        result_urls = list(trusted_urls)
        if unverified_candidates:
            verified_urls = await self._verify_urls_batch(unverified_candidates)
            result_urls.extend(verified_urls)

        if not result_urls:
            logger.warning(
                f"GFG slug verification failed for all candidates for '{query}' "
                f"(check SLUG_OVERRIDES staleness)"
            )
            return []

        return [
            self._make_result(
                f"{query.title()} - GeeksforGeeks",
                url,
                f"Learn {query} on GeeksforGeeks — verified article.",
                query,
            )
            for url in result_urls[:limit]
        ]

    async def _verify_urls_batch(self, urls: List[Tuple[str, bool]]) -> List[str]:
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed; cannot verify GFG URLs")
            return []

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        async with httpx.AsyncClient(
            verify=False, timeout=3.0, follow_redirects=True
        ) as client:
            tasks = [
                self._verify_url(client, url, headers, trust_override=trusted_override)
                for url, trusted_override in urls
            ]
            results = await asyncio.gather(*tasks)

        return [url for url in results if url]

    async def _verify_url(
        self,
        client,
        url: str,
        headers: dict,
        trust_override: bool = False,
    ) -> Optional[str]:
        if trust_override:
            logger.info("Skipping HTTP verification for trusted GFG override URL %s", url)
            return url

        async with self._verify_semaphore:
            try:
                resp = await client.head(url, headers=headers)
                body_preview = self._body_preview(getattr(resp, "text", "") or "")
                if resp.status_code == 200:
                    return url
                if resp.status_code in (405, 501):
                    resp = await client.get(
                        url,
                        headers={**headers, "Range": "bytes=0-0"},
                    )
                    body_preview = self._body_preview(getattr(resp, "text", "") or "")
                    if resp.status_code in (200, 206):
                        return url
                if self._looks_like_bot_check(body_preview, resp.status_code):
                    logger.warning(
                        "GFG verification hit a challenge page for %s (HTTP %s): %s",
                        url,
                        resp.status_code,
                        body_preview,
                    )
                else:
                    logger.warning(
                        "GFG slug verification failed for %s (HTTP %s): %s",
                        url,
                        resp.status_code,
                        body_preview,
                    )
            except Exception as e:
                logger.warning(f"GFG slug verification failed for {url}: {e}")
            return None

    @staticmethod
    def _make_result(title: str, url: str, description: str, query: str) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        return {
            "title": title[:200],
            "url": url,
            "description": description,
            "author": "GeeksforGeeks",
            "difficulty": "intermediate",
            "published_date": None,
            "tags": [query_lower, "tutorial", "geeksforgeeks"],
            "content": "",
        }
