# services/learning_resources/fetchers/geeksforgeeks_fetcher.py
# GeeksforGeeks resource fetcher — real scraping only, no mock data

import asyncio
import re
from urllib.parse import parse_qs, quote_plus, urlparse
from typing import List, Dict, Any, Optional, Tuple
from .base_fetcher import BaseFetcher
from utils.logger import Logger

logger = Logger(__name__)


class GeeksForGeeksFetcher(BaseFetcher):
    """
    Fetch real learning resources from GeeksforGeeks.

    Uses GFG search scraping first, then Google site search to resolve a direct
    GeeksforGeeks article URL. It never returns a generic search page.
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

            results = await self._google_article_search(query, limit)
            if not results:
                results = await self._slug_fallback(query, limit)
            logger.info(f"GFG direct-article fallback: {len(results)} results for '{query}'")
            return results

        except Exception as e:
            logger.error(f"GFG search failed for '{query}': {e}")
            results = await self._google_article_search(query, limit)
            return results or await self._slug_fallback(query, limit)

    async def _scrape_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed; cannot scrape GFG")
            return []

        try:
            encoded_query = quote_plus(query)
            url = f"{self.GFG_SEARCH_URL}?q={encoded_query}"
            headers = self._build_headers()

            async with httpx.AsyncClient(verify=False, timeout=15, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)

            if resp.status_code != 200:
                logger.warning(f"GFG scrape for '{query}' returned HTTP {resp.status_code} from {url}")
                return []

            body_text = getattr(resp, "text", "") or ""
            if self._looks_like_bot_check(body_text, resp.status_code):
                logger.warning(
                    f"GFG scrape for '{query}' looks bot-protected (HTTP {resp.status_code}): "
                    f"{self._body_preview(body_text)}"
                )
                return []

            parsed = self._parse_search_html(body_text, limit, query)
            if not parsed:
                logger.info(f"GFG scrape for '{query}' returned no parseable article results")
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

    async def _google_article_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Use Google site search and return its top direct GFG article results."""
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("httpx/BeautifulSoup unavailable; cannot resolve direct GFG search results")
            return []

        try:
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                response = await client.get(
                    "https://www.google.com/search",
                    params={"q": f"site:geeksforgeeks.org {query}", "num": min(limit, 10)},
                    headers={"User-Agent": self._build_headers()["User-Agent"]},
                )
            if response.status_code != 200:
                logger.warning("Google GFG search for '%s' returned HTTP %s", query, response.status_code)
                return []

            soup = BeautifulSoup(response.text or "", "html.parser")
            results = []
            seen = set()
            for link in soup.select("a[href]"):
                href = link.get("href", "")
                parsed = urlparse(href)
                if parsed.path == "/url" and (not parsed.netloc or parsed.netloc.casefold().endswith("google.com")):
                    params = parse_qs(parsed.query)
                    href = (params.get("q") or params.get("url") or [""])[0]
                if not href.startswith("https://"):
                    continue
                article = urlparse(href)
                article_host = article.netloc.casefold().split(":", 1)[0]
                if (
                    not (article_host == "geeksforgeeks.org" or article_host.endswith(".geeksforgeeks.org"))
                    or article.path in {"/", "/search", "/search/"}
                    or article.path.startswith("/search/")
                    or any(part in article.path for part in ("/tag/", "/category/", "/courses/", "/jobs/", "/videos/"))
                    or not re.match(r"^/[a-z0-9][a-z0-9\-/]+/?$", article.path, re.IGNORECASE)
                ):
                    continue
                direct_url = f"https://www.geeksforgeeks.org{article.path.rstrip('/')}/"
                key = direct_url.casefold()
                if key in seen:
                    continue
                seen.add(key)
                title = link.get_text(" ", strip=True)
                if not title:
                    heading = link.find(["h2", "h3"])
                    title = heading.get_text(" ", strip=True) if heading else article.path.strip("/").split("/")[-1].replace("-", " ").title()
                results.append(self._make_result(
                    title, direct_url,
                    f"GeeksforGeeks article selected from Google results for {query}.", query,
                ))
                if len(results) >= limit:
                    break
            return results
        except Exception as exc:
            logger.warning("Google GFG article search failed for '%s': %s", query, exc)
            return []

    async def _slug_fallback(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Use only a verified known article when Google search is unavailable."""
        normalized_query = " ".join(query.casefold().split())
        if normalized_query not in self.SLUG_OVERRIDES:
            return []
        candidates = self._candidate_urls(query, limit)
        if not candidates:
            return []

        result_urls = await self._verify_urls_batch([(url, False) for url, _ in candidates])

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
            logger.info(f"Skipping HTTP verification for trusted GFG override URL {url}")
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
                        f"GFG verification hit a challenge page for {url} "
                        f"(HTTP {resp.status_code}): {body_preview}"
                    )
                else:
                    logger.warning(
                        f"GFG slug verification failed for {url} "
                        f"(HTTP {resp.status_code}): {body_preview}"
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
