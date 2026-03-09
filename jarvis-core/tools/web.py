import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus


class WebTool:
    def __init__(self, brave_api_key: str | None = None):
        self._brave_key = brave_api_key

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        try:
            if self._brave_key:
                return self._brave_search(query, num_results)
            return self._ddg_search(query, num_results)
        except Exception as e:
            return [{"title": "", "url": "", "snippet": "", "error": str(e)}]

    def _brave_search(self, query: str, num_results: int) -> list[dict]:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"Accept": "application/json", "X-Subscription-Token": self._brave_key}
        resp = requests.get(url, params={"q": query, "count": num_results}, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("web", {}).get("results", [])
        return [{"title": r.get("title"), "url": r.get("url"), "snippet": r.get("description")} for r in results]

    def _ddg_search(self, query: str, num_results: int) -> list[dict]:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for title_el in soup.select(".result__title")[:num_results]:
            link = title_el.find("a")
            snippet_el = title_el.find_next(".result__snippet")
            results.append({
                "title": link.get_text(strip=True) if link else "",
                "url": link.get("href", "") if link else "",
                "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
            })
        return results

    def fetch_page(self, url: str, max_chars: int = 4000) -> dict:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return {"text": text[:max_chars], "error": None}
        except Exception as e:
            return {"text": None, "error": str(e)}
