def search_web_with_duckduckgo(query: str, max_results: int = 10, exclude_domains: list[str] = None):
    """Search the web using DuckDuckGo search engine.
    
    Args:
        query: The query to search for
        max_results: The maximum number of results to return. Default is 5
        exclude_domains: The domains to exclude from the search. Default is None
    
    Returns:
        A dictionary containing:
            data: The search results as a list of dictionaries
            text: The results merged into a single string        
    """
    search_engine = _DuckDuckGoSearch(max_results, exclude_domains)
    response, text_response = search_engine.search(query)
    return {"data": response, "text": text_response}


def search_web_with_tavily(query: str, max_results: int = 10, exclude_domains: list[str] = None):
    """Search the web using Tavily search engine.
    
    Args:
        query: The query to search for
        max_results: The maximum number of results to return. Default is 5
        exclude_domains: The domains to exclude from the search. Default is None
    
    Returns:
        A dictionary containing:
            data: The search results as a list of dictionaries
            text: The results merged into a single string        
    """
    search_engine = _TavilySearch(max_results, exclude_domains)
    response, text_response = search_engine.search(query)
    return {"data": response, "text": text_response}


class _BaseSearch():

    def __init__(self, max_results: int = 5, exclude_domains: list[str] = None):
        self._max_results = max_results
        self._exclude_domains = exclude_domains

    def search(self, query: str):
        pass

    def _post_process(self, response: list[dict], title_key: str = "title", text_key: str = "body", url_key: str = "url"):
        response = [{"title": item[title_key], "text": item[text_key], "url": item[url_key]} for item in response]
        text_response = "\n\n".join([item["title"] + "\n\n" + item["text"] for item in response])
        return response, text_response

class _DuckDuckGoSearch(_BaseSearch):

    def __init__(self, max_results: int = 5, exclude_domains: list[str] = None):
        super().__init__(max_results, exclude_domains)

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError("duckduckgo-search is not installed. Please install it with 'pip install duckduckgo-search'")


        self._client = DDGS()

    def search(self, query: str):
        response = self._client.text(query, max_results=self._max_results)
        return self._post_process(response, title_key="title", text_key="body", url_key="href")
    

from aidk.keys.keys_manager import load_key

class _TavilySearch(_BaseSearch):

    def __init__(self, max_results: int = 5, exclude_domains: list[str] = None):
        super().__init__(max_results, exclude_domains)

        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError("tavily is not installed. Please install it with 'pip install tavily-python'")

        load_key("tavily")
        self._client = TavilyClient()

    def search(self, query: str):
        response = self._client.search(query, max_results=self._max_results, exclude_domains=self._exclude_domains)
        response = response["results"]
        return self._post_process(response, title_key="title", text_key="content", url_key="url")


class WebSearch:
    """Compatibility wrapper that exposes a get_tool() method like older API."""
    def __init__(self, engine: str = "duckduckgo", max_results: int = 10, exclude_domains: list[str] = None):
        self.engine = engine
        self.max_results = max_results
        self.exclude_domains = exclude_domains

    def get_tool(self):
        if self.engine == "tavily":
            def tool(query: str):
                return search_web_with_tavily(query, max_results=self.max_results, exclude_domains=self.exclude_domains)
        else:
            def tool(query: str):
                return search_web_with_duckduckgo(query, max_results=self.max_results, exclude_domains=self.exclude_domains)
        return tool
