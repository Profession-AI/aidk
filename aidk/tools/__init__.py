"""
Tools are used to extend the capabilities of an agent.

In AIDK tools are simple functions that return a string, you can easily create your own tools by defining a function and registering it with the agent.
To let the agent know how to use the tool, you need to add a docstring to the function in google format.

For example:
```python
def get_weather(location:str):
    \"\"\"
    Get the weather for a given location:
    Args:
        location: The location to get the weather for
    Returns:
        The weather for the given location
    \"\"\"

    return f"The weather for {location} is sunny"
```
"""

from .domain_whois import domain_whois
from .webscraping import scrape_web_with_requests, scrape_web_with_selenium, scrape_web_with_tavily, scrape_web
from .websearch import search_web_with_duckduckgo, search_web_with_tavily, WebSearch

# Backwards-compatible alias used by older code/tests
search_web = search_web_with_duckduckgo

__all__ = [
    "domain_whois",
    "search_web_with_duckduckgo",
    "search_web_with_tavily",
    "search_web",
    "WebSearch",
    "scrape_web_with_requests",
    "scrape_web_with_selenium",
    "scrape_web_with_tavily",
    "scrape_web",
]