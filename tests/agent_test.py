from aidk.tools.websearch import WebSearch
from os import environ
from aidk.models import Model
from aidk._agents import Agent

agent = Agent(provider="openai", model="gpt-4o-mini", tools=[WebSearch().get_tool()])
result = agent.run("Who is the president of the United States in 2025?")
print(result)