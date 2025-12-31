from aidk.models import Model
from aidk.prompts.prompt_chain import PromptChain

model = Model(
    provider="openai",
    model="gpt-4o-mini")

chain = PromptChain([
    "What is the capital of France?",
    "Given the city name, tell me its population."
])

result = model.ask(chain)
print(result)
