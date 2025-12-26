from aidk.models import CollectiveModel
from aidk.prompts.prompt_chain import PromptChain

collective_model = CollectiveModel(
    models=[
        {
            "provider": "openai",
            "model": "gpt-4o-mini"
        },
        {
            "provider": "openai",
            "model": "gpt-4o-mini"
        }
    ],
    aggregator={
        "provider": "openai",
        "model": "gpt-4o-mini"
    }
)

chain = PromptChain([
    "What is the capital of France?",
    "Given the city name, tell me its population."
])

result = collective_model.ask(chain)
print(result)
