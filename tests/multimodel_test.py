from aidk.models import MultiModel
from aidk.prompts.prompt_chain import PromptChain

multi_model = MultiModel(
    models=[
        {
            "provider": "openai",
            "model": "gpt-4o-mini"
        },
        {
            "provider": "openai",
            "model": "gpt-4o-mini"
        }
    ]
)

chain = PromptChain([
    "What is the capital of France?",
    "Given the city name, tell me its population."
])

result = multi_model.ask(chain)
print(result)
