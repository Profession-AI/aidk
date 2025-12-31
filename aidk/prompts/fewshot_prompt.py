from typing import List

class FewShotPrompt:

    def __init__(self, prompt: str, examples: List[str]):
        self._prompt = prompt
        self._examples = examples

    def _format(self) -> str:
        return self._prompt + "\n\n" + \
            "Use the information from the following examples:" + "\n" + \
            "\n".join(self._examples)
