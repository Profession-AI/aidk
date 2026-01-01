from typing import List
from .prompt import Prompt, _PromptParser

class PromptChain(Prompt):

    """
    PromptChain class for handling sequences of prompts, the output of one prompt is used as context for the next one.

    .prompt file example:
    --------
    ```
    <promptchain>
        <prompt>
            What is the capital of {country}?
        </prompt>
        <prompt>
            What is its population?
        </prompt>
    </promptchain>
    ```

    Examples
    --------
    Create a chain from a sequence of prompts:
    ```
    prompts = [
        Prompt(prompt="What is the capital of France?"),
        Prompt(prompt="What is its population?")
    ]
    chain = PromptChain(prompts=prompts)
    ```

    Create a chain from a file:
    ```
    chain = PromptChain(
        promptchain_id="city_analysis",
        prompts_data=[{"country": "France"}]
    )
    ```
    """

    def __init__(self, 
                 promptchain_id: str = None,
                 prompts_data: list[dict] = None,
                 prompts: List[Prompt] = None,
                 response_type: type | None = None):

        """
        Initialize a new PromptChain instance.

        Parameters
        ----------
        promptchain_id : str, optional
            A .prompt file name for loading a prompt chain from file
        prompts_data : list[dict], optional
            List of dictionaries containing formatting data for each prompt
        prompts : List[Prompt], optional
            Direct list of Prompt objects to form the chain if promptchain_id is not provided

        Raises
        ------
        - ValueError
            If neither promptchain_id nor prompts is provided
        """
        if promptchain_id is not None:
            self._prompts, self._response_type = _PromptChainParser().parse(promptchain_id)
            for i in range(len(self._prompts)):
                self._prompts[i] = Prompt(prompt=self._prompts[i], prompt_data=prompts_data[i])
        elif prompts is not None:
            self._prompts = prompts
        else:
            raise ValueError("Either promptchain_id or prompts must be provided")
        self._size = len(self._prompts)
        self.response_type = response_type

    def _format(self, index: int, context: str | None = None) -> str:
        """
        Format a specific prompt in the chain with optional context.

        This method formats the prompt at the specified index, optionally including
        context from previous prompts' responses.

        Parameters
        ----------
        index : int
            Index of the prompt to format
        context : str, optional
            Context from previous prompts' responses to include

        Returns
        -------
        str
            The formatted prompt text with optional context
        """
        current_prompt = self._prompts[index]
        if context is None:
            return Prompt(prompt=current_prompt) if isinstance(current_prompt, str) else current_prompt
        else:
            return Prompt(prompt=str(current_prompt)+"\nContext: "+str(context), 
                          response_type=None if isinstance(current_prompt, str) else current_prompt.response_type)

    def __str__(self) -> str:
        """
        Get the string representation of the prompt chain.

        Returns
        -------
        str
            All prompts in the chain joined by newlines
        """
        return "\n".join([str(prompt) for prompt in self._prompts])

    def __repr__(self) -> str:
        """
        Get the official string representation of the prompt chain.

        Returns
        -------
        str
            All prompts in the chain joined by newlines
        """
        return self.__str__()


class _PromptChainParser(_PromptParser):
    def _parse(self, prompt_dict):
        prompt_dict = prompt_dict["promptchain"]
        return prompt_dict["prompt"], prompt_dict.get("@response_type")
    
