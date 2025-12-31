from typing import List
from .prompt import Prompt, _PromptParser

class IterativePrompt(Prompt):
    """
    IterativePrompt class for handling prompts that iterate over a sequence of data, with optional memory
    of previous iterations' responses. This allow for the generation of longer and structured responses.

    .prompt file example:
    --------
    ```
    <iterativeprompt>
        <prompt>
            Generate the content of a chapter of a book about {topic}
            The chapters are {chapters}. Generate the chapter {{data}}.
        </prompt>
        <prompt_memory>
            Be sure that the chapter is coherent with the previous chapters, this is the content of the previous chapters:
            {{data}}
        </prompt_memory>
    </iterativeprompt>
    ```

    Examples
    --------
    
    Iterative prompt with memory:
    ```
    data = ["data types", "conditional statements", "iterative statements"]
    prompt = IterativePrompt(
        prompt="Generate the content of a chapter of a book about {topic}. The chapters are {chapters}. Generate the chapter {{data}}.",
        iter_data=data,
        prompt_memory="Be sure that the chapter is coherent with the previous chapters, this is the content of the previous chapters: {data}"
    )
    ```
    Iterative prompt with memory from a .prompt file:
    ```
    data = ["data types", "conditional statements", "iterative statements"]
    prompt = IterativePrompt(
        prompt_id="book_generation",
        prompt_data={"topic": "python programming", "chapters": data},
        iter_data=data
    )
    ```
    """

    def __init__(self, 
                 prompt_id: str = None,
                 prompt: str = None,
                 prompt_data: dict = None,
                 iter_data: List[str] = None, 
                 prompt_memory: str = "",
                 retain_all: bool = False):
        """
        Initialize a new IterativePrompt instance.

        Parameters
        ----------
        prompt_id : str, optional
            .prompt file name for loading a prompt from file
        prompt : str, optional
            Direct prompt text with {{data}} placeholder if prompt_id is not provided
        prompt_data : dict, optional
            Dictionary of values for formatting the base prompt
        iter_data : List[str], optional
            Sequence of data items to iterate over
        prompt_memory : str, optional
            Template for including memory of previous iterations
        retain_all : bool, optional
            If True, all responses are retained in memory, otherwise only the last response is retained

        Raises
        ------
        ValueError
            If neither prompt_id nor prompt is provided
        """
        if prompt_id is not None:
            self._prompt, prompt_memory = _IterativePromptParser().parse(prompt_id)
        elif prompt is not None:
            self._prompt = prompt
        else:       
            raise ValueError("Either prompt_id or prompt must be provided")
        
        if prompt_data is not None:
            self._prompt = self._prompt.format(**prompt_data)
            
        self._iter_data = iter_data
        self._size = len(iter_data)
        self._prompt_memory = prompt_memory.replace("{{", "{").replace("}}", "}")
        self._has_memory = prompt_memory != "" 
        self._retain_all = retain_all        

    def _format(self, index: int, context: str = "") -> str:
        """
        Format the prompt for a specific iteration with optional context.

        This method formats the prompt for the data item at the specified index,
        optionally including context from previous iterations if memory is enabled.

        Parameters
        ----------
        index : int
            Index of the current data item
        context : str, optional
            Context from previous iterations, default ""

        Returns
        -------
        str
            The formatted prompt text with current data and optional memory

        Examples
        --------
        Format without memory:
            >>> prompt = IterativePrompt(
            ...     prompt="Analyze {data}",
            ...     iter_data=["item1", "item2"]
            ... )
            >>> formatted = prompt.format(0)

        Format with memory:
            >>> prompt = IterativePrompt(
            ...     prompt="Compare {data}",
            ...     iter_data=["item1", "item2"],
            ...     prompt_memory="Previous: {data}"
            ... )
            >>> formatted = prompt.format(1, "Analysis of item1")
        """
        prompt = self._prompt.format(data=self._iter_data[index])
        if self._has_memory and index > 0:
            prompt += "\n\n" + self._prompt_memory.format(data=context)
        return prompt

    def __str__(self) -> str:
        """
        Get the string representation of the prompt.

        Returns
        -------
        str
            The base prompt template
        """
        return self._prompt

    def __repr__(self) -> str:
        """
        Get the official string representation of the prompt.

        Returns
        -------
        str
            The base prompt template
        """
        return self.__str__()


class _IterativePromptParser(_PromptParser):
    def _parse(self, prompt_dict):
        prompt_dict = prompt_dict["iterativeprompt"]
        return prompt_dict["prompt"], prompt_dict.get("prompt_memory")
    
    
if __name__ == "__main__":
    parser = _IterativePromptParser()
    print(parser.parse("test_iter"))
