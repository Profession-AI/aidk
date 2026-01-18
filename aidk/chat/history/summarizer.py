"""
Chat message summarization utility.
Provides functionality to summarize chat histories using language models.
"""

from aidk.models import Model
from aidk.chat.history.models import Message


class HistorySummarizer:
    """
    Summarizes chat histories using a language model.
    
    This class takes chat message histories and generates concise summaries
    using a specified language model and provider.
    """

    def __init__(self, model: Model, max_tokens: int = None):
        """Initialize the history summarizer.
        
        Parameters
        ----------
        model : Model
            The language model to use for summarization
        max_tokens : int, optional
            Maximum number of tokens for the summary
        """
        self._model = model
        self._max_tokens = max_tokens

    def summarize(self, current_summary: str, last_messages: list):
        """Generate or update a summary of the conversation.
        
        Parameters
        ----------
        current_summary : str
            The existing summary to build upon
        last_messages : list
            List of recent messages to summarize
            
        Returns
        -------
        str
            The updated summary
        """

        response = self._model.ask(
            "Summarize the following conversation: " + 
            current_summary + "\n" + 
            "\n".join([msg.content for msg in last_messages])
        )
        
        response = response.response
        return response
