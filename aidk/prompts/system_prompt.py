from .prompt import Prompt

class SystemPrompt(Prompt):
    """
    A specialized Prompt class that is always a system prompt.
    
    This class extends Prompt but automatically sets is_system=True,
    making it convenient for creating system prompts without having
    to specify the is_system parameter every time.
    """

    def __init__(self, **kwargs):
        """
        Initialize a new SystemPrompt instance.
        
        All parameters are passed directly to the parent Prompt class,
        with is_system automatically set to True.
        
        Parameters
        ----------
        **kwargs
            All parameters accepted by the parent Prompt class:
            - prompt_id: str, optional - A .prompt file name
            - prompt_data: dict, optional - Data for prompt formatting
            - prompt: str, optional - Direct prompt text
            - response_type: type, optional - Expected response type
        """
        # Always set is_system=True, but allow it to be overridden if needed
        kwargs.setdefault('is_system', True)
        super().__init__(**kwargs)