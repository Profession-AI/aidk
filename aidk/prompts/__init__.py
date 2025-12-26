"""
This module provides classes for handling different types of prompts in the COICOI framework.
Prompt can be provided as a string or as a xml-like .prompt file.
"""

from .prompt_chain import PromptChain
from .prompt import Prompt  
from .iterative_prompt import IterativePrompt
from .system_prompt import SystemPrompt

__all__ = ['Prompt', 'PromptChain', 'IterativePrompt', 'SystemPrompt']
