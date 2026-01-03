"""
This module provides classes for handling different types of prompts in the COICOI framework.
Prompt can be provided as a string or as a xml-like .prompt file.
"""

from .prompt import Prompt  
from .system_prompt import SystemPrompt

__all__ = ['Prompt', 'SystemPrompt']
