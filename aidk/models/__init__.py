"""
Models are the core of AIDK. They are responsible for executing prompts and returning responses.

This package uses lazy loading to avoid importing heavy optional dependencies
at module import time. Classes are imported only when accessed.
"""

from .model import Model
from .hosted_model import HostedModel
from .image_model import ImageModel
from .voice_model import VoiceModel
from ._response_processor import (
    ModelResponse,
    ModelUsage,
    ModelStreamHead,
    ModelStreamChunk,
    ModelStreamTail,
)

__all__ = [
    'Model',
    'HostedModel',
    'ImageModel',
    'VoiceModel',
    'ModelResponse',
    'ModelUsage',
    'ModelStreamHead',
    'ModelStreamChunk',
    'ModelStreamTail',
]