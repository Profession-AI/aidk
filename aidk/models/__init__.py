"""
Models are the core of AIDK. They are responsible for executing prompts and returning responses.

This package uses lazy loading to avoid importing heavy optional dependencies
at module import time. Classes are imported only when accessed.
"""

from .model import Model
from .hosted_model import HostedModel
from .multi_model import MultiModel
from .collaborative_model import CollaborativeModel
from .image_model import ImageModel
from .voice_model import VoiceModel

__all__ = ['Model', "HostedModel", 'MultiModel', 'CollaborativeModel', 'ImageModel', 'VoiceModel']

# Backwards compatibility: old name 'CollectiveModel' -> 'CollaborativeModel'
CollectiveModel = CollaborativeModel
__all__.append('CollectiveModel')
