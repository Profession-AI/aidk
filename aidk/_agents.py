"""Compatibility module providing legacy agent classes used by older tests.

This module exposes Agent (alias), MultiAgent and CollaborativeAgent convenience
wrappers to keep older tests/imports working.
"""
from .agents.agent import Agent as _AgentClass
from .models import Model
from typing import List, Dict, Any


class Agent(_AgentClass):
    """Compatibility wrapper: allow provider/model kwargs like older API."""
    def __init__(self, provider: str | None = None, model: str | None = None, *args, **kwargs):
        # If first arg is a Model instance, pass it through
        if isinstance(provider, Model):
            super().__init__(provider, *args, **kwargs)
            return

        # Otherwise, construct a Model from provider/model strings
        m = Model(provider=provider, model=model)
        super().__init__(m, *args, **kwargs)


class MultiAgent:
    """Simple multi-model wrapper: uses the first model for execution."""
    def __init__(self, models: List[Dict[str, str]], tools: List[Any] = None):
        self._models = [Model(provider=m["provider"], model=m["model"]) for m in models]
        self._tools = tools

    def run(self, prompt):
        # use first model to run
        return self._models[0].ask(prompt)


class CollaborativeAgent(MultiAgent):
    """Alias for old collaborative agent interface."""
    pass


__all__ = ["Agent", "MultiAgent", "CollaborativeAgent"]
