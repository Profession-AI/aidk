"""Compatibility shim for older import paths.

Historically tests imported from aidk.agents.agentic_loop; newer layout uses
`agentic_loops` package. Re-export commonly used classes for backwards
compatibility.
"""
from .agentic_loops._other_agentic_loops import (
    ProgrammaticAgenticLoop,
    PlanAndExecuteAgenticLoop,
    ReflexionAgenticLoop,
    SelfAskAgenticLoop,
    SelfAskWithSearchLoop,
)

from .agentic_loops._function_calling_agentic_loop import FunctionCallingAgenticLoop
from .agentic_loops._react_agentic_loop import ReactAgenticLoop, ReactWithFCAgenticLoop

__all__ = [
    "ProgrammaticAgenticLoop",
    "PlanAndExecuteAgenticLoop",
    "ReflexionAgenticLoop",
    "SelfAskAgenticLoop",
    "SelfAskWithSearchLoop",
    "FunctionCallingAgenticLoop",
    "ReactAgenticLoop",
    "ReactWithFCAgenticLoop",
]
