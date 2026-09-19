"""Central type definitions for promptise.

This module provides type aliases and definitions used across the codebase,
particularly for orchestration and agent integration.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

# Import the actual definitions from their source modules
from .config import ServerSpec
from .cross_agent import CrossAgent

# Model type alias - can be a provider string, a chat model instance, or a Runnable
from .models import Model

ModelLike = str | Model | BaseChatModel | Runnable[Any, Any]

__all__ = [
    "ServerSpec",
    "CrossAgent",
    "ModelLike",
]
