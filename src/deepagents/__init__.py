"""DeepAgents: Advanced agent framework with middleware-based architecture.

DeepAgents provides a flexible framework for building AI agents with rich capabilities
including planning, filesystem operations, and hierarchical subagent delegation.

Main exports:
    - create_deep_agent: Create a synchronous deep agent
    - async_create_deep_agent: Create an asynchronous deep agent
    - PlanningMiddleware: Middleware for todo/planning capabilities
    - FilesystemMiddleware: Middleware for file operations
    - SubAgentMiddleware: Middleware for subagent delegation
    - DeepAgentState: Full state schema for deep agents
    - SubAgent: Type definition for subagent configuration
    - CustomSubAgent: Type definition for custom graph-based subagents
    - get_default_model: Get the default language model
"""

from deepagents.graph import create_deep_agent, async_create_deep_agent
from deepagents.middleware import PlanningMiddleware, FilesystemMiddleware, SubAgentMiddleware
from deepagents.state import DeepAgentState
from deepagents.types import SubAgent, CustomSubAgent
from deepagents.model import get_default_model
