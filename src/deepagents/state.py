"""State schemas and reducers for DeepAgents.

This module defines the state structures used throughout the DeepAgents framework,
including todo tracking, file management, and custom reducer functions.
"""

from langchain.agents.middleware import AgentState
from typing import NotRequired, Annotated
from typing import Literal
from typing_extensions import TypedDict


class Todo(TypedDict):
    """Todo to track.

    Attributes:
        content: Description of the todo item.
        status: Current status of the todo (pending, in_progress, or completed).
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(left: dict[str, str] | None, right: dict[str, str] | None) -> dict[str, str] | None:
    """Reduce two file dictionaries by merging them.

    This reducer function is used to combine file state updates. If either argument
    is None, it returns the other. Otherwise, it merges both dictionaries with the
    right dictionary taking precedence for overlapping keys.

    Args:
        left: The left-hand file dictionary or None.
        right: The right-hand file dictionary or None.

    Returns:
        The merged dictionary, or None if both arguments are None.
    """
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}


class DeepAgentState(AgentState):
    """Full state schema for deep agents.

    This state includes both todo tracking and file management capabilities,
    combining all features available to a deep agent.

    Attributes:
        todos: Optional list of Todo items for task tracking.
        files: Optional dictionary mapping file paths to their contents, with
            custom reducer for merging file updates.
    """
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]


class PlanningState(AgentState):
    """State schema for agents with todo/planning capabilities.

    This state is used by agents that need task tracking functionality.

    Attributes:
        todos: Optional list of Todo items for task tracking.
    """
    todos: NotRequired[list[Todo]]


class FilesystemState(AgentState):
    """State schema for agents with filesystem capabilities.

    This state is used by agents that need to manage and edit files.

    Attributes:
        files: Optional dictionary mapping file paths to their contents, with
            custom reducer for merging file updates.
    """
    files: Annotated[NotRequired[dict[str, str]], file_reducer]