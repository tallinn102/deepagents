"""
Prompts package for deepagents.

This package contains all prompts used throughout the deepagents system, organized into
logical modules for better maintainability.

All public constants are re-exported from this package for backward compatibility,
allowing imports like:
    from deepagents.prompts import TASK_TOOL_DESCRIPTION

Modules:
    - tool_descriptions: Tool-related prompts and descriptions
    - system_prompts: System-level prompts for agent configuration
"""

from deepagents.prompts.tool_descriptions import (
    WRITE_TODOS_TOOL_DESCRIPTION,
    TASK_TOOL_DESCRIPTION,
    LIST_FILES_TOOL_DESCRIPTION,
    READ_FILE_TOOL_DESCRIPTION,
    EDIT_FILE_TOOL_DESCRIPTION,
    WRITE_FILE_TOOL_DESCRIPTION,
)

from deepagents.prompts.system_prompts import (
    WRITE_TODOS_SYSTEM_PROMPT,
    TASK_SYSTEM_PROMPT,
    FILESYSTEM_SYSTEM_PROMPT,
    BASE_AGENT_PROMPT,
)

__all__ = [
    # Tool descriptions
    "WRITE_TODOS_TOOL_DESCRIPTION",
    "TASK_TOOL_DESCRIPTION",
    "LIST_FILES_TOOL_DESCRIPTION",
    "READ_FILE_TOOL_DESCRIPTION",
    "EDIT_FILE_TOOL_DESCRIPTION",
    "WRITE_FILE_TOOL_DESCRIPTION",
    # System prompts
    "WRITE_TODOS_SYSTEM_PROMPT",
    "TASK_SYSTEM_PROMPT",
    "FILESYSTEM_SYSTEM_PROMPT",
    "BASE_AGENT_PROMPT",
]
