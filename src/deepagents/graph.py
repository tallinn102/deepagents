"""Graph construction and agent creation for DeepAgents.

This module provides the main entry points for creating deep agents with full
middleware stacks including planning, filesystem, subagent delegation, and more.
"""

from typing import Sequence, Union, Callable, Any, Type, Optional
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import Runnable
from langgraph.types import Checkpointer
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware, HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import ToolConfig
from langchain.agents.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from deepagents.middleware import PlanningMiddleware, FilesystemMiddleware, SubAgentMiddleware
from deepagents.prompts import BASE_AGENT_PROMPT
from deepagents.model import get_default_model
from deepagents.types import SubAgent, CustomSubAgent

def agent_builder(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    middleware: Optional[list[AgentMiddleware]] = None,
    tool_configs: Optional[dict[str, bool | ToolConfig]] = None,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: Optional[list[SubAgent | CustomSubAgent]] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    is_async: bool = False,
) -> Runnable:
    """Build a deep agent with a complete middleware stack.

    This internal builder function creates agents with the full DeepAgents middleware
    stack, including planning, filesystem, subagent delegation, summarization, and
    prompt caching. It serves as the foundation for both sync and async agent creation.

    Args:
        tools: Tools the agent should have access to.
        instructions: Additional instructions for the system prompt.
        middleware: Optional additional middleware to append to the stack.
        tool_configs: Optional tool interrupt configurations for human-in-the-loop.
        model: Language model to use. Defaults to Claude Sonnet 4 if None.
        subagents: Optional list of subagent configurations.
        context_schema: Optional schema for the agent's context/state.
        checkpointer: Optional checkpointer for persisting state between runs.
        is_async: Whether to create async or sync subagent tools.

    Returns:
        A configured Runnable agent instance.
    """
    if model is None:
        model = get_default_model()

    deepagent_middleware = [
        PlanningMiddleware(),
        FilesystemMiddleware(),
        SubAgentMiddleware(
            default_subagent_tools=tools,   # NOTE: These tools are piped to the general-purpose subagent.
            subagents=subagents if subagents is not None else [],
            model=model,
            is_async=is_async,
        ),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=120000,
            messages_to_keep=20,
        ),
        AnthropicPromptCachingMiddleware(ttl="5m", unsupported_model_behavior="ignore")
    ]
    # Add tool interrupt config if provided
    if tool_configs is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=tool_configs))

    if middleware is not None:
        deepagent_middleware.extend(middleware)

    return create_agent(
        model,
        prompt=instructions + "\n\n" + BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        context_schema=context_schema,
        checkpointer=checkpointer,
    )

def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]] = [],
    instructions: str = "",
    middleware: Optional[list[AgentMiddleware]] = None,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: Optional[list[SubAgent | CustomSubAgent]] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    tool_configs: Optional[dict[str, bool | ToolConfig]] = None,
) -> Runnable:
    """Create a deep agent.
    This agent will by default have access to a tool to write todos (write_todos),
    four file editing tools: write_file, ls, read_file, edit_file, and a tool to call subagents.
    Args:
        tools: The tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict settings)
                - (optional) `middleware` (list of AgentMiddleware)
        context_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        tool_configs: Optional Dict[str, HumanInTheLoopConfig] mapping tool names to interrupt configs.
    """
    return agent_builder(
        tools=tools,
        instructions=instructions,
        middleware=middleware,
        model=model,
        subagents=subagents,
        context_schema=context_schema,
        checkpointer=checkpointer,
        tool_configs=tool_configs,
        is_async=False,
    )

def async_create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]] = [],
    instructions: str = "",
    middleware: Optional[list[AgentMiddleware]] = None,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: Optional[list[SubAgent | CustomSubAgent]] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    tool_configs: Optional[dict[str, bool | ToolConfig]] = None,
) -> Runnable:
    """Create a deep agent.
    This agent will by default have access to a tool to write todos (write_todos),
    four file editing tools: write_file, ls, read_file, edit_file, and a tool to call subagents.
    Args:
        tools: The tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict settings)
                - (optional) `middleware` (list of AgentMiddleware)
        context_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        tool_configs: Optional Dict[str, HumanInTheLoopConfig] mapping tool names to interrupt configs.
    """
    return agent_builder(
        tools=tools,
        instructions=instructions,
        middleware=middleware,
        model=model,
        subagents=subagents,
        context_schema=context_schema,
        checkpointer=checkpointer,
        tool_configs=tool_configs,
        is_async=True,
    )