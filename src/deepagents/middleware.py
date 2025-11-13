"""DeepAgents implemented as Middleware.

This module defines the core middleware components that extend agent capabilities
with planning, filesystem access, and subagent management features.
"""

from typing import Annotated, Optional, Any
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, SummarizationMiddleware
from langchain.agents.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_core.language_models import LanguageModelLike
from langchain.chat_models import init_chat_model
from langgraph.types import Command
from langchain.agents.tool_node import InjectedState
from deepagents.state import PlanningState, FilesystemState
from deepagents.tools import write_todos, ls, read_file, write_file, edit_file
from deepagents.prompts import WRITE_TODOS_SYSTEM_PROMPT, TASK_SYSTEM_PROMPT, FILESYSTEM_SYSTEM_PROMPT, TASK_TOOL_DESCRIPTION, BASE_AGENT_PROMPT
from deepagents.types import SubAgent, CustomSubAgent

###########################
# Planning Middleware
###########################

class PlanningMiddleware(AgentMiddleware):
    """Middleware that adds todo/planning capabilities to agents.

    This middleware extends agents with the ability to create and manage todo lists
    for tracking complex multi-step tasks. It adds the write_todos tool and injects
    planning-specific instructions into the system prompt.

    Attributes:
        state_schema: PlanningState schema for todo tracking.
        tools: List containing the write_todos tool.
    """
    state_schema = PlanningState
    tools = [write_todos]

    def modify_model_request(self, request: ModelRequest, agent_state: PlanningState) -> ModelRequest:
        """Modify the model request to include planning instructions.

        Args:
            request: The model request to modify.
            agent_state: The current planning state.

        Returns:
            The modified model request with planning instructions appended.
        """
        request.system_prompt = request.system_prompt + "\n\n" + WRITE_TODOS_SYSTEM_PROMPT
        return request

###########################
# Filesystem Middleware
###########################

class FilesystemMiddleware(AgentMiddleware):
    """Middleware that adds filesystem capabilities to agents.

    This middleware extends agents with the ability to interact with a mock filesystem,
    including listing, reading, writing, and editing files. It adds filesystem tools
    and injects filesystem-specific instructions into the system prompt.

    Attributes:
        state_schema: FilesystemState schema for file management.
        tools: List of filesystem tools (ls, read_file, write_file, edit_file).
    """
    state_schema = FilesystemState
    tools = [ls, read_file, write_file, edit_file]

    def modify_model_request(self, request: ModelRequest, agent_state: FilesystemState) -> ModelRequest:
        """Modify the model request to include filesystem instructions.

        Args:
            request: The model request to modify.
            agent_state: The current filesystem state.

        Returns:
            The modified model request with filesystem instructions appended.
        """
        request.system_prompt = request.system_prompt + "\n\n" + FILESYSTEM_SYSTEM_PROMPT
        return request

###########################
# SubAgent Middleware
###########################

class SubAgentMiddleware(AgentMiddleware):
    """Middleware that adds subagent delegation capabilities to agents.

    This middleware extends agents with the ability to spawn and manage subagents
    for handling isolated, complex tasks. It creates a task tool that allows the
    main agent to delegate work to specialized subagents with their own contexts.

    The middleware supports both synchronous and asynchronous subagent execution,
    and allows for custom subagent configurations with different tools, models,
    and middleware stacks.
    """
    def __init__(
        self,
        default_subagent_tools: list[BaseTool] = [],
        subagents: list[SubAgent | CustomSubAgent] = [],
        model: Optional[LanguageModelLike] = None,
        is_async: bool = False,
    ) -> None:
        """Initialize the SubAgentMiddleware.

        Args:
            default_subagent_tools: Tools available to the default general-purpose subagent.
            subagents: List of custom subagent configurations.
            model: Language model to use for subagents. If None, subagents must specify
                their own models.
            is_async: Whether to create async or sync task tools.
        """
        super().__init__()
        task_tool = create_task_tool(
            default_subagent_tools=default_subagent_tools,
            subagents=subagents,
            model=model,
            is_async=is_async,
        )
        self.tools = [task_tool]

    def modify_model_request(self, request: ModelRequest, agent_state: AgentState) -> ModelRequest:
        """Modify the model request to include subagent task instructions.

        Args:
            request: The model request to modify.
            agent_state: The current agent state.

        Returns:
            The modified model request with task tool instructions appended.
        """
        request.system_prompt = request.system_prompt + "\n\n" + TASK_SYSTEM_PROMPT
        return request

def _get_agents(
    default_subagent_tools: list[BaseTool],
    subagents: list[SubAgent | CustomSubAgent],
    model: Optional[LanguageModelLike]
) -> dict[str, Any]:
    """Create a dictionary of agent instances for subagent delegation.

    This function creates a general-purpose agent and any custom subagents specified
    in the configuration. Each agent is configured with its own tools, model, and
    middleware stack.

    Args:
        default_subagent_tools: Tools to provide to the default general-purpose agent.
        subagents: List of custom subagent configurations to create.
        model: Language model to use as default for agents.

    Returns:
        Dictionary mapping agent names to their configured agent instances.
    """
    default_subagent_middleware = [
        PlanningMiddleware(),
        FilesystemMiddleware(),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=120000,
            messages_to_keep=20,
        ),
        AnthropicPromptCachingMiddleware(ttl="5m", unsupported_model_behavior="ignore"),
    ]
    agents = {
        "general-purpose": create_agent(
            model,
            prompt=BASE_AGENT_PROMPT,
            tools=default_subagent_tools,
            checkpointer=False,
            middleware=default_subagent_middleware
        )
    }
    for _agent in subagents:
        if "graph" in _agent:
            agents[_agent["name"]] = _agent["graph"]
            continue
        if "tools" in _agent:
            _tools = _agent["tools"]
        else:
            _tools = default_subagent_tools.copy()
        # Resolve per-subagent model: can be instance or dict
        if "model" in _agent:
            agent_model = _agent["model"]
            if isinstance(agent_model, dict):
                # Dictionary settings - create model from config
                sub_model = init_chat_model(**agent_model)
            else:
                # Model instance - use directly
                sub_model = agent_model
        else:
            # Fallback to main model
            sub_model = model
        if "middleware" in _agent:
            _middleware = [*default_subagent_middleware, *_agent["middleware"]]
        else:
            _middleware = default_subagent_middleware
        agents[_agent["name"]] = create_agent(
            sub_model,
            prompt=_agent["prompt"],
            tools=_tools,
            middleware=_middleware,
            checkpointer=False,
        )
    return agents


def _get_subagent_description(subagents: list[SubAgent | CustomSubAgent]) -> list[str]:
    """Generate formatted descriptions of subagents for tool documentation.

    Args:
        subagents: List of subagent configurations.

    Returns:
        List of formatted strings describing each subagent's name and purpose.
    """
    return [f"- {_agent['name']}: {_agent['description']}" for _agent in subagents]


def create_task_tool(
    default_subagent_tools: list[BaseTool],
    subagents: list[SubAgent | CustomSubAgent],
    model: Optional[LanguageModelLike],
    is_async: bool = False,
) -> BaseTool:
    """Create a task tool for delegating work to subagents.

    This function creates either a synchronous or asynchronous task tool that allows
    the main agent to spawn subagents for handling isolated tasks. The tool manages
    subagent lifecycle, state updates, and result integration.

    Args:
        default_subagent_tools: Tools available to the default general-purpose subagent.
        subagents: List of custom subagent configurations.
        model: Language model to use for subagents.
        is_async: If True, creates an async task tool; otherwise creates sync version.

    Returns:
        A configured task tool (either async or sync) for subagent delegation.
    """
    agents = _get_agents(
        default_subagent_tools, subagents, model
    )
    other_agents_string = _get_subagent_description(subagents)

    if is_async:
        @tool(
            description=TASK_TOOL_DESCRIPTION.format(other_agents=other_agents_string)
        )
        async def task(
            description: str,
            subagent_type: str,
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Async task tool for delegating work to a subagent.

            Args:
                description: Detailed task description for the subagent.
                subagent_type: Type of subagent to invoke (e.g., 'general-purpose').
                state: Current agent state (injected).
                tool_call_id: Tool call identifier (injected).

            Returns:
                Command with state updates from the subagent execution.
            """
            if subagent_type not in agents:
                return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
            sub_agent = agents[subagent_type]
            state["messages"] = [{"role": "user", "content": description}]
            result = await sub_agent.ainvoke(state)
            state_update = {}
            for k, v in result.items():
                if k not in ["todos", "messages"]:
                    state_update[k] = v
            return Command(
                update={
                    **state_update,
                    "messages": [
                        ToolMessage(
                            result["messages"][-1].content, tool_call_id=tool_call_id
                        )
                    ],
                }
            )
    else:
        @tool(
            description=TASK_TOOL_DESCRIPTION.format(other_agents=other_agents_string)
        )
        def task(
            description: str,
            subagent_type: str,
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Sync task tool for delegating work to a subagent.

            Args:
                description: Detailed task description for the subagent.
                subagent_type: Type of subagent to invoke (e.g., 'general-purpose').
                state: Current agent state (injected).
                tool_call_id: Tool call identifier (injected).

            Returns:
                Command with state updates from the subagent execution.
            """
            if subagent_type not in agents:
                return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
            sub_agent = agents[subagent_type]
            state["messages"] = [{"role": "user", "content": description}]
            result = sub_agent.invoke(state)
            state_update = {}
            for k, v in result.items():
                if k not in ["todos", "messages"]:
                    state_update[k] = v
            return Command(
                update={
                    **state_update,
                    "messages": [
                        ToolMessage(
                            result["messages"][-1].content, tool_call_id=tool_call_id
                        )
                    ],
                }
            )
    return task
