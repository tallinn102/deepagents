# 🧠🤖Deep Agents

Using an LLM to call tools in a loop is the simplest form of an agent. 
This architecture, however, can yield agents that are “shallow” and fail to plan and act over longer, more complex tasks. 
Applications like “Deep Research”, "Manus", and “Claude Code” have gotten around this limitation by implementing a combination of four things:
a **planning tool**, **sub agents**, access to a **file system**, and a **detailed prompt**.

<img src="deep_agents.png" alt="deep agent" width="600"/>

`deepagents` is a Python package that implements these in a general purpose way so that you can easily create a Deep Agent for your application.

**Acknowledgements: This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose, and make it even more so.**

## Installation

### Basic Installation

```bash
pip install deepagents
```

### Development Installation

To install for development with all dependencies:

```bash
# Clone the repository
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### Optional Dependencies

For specific features, you may need additional packages:

```bash
# For web search capabilities (examples)
pip install tavily-python

# For MCP tool integration
pip install langchain-mcp-adapters

# For Ollama models
pip install langchain-ollama

# For other LangChain integrations
pip install langchain-openai  # OpenAI models
pip install langchain-anthropic  # Anthropic models
```

### System Requirements

- Python 3.9 or higher
- pip (latest version recommended)

## Usage

(To run the example below, will need to `pip install tavily-python`)

```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

# Initialize the Tavily client for web search
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Define a search tool that the agent can use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search using Tavily API"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# Provide custom instructions to steer the agent's behavior
# This will be combined with the built-in deep agent system prompt
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# Create the deep agent with tools and instructions
# The agent will automatically have access to:
# - Planning tools (write_todos)
# - File system tools (read_file, write_file, edit_file, ls)
# - Subagent capabilities (task tool)
agent = create_deep_agent(
    [internet_search],  # List of tools the agent can use
    research_instructions,  # Custom instructions for the agent
)

# Invoke the agent with a user message
# The agent will use its tools to research and respond
result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})

# Access the final response
print(result["messages"][-1].content)

# Access any files created during execution
if "files" in result:
    print(f"Files created: {list(result['files'].keys())}")
```

See [examples/research/research_agent.py](examples/research/research_agent.py) for a more complex example.

The agent created with `create_deep_agent` is just a LangGraph graph - so you can interact with it (streaming, human-in-the-loop, memory, studio)
in the same way you would any LangGraph agent.

## Creating a custom deep agent

There are several parameters you can pass to `create_deep_agent` to create your own custom deep agent.

### `tools` (Required)

The first argument to `create_deep_agent` is `tools`.
This should be a list of functions or LangChain `@tool` objects.
The agent (and any subagents) will have access to these tools.

### `instructions` (Required)

The second argument to `create_deep_agent` is `instructions`.
This will serve as part of the prompt of the deep agent.
Note that our deep agent middleware appends further instructions to the deep agent regarding to-do list, filesystem, and subagent usage, so this is not the *entire* prompt the agent will see.

### `subagents` (Optional)

A keyword-only argument to `create_deep_agent` is `subagents`.
This can be used to specify any custom subagents this deep agent will have access to.
You can read more about why you would want to use subagents [here](#sub-agents)

`subagents` should be a list of dictionaries, where each dictionary follow this schema:

```python
class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    model: NotRequired[Union[LanguageModelLike, dict[str, Any]]]
    middleware: NotRequired[list[AgentMiddleware]]

class CustomSubAgent(TypedDict):
    name: str
    description: str
    graph: Runnable
```

**SubAgent fields:**
- **name**: This is the name of the subagent, and how the main agent will call the subagent
- **description**: This is the description of the subagent that is shown to the main agent
- **prompt**: This is the prompt used for the subagent
- **tools**: This is the list of tools that the subagent has access to. By default will have access to all tools passed in, as well as all built-in tools.
- **model**: Optional model instance OR dictionary for per-subagent model configuration (inherits the main model when omitted).
- **middleware** Additional middleware to attach to the subagent. See [here](https://docs.langchain.com/oss/python/langchain/middleware) for an introduction into middleware and how it works with create_agent.

**CustomSubAgent fields:**
- **name**: This is the name of the subagent, and how the main agent will call the subagent
- **description**: This is the description of the subagent that is shown to the main agent  
- **graph**: A pre-built LangGraph graph/agent that will be used as the subagent

#### Using SubAgent

```python
research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "prompt": sub_research_prompt,
    "tools": [internet_search]
}
subagents = [research_subagent]
agent = create_deep_agent(
    tools,
    prompt,
    subagents=subagents
)
```

#### Using CustomSubAgent

For more complex use cases, you can provide your own pre-built LangGraph graph as a subagent:

```python
from langchain.agents import create_agent

# Create a custom agent graph
custom_graph = create_agent(
    model=your_model,
    tools=specialized_tools,
    prompt="You are a specialized agent for data analysis..."
)

# Use it as a custom subagent
custom_subagent = {
    "name": "data-analyzer",
    "description": "Specialized agent for complex data analysis tasks",
    "graph": custom_graph
}

subagents = [custom_subagent]
agent = create_deep_agent(
    tools,
    prompt,
    subagents=subagents
)
```

### `model` (Optional)

By default, `deepagents` uses `"claude-sonnet-4-20250514"`. You can customize this by passing any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

#### Example: Using a Custom Model

Here's how to use a custom model (like OpenAI's `gpt-oss` model via Ollama):

(Requires `pip install langchain` and then `pip install langchain-ollama` for Ollama models)

```python
from deepagents import create_deep_agent

# ... existing agent definitions ...

model = init_chat_model(
    model="ollama:gpt-oss:20b",  
)
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
    ...
)
```

#### Example: Per-subagent model override (optional)

Use a fast, deterministic model for a critique sub-agent, while keeping a different default model for the main agent and others:

```python
from deepagents import create_deep_agent

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Critique the final report",
    "prompt": "You are a tough editor.",
    "model_settings": {
        "model": "anthropic:claude-3-5-haiku-20241022",
        "temperature": 0,
        "max_tokens": 8192
    }
}

agent = create_deep_agent(
    tools=[internet_search],
    instructions="You are an expert researcher...",
    model="claude-sonnet-4-20250514",  # default for main agent and other sub-agents
    subagents=[critique_sub_agent],
)
```


### `middleware` (Optional)
Both the main agent and sub-agents can take additional custom AgentMiddleware. Middleware is the best supported approach for extending the state_schema, adding additional tools, and adding pre / post model hooks. See this [doc](https://docs.langchain.com/oss/python/langchain/middleware) to learn more about Middleware and how you can use it!

### `tool_configs` (Optional)
Tool configs are used to specify how to handle Human In The Loop interactions on certain tools that require additional human oversight. 

These tool_configs are passed to our prebuilt [HITL middleware](https://docs.langchain.com/oss/python/langchain/middleware#human-in-the-loop) so that the agent pauses execution and waits for feedback from the user before executing configured tools.

## API Reference

This section provides a comprehensive reference for all public functions, classes, and structures in the `deepagents` library.

### Core Functions

#### `create_deep_agent()`

Creates a synchronous deep agent with built-in planning, file system, and subagent capabilities.

```python
def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]] = [],
    instructions: str = "",
    middleware: Optional[list[AgentMiddleware]] = None,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: Optional[list[SubAgent | CustomSubAgent]] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    tool_configs: Optional[dict[str, bool | ToolConfig]] = None,
) -> CompiledGraph
```

**Parameters:**
- `tools`: List of functions, LangChain tools, or tool dictionaries that the agent can use
- `instructions`: Custom instructions added to the system prompt to steer agent behavior
- `middleware`: Additional custom middleware to extend agent capabilities
- `model`: LangChain model instance or model identifier (defaults to `claude-sonnet-4-20250514`)
- `subagents`: List of SubAgent or CustomSubAgent dictionaries for specialized tasks
- `context_schema`: Custom state schema extending DeepAgentState
- `checkpointer`: LangGraph checkpointer for persisting state between runs
- `tool_configs`: Dictionary mapping tool names to HumanInTheLoopConfig for approval workflows

**Returns:**
- A LangGraph `CompiledGraph` that can be invoked, streamed, or used with LangGraph Studio

**Example:**
```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    tools=[my_tool_1, my_tool_2],
    instructions="You are a helpful research assistant.",
    model="claude-sonnet-4-20250514",
)

result = agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
```

#### `async_create_deep_agent()`

Creates an asynchronous deep agent. Identical to `create_deep_agent()` but optimized for async tools and operations.

```python
def async_create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]] = [],
    instructions: str = "",
    middleware: Optional[list[AgentMiddleware]] = None,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: Optional[list[SubAgent | CustomSubAgent]] = None,
    context_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    tool_configs: Optional[dict[str, bool | ToolConfig]] = None,
) -> CompiledGraph
```

**Use this when:**
- Working with async tools (e.g., MCP tools)
- Need to await agent invocations
- Building async applications

**Example:**
```python
from deepagents import async_create_deep_agent

agent = async_create_deep_agent(
    tools=[async_tool_1, async_tool_2],
    instructions="You are a helpful assistant.",
)

result = await agent.ainvoke({"messages": [{"role": "user", "content": "Hello!"}]})
```

### Middleware Classes

Middleware extends agent capabilities by adding tools, modifying prompts, and managing state.

#### `PlanningMiddleware`

Adds todo list planning capabilities to the agent.

- **Tool added:** `write_todos` - Creates and manages task lists
- **State added:** `todos` field containing list of Todo items
- **Prompt modification:** Adds instructions for when and how to use planning

```python
from deepagents import PlanningMiddleware

# Already included by default in create_deep_agent()
# Can be used standalone with create_agent():
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[PlanningMiddleware()],
)
```

#### `FilesystemMiddleware`

Adds virtual file system capabilities to the agent.

- **Tools added:** `ls`, `read_file`, `write_file`, `edit_file`
- **State added:** `files` field containing dictionary of filename to content
- **Prompt modification:** Adds instructions for file system tool usage

```python
from deepagents import FilesystemMiddleware

# Already included by default in create_deep_agent()
# Can be used standalone:
agent = create_agent(
    model=model,
    tools=tools,
    middleware=[FilesystemMiddleware()],
)
```

#### `SubAgentMiddleware`

Adds the ability to spawn ephemeral subagents for isolated tasks.

- **Tool added:** `task` - Launches subagents with isolated context
- **Built-in subagent:** `general-purpose` agent with same tools as main agent
- **Prompt modification:** Adds instructions for when and how to use subagents

```python
from deepagents import SubAgentMiddleware

# Customize subagent behavior:
middleware = SubAgentMiddleware(
    default_subagent_tools=[tool1, tool2],
    subagents=[
        {
            "name": "researcher",
            "description": "Expert at research tasks",
            "prompt": "You are a research expert...",
        }
    ],
    model=model,
    is_async=False,
)
```

### State Schema

#### `DeepAgentState`

The default state schema for deep agents, combining planning and file system state.

```python
from typing import NotRequired
from langchain.agents.middleware import AgentState

class DeepAgentState(AgentState):
    todos: NotRequired[list[Todo]]  # Task list for planning
    files: Annotated[NotRequired[dict[str, str]], file_reducer]  # Virtual filesystem
```

**Fields:**
- `messages`: List of messages in the conversation (inherited from AgentState)
- `todos`: Optional list of Todo items for task tracking
- `files`: Optional dictionary mapping file paths to file contents

#### `Todo`

Structure for individual todo items in the planning system.

```python
class Todo(TypedDict):
    content: str  # Description of the task
    status: Literal["pending", "in_progress", "completed"]  # Current status
```

### Type Definitions

#### `SubAgent`

Type definition for creating custom subagents with specific prompts and tools.

```python
class SubAgent(TypedDict):
    name: str  # Identifier for the subagent
    description: str  # Description shown to main agent for deciding when to use
    prompt: str  # System prompt for the subagent
    tools: NotRequired[list[BaseTool]]  # Tools available to subagent (defaults to all)
    model: NotRequired[Union[LanguageModelLike, dict[str, Any]]]  # Optional custom model
    middleware: NotRequired[list[AgentMiddleware]]  # Additional middleware
```

#### `CustomSubAgent`

Type definition for using pre-built LangGraph graphs as subagents.

```python
class CustomSubAgent(TypedDict):
    name: str  # Identifier for the subagent
    description: str  # Description shown to main agent
    graph: Runnable  # Pre-built LangGraph graph or agent
```

### Built-in Tools

Deep agents automatically include these tools:

#### Planning Tool

- **`write_todos(todos: list[Todo])`**: Creates or updates the task list. Used for complex multi-step objectives.

#### File System Tools

- **`ls()`**: Lists all files in the virtual filesystem
- **`read_file(file_path: str, offset: int = 0, limit: int = 2000)`**: Reads file contents with optional line offset/limit
- **`write_file(file_path: str, content: str)`**: Creates or overwrites a file
- **`edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False)`**: Performs exact string replacements in files

#### Subagent Tool

- **`task(description: str, subagent_type: str)`**: Launches an ephemeral subagent to handle isolated tasks. Available subagent types include `general-purpose` and any custom subagents defined.

## Deep Agent Details

The below components are built into `deepagents` and helps make it work for deep tasks off-the-shelf.

### System Prompt

`deepagents` comes with a [built-in system prompt](src/deepagents/prompts/). This is relatively detailed prompt that is heavily based on and inspired by [attempts](https://github.com/kn1026/cc/blob/main/claudecode.md) to [replicate](https://github.com/asgeirtj/system_prompts_leaks/blob/main/Anthropic/claude-code.md)
Claude Code's system prompt. It was made more general purpose than Claude Code's system prompt.
This contains detailed instructions for how to use the built-in planning tool, file system tools, and sub agents.
Note that part of this system prompt [can be customized](#instructions-required)

Without this default system prompt - the agent would not be nearly as successful at going as it is.
The importance of prompting for creating a "deep" agent cannot be understated.

### Planning Tool

`deepagents` comes with a built-in planning tool. This planning tool is very simple and is based on ClaudeCode's TodoWrite tool.
This tool doesn't actually do anything - it is just a way for the agent to come up with a plan, and then have that in the context to help keep it on track.

### File System Tools

`deepagents` comes with four built-in file system tools: `ls`, `edit_file`, `read_file`, `write_file`.
These do not actually use a file system - rather, they mock out a file system using LangGraph's State object.
This means you can easily run many of these agents on the same machine without worrying that they will edit the same underlying files.

Right now the "file system" will only be one level deep (no sub directories).

These files can be passed in (and also retrieved) by using the `files` key in the LangGraph State object.

```python
agent = create_deep_agent(...)

result = agent.invoke({
    "messages": ...,
    # Pass in files to the agent using this key
    # "files": {"foo.txt": "foo", ...}
})

# Access any files afterwards like this
result["files"]
```

### Sub Agents

`deepagents` comes with the built-in ability to call sub agents (based on Claude Code).
It has access to a `general-purpose` subagent at all times - this is a subagent with the same instructions as the main agent and all the tools that is has access to.
You can also specify [custom sub agents](#subagents-optional) with their own instructions and tools.

Sub agents are useful for ["context quarantine"](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html#context-quarantine) (to help not pollute the overall context of the main agent)
as well as custom instructions.

### Built In Tools

By default, deep agents come with five built-in tools:

- `write_todos`: Tool for writing todos
- `write_file`: Tool for writing to a file in the virtual filesystem
- `read_file`: Tool for reading from a file in the virtual filesystem
- `ls`: Tool for listing files in the virtual filesystem
- `edit_file`: Tool for editing a file in the virtual filesystem

If you want to omit some deepagents functionality, use specific middleware components directly!

### Human-in-the-Loop

`deepagents` supports human-in-the-loop approval for tool execution. You can configure specific tools to require human approval before execution using the `tool_configs` parameter, which maps tool names to a `HumanInTheLoopConfig`.

`HumanInTheLoopConfig` is how you specify what type of human in the loop patterns are supported. 
It is a dictionary with four specific keys:

- `allow_accept`: Whether the human can approve the current action without changes
- `allow_respond`: Whether the human can reject the current action with feedback
- `allow_edit`: Whether the human can approve the current action with edited content

Instead of specifying a `HumanInTheLoopConfig` for a tool, you can also just set `True`. This will set `allow_ignore`, `allow_respond`, `allow_edit`, and `allow_accept` to be `True`.

In order to use human in the loop, you need to have a checkpointer attached.
Note: if you are using LangGraph Platform, this is automatically attached.

Example usage:

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

# Create agent with specific tools requiring human approval
agent = create_deep_agent(
    tools=[your_tools],
    instructions="Your instructions here",
    tool_configs={
        # Fine-grained control: specify which approval actions are allowed
        "tool_1": {
            "allow_respond": True,  # Allow rejecting with feedback
            "allow_edit": True,     # Allow modifying tool arguments
            "allow_accept": True,   # Allow approving as-is
        },
        # Shortcut: True enables all approval options
        # Equivalent to the dictionary above
        "tool_2": True,
    }
)

# Attach checkpointer to enable interrupts
# Required for human-in-the-loop functionality
checkpointer = InMemorySaver()
agent.checkpointer = checkpointer
```

#### Approve

To "approve" a tool call means the agent will execute the tool call as-is without modifications.

This flow shows how to approve a tool call (assuming the tool requiring approval is called):

```python
from langgraph.types import Command

# Configuration with thread_id for state persistence
config = {"configurable": {"thread_id": "1"}}

# Initial agent invocation - will pause at tool requiring approval
for s in agent.stream({"messages": [{"role": "user", "content": message}]}, config=config):
    print(s)

# Resume with approval - execute the tool call as-is
for s in agent.stream(Command(resume=[{"type": "accept"}]), config=config):
    print(s)
```

#### Edit

To "edit" a tool call means the agent will execute a modified version of the tool call. You can change both the tool to call and the arguments to pass to that tool.

The `args` parameter you pass back should be a dictionary with two keys:

- `action`: String name of the tool to call
- `args`: Dictionary of arguments to pass to the tool

This flow shows how to edit a tool call (assuming the tool requiring approval is called):

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "1"}}

# Initial agent invocation - will pause at tool requiring approval
for s in agent.stream({"messages": [{"role": "user", "content": message}]}, config=config):
    print(s)

# Resume with edited tool call
# Modify the action and args to match your desired tool call
for s in agent.stream(
    Command(resume=[{
        "type": "edit",
        "args": {
            "action": "write_file",  # Tool to call
            "args": {                # Arguments for the tool
                "file_path": "output.txt",
                "content": "Modified content"
            }
        }
    }]),
    config=config
):
    print(s)
```

#### Respond

To "respond" to a tool call means that tool is NOT executed. Instead, a tool message with your response is appended to the conversation, and control returns to the model with your feedback.

The `args` parameter you pass back should be a string with your response message.

This flow shows how to respond to a tool call (assuming the tool requiring approval is called):

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "1"}}

# Initial agent invocation - will pause at tool requiring approval
for s in agent.stream({"messages": [{"role": "user", "content": message}]}, config=config):
    print(s)

# Resume with rejection and feedback
# The tool won't execute; instead, the agent receives your feedback
for s in agent.stream(
    Command(resume=[{
        "type": "response",
        "args": "I don't approve this action. Please try a different approach."
    }]),
    config=config
):
    print(s)
```
## Async

If you are passing async tools to your agent, you will want to use `from deepagents import async_create_deep_agent`
## MCP

The `deepagents` library can be ran with MCP tools. This can be achieved by using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

**NOTE:** You will want to use `from deepagents import async_create_deep_agent` to use the async version of `deepagents`, since MCP tools are async

(To run the example below, will need to `pip install langchain-mcp-adapters`)

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import async_create_deep_agent

async def main():
    # Initialize MCP client and collect MCP tools
    # MCP (Model Context Protocol) provides standardized tool integrations
    mcp_client = MultiServerMCPClient(...)
    mcp_tools = await mcp_client.get_tools()

    # Create async agent with MCP tools
    # Must use async_create_deep_agent for async tools
    agent = async_create_deep_agent(
        tools=mcp_tools,
        instructions="You are a helpful assistant with access to MCP tools.",
    )

    # Stream agent responses asynchronously
    # The agent will use MCP tools as needed to answer questions
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is langgraph?"}]},
        stream_mode="values"
    ):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()

asyncio.run(main())
```

## Troubleshooting

### Common Issues and Solutions

#### Model Configuration Issues

**Problem:** Error when initializing custom models

```
ValueError: Could not identify model provider
```

**Solution:** Ensure you're using `langchain.chat_models.init_chat_model()` or passing a valid LangChain model instance:

```python
from langchain.chat_models import init_chat_model

# Correct: Use init_chat_model for string identifiers
model = init_chat_model("anthropic:claude-sonnet-4-20250514")

# Or pass a model instance directly
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-sonnet-4-20250514")

agent = create_deep_agent(tools=tools, instructions=instructions, model=model)
```

**Problem:** Model not supporting tool calling

```
NotImplementedError: Model does not support tool calling
```

**Solution:** Ensure you're using a model that supports function calling. Most modern LLMs (GPT-4, Claude 3+, Gemini) support this. If using a local or custom model, verify it has tool calling capabilities.

#### Memory and Checkpoint Issues

**Problem:** Agent state not persisting between runs

**Solution:** Attach a checkpointer to enable persistence:

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_deep_agent(tools=tools, instructions=instructions)
agent.checkpointer = InMemorySaver()

# Use thread_id in config to maintain conversation history
config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke({"messages": [...]}, config=config)
```

**Problem:** Memory usage growing too large

**Solution:** The built-in `SummarizationMiddleware` automatically manages context by summarizing old messages. You can adjust the settings:

```python
from langchain.agents.middleware import SummarizationMiddleware

# Create custom summarization middleware with different thresholds
custom_summarization = SummarizationMiddleware(
    model=model,
    max_tokens_before_summary=80000,  # Lower threshold
    messages_to_keep=10,  # Keep fewer recent messages
)

# Replace default middleware
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    middleware=[custom_summarization],
)
```

#### File System Limitations

**Problem:** Virtual file system not persisting between runs

**Solution:** The virtual file system is stored in the agent state. To persist files, you need to:

1. Use a checkpointer
2. Pass files between invocations using the state

```python
# First run - files are created
result1 = agent.invoke({"messages": [{"role": "user", "content": "Create a file"}]})

# Access files from state
created_files = result1["files"]

# Pass files to next run
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "Read the file"}],
    "files": created_files  # Pass files forward
})
```

**Problem:** File paths with subdirectories not working

**Solution:** The current virtual file system only supports flat structure (no subdirectories). Use flat file naming:

```python
# Instead of: data/reports/analysis.txt
# Use: data_reports_analysis.txt
```

#### Human-in-the-Loop Issues

**Problem:** Interrupts not working

**Solution:** Ensure you have a checkpointer attached and are using the correct config:

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    tool_configs={"write_file": True},
)
agent.checkpointer = InMemorySaver()

# Must use thread_id for interrupts to work
config = {"configurable": {"thread_id": "1"}}
```

**Problem:** Want to disable interrupts temporarily

**Solution:** Use a different config or omit tool_configs when creating the agent:

```python
# Create agent without tool_configs for no interrupts
agent = create_deep_agent(tools=tools, instructions=instructions)
```

#### Async Issues

**Problem:** Async tools not working with `create_deep_agent()`

**Solution:** Use `async_create_deep_agent()` for async tools:

```python
from deepagents import async_create_deep_agent

agent = async_create_deep_agent(
    tools=[async_tool],  # Async tools
    instructions=instructions,
)

# Use async methods
result = await agent.ainvoke({"messages": [...]})
```

## Contributing

We welcome contributions to `deepagents`! Whether you're fixing bugs, improving documentation, or adding new features, your help is appreciated.

### Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes and add tests
5. Run tests and ensure they pass
6. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Format code
ruff format .
```

### Contribution Guidelines

- Write clear, descriptive commit messages
- Add tests for new features
- Update documentation as needed
- Follow existing code style and conventions
- Ensure all tests pass before submitting PR

For more detailed information, please see [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon).

### Reporting Issues

If you encounter a bug or have a feature request:

1. Check existing issues to avoid duplicates
2. Create a new issue with a clear title and description
3. Include code examples and error messages when applicable
4. Specify your environment (Python version, OS, package versions)

## Roadmap

- [ ] Allow users to customize full system prompt
- [x] Code cleanliness (type hinting, docstrings, formatting)
- [ ] Allow for more of a robust virtual filesystem
- [ ] Create an example of a deep coding agent built on top of this
- [ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)
