# LangGraph-Specific Testing Guide for DeepAgents

## Executive Summary

This guide provides LangGraph-specific testing patterns for the deepagents project. DeepAgents is built on LangGraph's agent framework and requires specialized testing approaches beyond standard Python unit tests.

## Table of Contents

1. [LangGraph Architecture Understanding](#langgraph-architecture-understanding)
2. [Critical LangGraph Components to Test](#critical-langgraph-components-to-test)
3. [Testing Patterns by Component](#testing-patterns-by-component)
4. [Common LangGraph Testing Pitfalls](#common-langgraph-testing-pitfalls)
5. [Test Helpers and Utilities](#test-helpers-and-utilities)
6. [Coverage Gaps and Recommendations](#coverage-gaps-and-recommendations)

---

## LangGraph Architecture Understanding

### What DeepAgents Uses from LangGraph

DeepAgents leverages:

1. **CompiledGraph** - Created via `langchain.agents.create_agent()`
2. **State Management** - TypedDict-based state schemas with reducers
3. **Checkpointing** - State persistence between runs
4. **AgentMiddleware** - Graph modification and state extension
5. **Command Objects** - State updates from tools
6. **Subgraphs** - Isolated agent instances for delegation
7. **Stream Channels** - State field accessibility
8. **InjectedState** - Tool parameter injection from graph state

### Graph Structure

```
create_agent() returns a CompiledGraph with:
├── nodes
│   ├── agent (model calling node)
│   └── tools (tool execution node)
├── stream_channels (accessible state fields)
├── checkpointer (optional state persistence)
└── compiled graph structure
```

---

## Critical LangGraph Components to Test

### 1. Graph Construction (/home/user/deepagents/src/deepagents/graph.py)

**What to Test:**
- CompiledGraph creation succeeds
- Graph has correct nodes (agent, tools)
- Stream channels match middleware state schemas
- Tools are bound correctly
- Middleware stack is applied in order

**Why It Matters:**
- Graph compilation can fail silently
- Tool binding issues prevent runtime execution
- State schema mismatches cause runtime errors

**Current Coverage:** ✅ Partial (structure only)
**Missing:** Compilation validation, node configuration

### 2. State Management (/home/user/deepagents/src/deepagents/state.py)

**What to Test:**
- State schema compatibility (TypedDict fields)
- Annotated fields with reducers (`file_reducer`)
- State transitions through graph execution
- State persistence with checkpointers
- NotRequired field handling

**Why It Matters:**
- Reducer bugs cause state corruption
- Schema incompatibility breaks middleware stacking
- State persistence failures lose data

**Current Coverage:** ❌ None
**Missing:** All state transition testing

### 3. Middleware (/home/user/deepagents/src/deepagents/middleware.py)

**What to Test:**
- `modify_model_request()` execution
- Middleware execution order
- State schema merging across middleware
- Tool injection from middleware
- Prompt modification chain

**Why It Matters:**
- Middleware order affects behavior
- Schema conflicts cause runtime errors
- Prompt modifications may not apply

**Current Coverage:** ✅ Partial (tools/state presence)
**Missing:** Execution flow, state modifications

### 4. Tools with InjectedState (/home/user/deepagents/src/deepagents/tools.py)

**What to Test:**
- InjectedState parameter injection
- InjectedToolCallId injection
- Command-based state updates
- Tool execution in graph context
- State access from tools

**Why It Matters:**
- Injection failures cause tool errors
- Command updates may not merge correctly
- State access requires active graph execution

**Current Coverage:** ❌ None
**Missing:** All injection and state update testing

### 5. Subagent/Subgraph System

**What to Test:**
- Subagent spawning and isolation
- State passing to subgraphs
- Subgraph result integration
- Error propagation from subagents
- Custom vs default subagent behavior

**Why It Matters:**
- State leakage between subagents
- Failed result integration corrupts state
- Errors may not propagate correctly

**Current Coverage:** ✅ Partial (spawning only)
**Missing:** Isolation, state flow, error handling

### 6. Checkpointing Integration

**What to Test:**
- State persistence across invocations
- Thread ID configuration
- Checkpoint loading and saving
- State reconstruction from checkpoints
- Interrupt/resume with checkpointers

**Why It Matters:**
- Persistence failures lose conversation state
- Incorrect thread handling causes state mixing
- HITL requires working checkpoints

**Current Coverage:** ✅ Minimal (HITL test only)
**Missing:** Direct checkpoint testing

---

## Testing Patterns by Component

### Pattern 1: Graph Construction Testing

```python
# Test file: tests/test_graph_construction.py

import pytest
from deepagents.graph import create_deep_agent, agent_builder
from langchain_core.tools import tool

def test_graph_compilation():
    """Test that graph compiles successfully"""
    agent = create_deep_agent(tools=[], instructions="test")

    # Verify it's a compiled graph
    assert hasattr(agent, 'nodes')
    assert hasattr(agent, 'stream_channels')
    assert hasattr(agent, 'invoke')
    assert hasattr(agent, 'stream')

def test_graph_nodes_present():
    """Test that graph has required nodes"""
    agent = create_deep_agent(tools=[], instructions="test")

    # LangGraph agents have 'agent' and 'tools' nodes
    assert 'agent' in agent.nodes
    assert 'tools' in agent.nodes

def test_stream_channels_from_middleware():
    """Test that middleware state schemas create stream channels"""
    agent = create_deep_agent(tools=[], instructions="test")

    # PlanningMiddleware adds 'todos'
    assert 'todos' in agent.stream_channels
    # FilesystemMiddleware adds 'files'
    assert 'files' in agent.stream_channels
    # AgentState always has 'messages'
    assert 'messages' in agent.stream_channels

def test_tool_binding_from_middleware():
    """Test that middleware tools are bound to graph"""
    agent = create_deep_agent(tools=[], instructions="test")

    # Access bound tools
    bound_tools = agent.nodes["tools"].bound._tools_by_name

    # From PlanningMiddleware
    assert "write_todos" in bound_tools
    # From FilesystemMiddleware
    assert "ls" in bound_tools
    assert "read_file" in bound_tools
    assert "write_file" in bound_tools
    assert "edit_file" in bound_tools
    # From SubAgentMiddleware
    assert "task" in bound_tools

def test_custom_tools_bound():
    """Test that custom tools are bound correctly"""
    @tool
    def custom_tool(x: str) -> str:
        """Custom tool"""
        return x

    agent = create_deep_agent(tools=[custom_tool], instructions="test")
    bound_tools = agent.nodes["tools"].bound._tools_by_name

    assert "custom_tool" in bound_tools

def test_middleware_order_affects_state():
    """Test that middleware is applied in order"""
    from deepagents.middleware import PlanningMiddleware, FilesystemMiddleware
    from langchain.agents import create_agent

    # Middleware should stack their state schemas
    agent = create_agent(
        model="claude-3-5-sonnet-20240620",
        middleware=[PlanningMiddleware(), FilesystemMiddleware()],
        tools=[]
    )

    # Both middleware state fields should be present
    assert 'todos' in agent.stream_channels
    assert 'files' in agent.stream_channels
```

### Pattern 2: State Reducer Testing

```python
# Test file: tests/test_state_reducers.py

import pytest
from deepagents.state import file_reducer, DeepAgentState
from deepagents.graph import create_deep_agent
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain.agents.tool_node import InjectedState
from typing import Annotated

def test_file_reducer_left_none():
    """Test file_reducer when left is None"""
    result = file_reducer(None, {"file1.txt": "content1"})
    assert result == {"file1.txt": "content1"}

def test_file_reducer_right_none():
    """Test file_reducer when right is None"""
    result = file_reducer({"file1.txt": "content1"}, None)
    assert result == {"file1.txt": "content1"}

def test_file_reducer_both_none():
    """Test file_reducer when both are None"""
    result = file_reducer(None, None)
    assert result is None

def test_file_reducer_merge():
    """Test file_reducer merges dictionaries"""
    left = {"file1.txt": "content1", "file2.txt": "old"}
    right = {"file2.txt": "new", "file3.txt": "content3"}
    result = file_reducer(left, right)

    # Right should override left for file2.txt
    assert result == {
        "file1.txt": "content1",
        "file2.txt": "new",
        "file3.txt": "content3"
    }

def test_file_reducer_in_graph_execution():
    """Test that file_reducer works in actual graph execution"""

    # Create a tool that updates files twice
    @tool
    def update_files(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Update files in state"""
        current_files = state.get("files", {})
        new_files = {**current_files, "new.txt": "new content"}
        return Command(update={"files": new_files})

    agent = create_deep_agent(tools=[update_files], instructions="test")

    # Initial invoke with existing files
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Update files"}],
        "files": {"old.txt": "old content"}
    })

    # The reducer should have merged the files
    assert "old.txt" in result.get("files", {})
    # New file should be added (if the tool was called)
    # Note: This depends on model behavior, may need mocking

def test_state_schema_compatibility():
    """Test that DeepAgentState is compatible with middleware states"""
    from deepagents.state import DeepAgentState, PlanningState, FilesystemState
    from typing import get_type_hints

    # Get fields from each state
    deep_hints = get_type_hints(DeepAgentState)
    planning_hints = get_type_hints(PlanningState)
    filesystem_hints = get_type_hints(FilesystemState)

    # DeepAgentState should include all fields
    assert 'todos' in deep_hints or 'todos' in DeepAgentState.__annotations__
    assert 'files' in deep_hints or 'files' in DeepAgentState.__annotations__
```

### Pattern 3: Middleware Testing

```python
# Test file: tests/test_middleware_execution.py

import pytest
from deepagents.middleware import PlanningMiddleware, FilesystemMiddleware, SubAgentMiddleware
from langchain.agents.middleware import ModelRequest, AgentState
from deepagents.prompts import WRITE_TODOS_SYSTEM_PROMPT, FILESYSTEM_SYSTEM_PROMPT

def test_planning_middleware_modify_request():
    """Test PlanningMiddleware modifies model request"""
    middleware = PlanningMiddleware()

    request = ModelRequest(
        messages=[],
        system_prompt="Original prompt"
    )
    state = {"messages": []}

    modified = middleware.modify_model_request(request, state)

    # Should append planning instructions
    assert WRITE_TODOS_SYSTEM_PROMPT in modified.system_prompt
    assert "Original prompt" in modified.system_prompt

def test_filesystem_middleware_modify_request():
    """Test FilesystemMiddleware modifies model request"""
    middleware = FilesystemMiddleware()

    request = ModelRequest(
        messages=[],
        system_prompt="Original prompt"
    )
    state = {"messages": [], "files": {}}

    modified = middleware.modify_model_request(request, state)

    # Should append filesystem instructions
    assert FILESYSTEM_SYSTEM_PROMPT in modified.system_prompt
    assert "Original prompt" in modified.system_prompt

def test_middleware_stacking_order():
    """Test that multiple middleware modify request in order"""
    from langchain.agents import create_agent

    # Create agent with stacked middleware
    agent = create_agent(
        model="claude-3-5-sonnet-20240620",
        middleware=[
            PlanningMiddleware(),
            FilesystemMiddleware(),
        ],
        prompt="Base prompt",
        tools=[]
    )

    # The system prompt should contain all middleware prompts
    # This requires invoking the graph to see the final prompt
    # For now, we verify the structure
    assert hasattr(agent, 'nodes')

def test_middleware_tools_injection():
    """Test that middleware tools are available in graph"""
    from langchain.agents import create_agent

    planning_mw = PlanningMiddleware()
    filesystem_mw = FilesystemMiddleware()

    agent = create_agent(
        model="claude-3-5-sonnet-20240620",
        middleware=[planning_mw, filesystem_mw],
        tools=[]
    )

    bound_tools = agent.nodes["tools"].bound._tools_by_name

    # Planning tools
    assert "write_todos" in bound_tools
    # Filesystem tools
    assert "ls" in bound_tools
    assert "read_file" in bound_tools
    assert "write_file" in bound_tools
    assert "edit_file" in bound_tools

def test_subagent_middleware_creates_task_tool():
    """Test SubAgentMiddleware creates task tool"""
    middleware = SubAgentMiddleware(
        default_subagent_tools=[],
        subagents=[],
        model="claude-3-5-sonnet-20240620"
    )

    # Should have created task tool
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "task"
```

### Pattern 4: InjectedState Testing

```python
# Test file: tests/test_injected_state.py

import pytest
from deepagents.tools import write_todos, read_file, write_file, edit_file, ls
from deepagents.graph import create_deep_agent
from deepagents.state import Todo
from langgraph.checkpoint.memory import MemorySaver
import uuid

def test_write_todos_injected_tool_call_id():
    """Test write_todos receives InjectedToolCallId"""
    agent = create_deep_agent(tools=[], instructions="test")

    # Invoke with request to write todos
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Create a todo list with one item: 'test task'"}]
    })

    # Check if todos were written (model-dependent, may need mocking)
    if "todos" in result:
        assert isinstance(result["todos"], list)

def test_read_file_injected_state():
    """Test read_file receives InjectedState"""
    agent = create_deep_agent(tools=[], instructions="test")

    # Invoke with existing files
    result = agent.invoke({
        "messages": [{"role": "user", "content": "List all files"}],
        "files": {"test.txt": "test content"}
    })

    # Model should be able to list files using ls tool
    # This is model-dependent, may need verification through messages

def test_write_file_command_update():
    """Test write_file returns Command with state update"""
    from langchain_core.tools import tool, InjectedToolCallId
    from langchain.agents.tool_node import InjectedState
    from typing import Annotated
    from langgraph.types import Command

    agent = create_deep_agent(tools=[], instructions="test")

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Write 'hello' to file.txt"}],
        "files": {}
    })

    # Check if file was written
    if "files" in result:
        # Model behavior dependent, verify structure
        assert isinstance(result["files"], dict)

def test_edit_file_with_state_access():
    """Test edit_file accesses state correctly"""
    agent = create_deep_agent(tools=[], instructions="test")

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Change 'old' to 'new' in test.txt"}],
        "files": {"test.txt": "old content"}
    })

    # Verify file was edited (model-dependent)
    if "files" in result and "test.txt" in result["files"]:
        # Check if edit occurred
        pass  # Model-dependent behavior

def test_injected_state_with_custom_tool():
    """Test custom tool with InjectedState"""
    from langchain_core.tools import tool, InjectedToolCallId
    from langchain.agents.tool_node import InjectedState
    from typing import Annotated
    from langgraph.types import Command

    @tool
    def custom_state_tool(
        value: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Custom tool that accesses state"""
        current_count = state.get("custom_count", 0)
        return Command(
            update={
                "custom_count": current_count + 1,
                "messages": [{"role": "tool", "content": f"Count: {current_count + 1}", "tool_call_id": tool_call_id}]
            }
        )

    # Need custom middleware to add state field
    from langchain.agents.middleware import AgentMiddleware, AgentState

    class CustomState(AgentState):
        custom_count: int

    class CustomMiddleware(AgentMiddleware):
        state_schema = CustomState
        tools = [custom_state_tool]

    from deepagents.graph import agent_builder

    agent = agent_builder(
        tools=[],
        instructions="test",
        middleware=[CustomMiddleware()],
        model="claude-3-5-sonnet-20240620"
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use custom state tool"}],
        "custom_count": 0
    })

    # Verify state was updated (model-dependent)
    assert "custom_count" in agent.stream_channels
```

### Pattern 5: Subagent/Subgraph Testing

```python
# Test file: tests/test_subagents.py

import pytest
from deepagents.graph import create_deep_agent
from langchain_core.tools import tool

@tool
def exclusive_tool(query: str) -> str:
    """Tool only for subagent"""
    return f"Exclusive: {query}"

def test_subagent_isolation():
    """Test that subagent has isolated context"""
    subagents = [
        {
            "name": "isolated_agent",
            "description": "Isolated subagent",
            "prompt": "You are isolated",
            "tools": [exclusive_tool],
            "model": "claude-3-5-sonnet-20240620"
        }
    ]

    agent = create_deep_agent(tools=[], subagents=subagents)

    # Invoke and check if subagent was created
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use the isolated agent to test"}]
    })

    # Verify agent exists (structure test)
    # The SubAgentMiddleware should have created it
    # Model-dependent behavior for actual invocation

def test_subagent_state_passing():
    """Test that state is passed to subagent"""
    subagents = [
        {
            "name": "state_agent",
            "description": "Agent that receives state",
            "prompt": "You receive state from parent",
            "model": "claude-3-5-sonnet-20240620"
        }
    ]

    agent = create_deep_agent(tools=[], subagents=subagents)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Pass files to state_agent"}],
        "files": {"parent.txt": "parent content"}
    })

    # Subagent should receive files in state
    # This is verified through the task tool implementation
    # which passes state to subagent

def test_subagent_result_integration():
    """Test that subagent results are integrated into parent state"""
    @tool
    def create_file_tool(filename: str) -> str:
        """Create a file in subagent"""
        return f"Created {filename}"

    subagents = [
        {
            "name": "file_creator",
            "description": "Creates files",
            "prompt": "You create files",
            "tools": [create_file_tool],
            "model": "claude-3-5-sonnet-20240620"
        }
    ]

    agent = create_deep_agent(tools=[], subagents=subagents)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use file_creator to make a file"}],
        "files": {}
    })

    # The task tool should integrate subagent's state updates
    # excluding 'todos' and 'messages' per the implementation
    # Files should be merged back

def test_subagent_error_propagation():
    """Test that subagent errors propagate correctly"""
    @tool
    def failing_tool() -> str:
        """Tool that fails"""
        raise ValueError("Tool failed")

    subagents = [
        {
            "name": "failing_agent",
            "description": "Agent with failing tool",
            "prompt": "You have a failing tool",
            "tools": [failing_tool],
            "model": "claude-3-5-sonnet-20240620"
        }
    ]

    agent = create_deep_agent(tools=[], subagents=subagents)

    # Invoke should handle error gracefully
    # LangGraph wraps tool errors in messages
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use failing_agent"}]
    })

    # Error should be in messages
    messages = result.get("messages", [])
    # Check for error message (model-dependent)

def test_custom_subagent_graph():
    """Test CustomSubAgent with pre-built graph"""
    from langchain.agents import create_agent

    custom_graph = create_agent(
        model="claude-3-5-sonnet-20240620",
        tools=[exclusive_tool],
        prompt="Custom graph agent"
    )

    subagents = [
        {
            "name": "custom_graph_agent",
            "description": "Pre-built graph",
            "graph": custom_graph
        }
    ]

    agent = create_deep_agent(tools=[], subagents=subagents)

    # Verify custom graph was integrated
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use custom_graph_agent"}]
    })

    # Model-dependent behavior

def test_general_purpose_subagent():
    """Test that general-purpose subagent exists by default"""
    agent = create_deep_agent(tools=[exclusive_tool], subagents=[])

    # The general-purpose subagent should have access to all tools
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use general-purpose subagent with exclusive_tool"}]
    })

    # Verify general-purpose can access parent tools
    # Model-dependent
```

### Pattern 6: Checkpointing Testing

```python
# Test file: tests/test_checkpointing.py

import pytest
from deepagents.graph import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
import uuid

def test_checkpoint_state_persistence():
    """Test that state persists across invocations"""
    checkpointer = MemorySaver()
    agent = create_deep_agent(tools=[], instructions="test", checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # First invocation
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "Write to file.txt"}],
        "files": {}
    }, config=config)

    # Second invocation - should have previous state
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "List files"}]
    }, config=config)

    # Both should share conversation history
    assert len(result2["messages"]) > len(result1["messages"])

def test_checkpoint_file_persistence():
    """Test that files persist with checkpointer"""
    checkpointer = MemorySaver()
    agent = create_deep_agent(tools=[], instructions="test", checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Create files in first invocation
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "Create file1.txt with content 'hello'"}],
        "files": {}
    }, config=config)

    # Second invocation should see files
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "List all files"}]
    }, config=config)

    # Files should persist (if created by model)
    if "files" in result1:
        assert "files" in result2

def test_checkpoint_thread_isolation():
    """Test that different threads are isolated"""
    checkpointer = MemorySaver()
    agent = create_deep_agent(tools=[], instructions="test", checkpointer=checkpointer)

    thread1 = str(uuid.uuid4())
    thread2 = str(uuid.uuid4())

    # Thread 1
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "Create thread1.txt"}],
        "files": {}
    }, config={"configurable": {"thread_id": thread1}})

    # Thread 2
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "List files"}],
        "files": {}
    }, config={"configurable": {"thread_id": thread2}})

    # Thread 2 should not see thread 1's files
    if "files" in result2:
        assert "thread1.txt" not in result2.get("files", {})

def test_checkpoint_interrupt_resume():
    """Test interrupt/resume with checkpointer (HITL)"""
    from langgraph.types import Command

    checkpointer = MemorySaver()
    tool_configs = {"write_file": True}

    agent = create_deep_agent(
        tools=[],
        instructions="test",
        checkpointer=checkpointer,
        tool_configs=tool_configs
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Invoke - should interrupt on write_file
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "Write to file.txt"}]
    }, config=config)

    # Should have interrupt
    if "__interrupt__" in result1:
        # Resume with approval
        result2 = agent.invoke(
            Command(resume=[{"type": "accept"}]),
            config=config
        )

        # Should complete
        assert "__interrupt__" not in result2

def test_checkpoint_state_reconstruction():
    """Test that state can be reconstructed from checkpoint"""
    checkpointer = MemorySaver()
    agent = create_deep_agent(tools=[], instructions="test", checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Create initial state
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "Create files"}],
        "files": {"initial.txt": "content"}
    }, config=config)

    # Get state from checkpoint
    state = agent.get_state(config)

    # State should contain messages and files
    assert "messages" in state.values
    if "files" in result1:
        assert "files" in state.values
```

### Pattern 7: Graph Streaming Testing

```python
# Test file: tests/test_streaming.py

import pytest
from deepagents.graph import create_deep_agent

def test_stream_mode_values():
    """Test streaming with mode='values'"""
    agent = create_deep_agent(tools=[], instructions="test")

    chunks = []
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "Hello"}]},
        stream_mode="values"
    ):
        chunks.append(chunk)

    # Should have multiple chunks
    assert len(chunks) > 0

    # Last chunk should have final state
    final = chunks[-1]
    assert "messages" in final

def test_stream_mode_updates():
    """Test streaming with mode='updates'"""
    agent = create_deep_agent(tools=[], instructions="test")

    updates = []
    for update in agent.stream(
        {"messages": [{"role": "user", "content": "Hello"}]},
        stream_mode="updates"
    ):
        updates.append(update)

    # Should have updates from nodes
    assert len(updates) > 0

def test_stream_channels_access():
    """Test accessing stream channels"""
    agent = create_deep_agent(tools=[], instructions="test")

    # Stream channels should match state schema
    assert "messages" in agent.stream_channels
    assert "todos" in agent.stream_channels
    assert "files" in agent.stream_channels

def test_streaming_with_custom_state():
    """Test streaming with custom state field"""
    from langchain.agents.middleware import AgentMiddleware, AgentState
    from deepagents.graph import agent_builder

    class CustomState(AgentState):
        custom_field: str

    class CustomMiddleware(AgentMiddleware):
        state_schema = CustomState

    agent = agent_builder(
        tools=[],
        instructions="test",
        middleware=[CustomMiddleware()],
        model="claude-3-5-sonnet-20240620"
    )

    # Custom field should be in stream channels
    assert "custom_field" in agent.stream_channels

    # Should be able to stream with custom field
    chunks = list(agent.stream(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "custom_field": "test"
        },
        stream_mode="values"
    ))

    assert len(chunks) > 0
```

---

## Common LangGraph Testing Pitfalls

### Pitfall 1: Testing Structure Instead of Behavior

**Problem:**
```python
# This only tests that tools exist, not that they work
def test_bad():
    agent = create_deep_agent()
    assert "write_todos" in agent.nodes["tools"].bound._tools_by_name
```

**Solution:**
```python
# Test actual execution
def test_good():
    agent = create_deep_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Create a todo"}]
    })
    # Verify behavior, not just structure
```

### Pitfall 2: Ignoring Model Non-Determinism

**Problem:**
```python
# This assumes model will call a specific tool
def test_bad():
    result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
    assert "specific_tool" in [msg.tool_calls[0]["name"] for msg in result["messages"]]
```

**Solution:**
```python
# Use mocking or focus on testing the framework, not model behavior
def test_good():
    # Mock the model to return specific tool calls
    # OR test that the tool CAN be called, not that it WILL be called
    agent = create_deep_agent(tools=[my_tool])
    assert "my_tool" in agent.nodes["tools"].bound._tools_by_name

    # For integration tests, use tool call verification
    result = agent.invoke(...)
    # If model called tool, verify the tool execution was correct
```

### Pitfall 3: Not Testing State Transitions

**Problem:**
```python
# Single invocation doesn't test state management
def test_bad():
    result = agent.invoke({"messages": [...]})
    assert "files" in result
```

**Solution:**
```python
# Test state across multiple steps
def test_good():
    agent = create_deep_agent(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "1"}}

    # Step 1
    r1 = agent.invoke({"messages": [...], "files": {}}, config=config)

    # Step 2 - should have state from step 1
    r2 = agent.invoke({"messages": [...]}, config=config)

    assert len(r2["messages"]) > len(r1["messages"])
```

### Pitfall 4: Not Testing Error Conditions

**Problem:**
```python
# Only testing happy path
def test_bad():
    agent = create_deep_agent()
    result = agent.invoke({"messages": [...]})
    assert result["messages"]
```

**Solution:**
```python
# Test error handling
def test_good():
    agent = create_deep_agent()

    # Test with invalid state
    result = agent.invoke({"messages": [], "files": "invalid"})
    # Should handle gracefully

    # Test with missing required fields
    result = agent.invoke({})  # No messages
    # Should handle gracefully
```

### Pitfall 5: Not Testing Middleware Interactions

**Problem:**
```python
# Testing middleware in isolation
def test_bad():
    middleware = PlanningMiddleware()
    assert middleware.tools
```

**Solution:**
```python
# Test middleware in graph context
def test_good():
    agent = create_agent(
        middleware=[PlanningMiddleware(), FilesystemMiddleware()],
        ...
    )

    # Verify both middleware are active
    assert "todos" in agent.stream_channels
    assert "files" in agent.stream_channels

    # Verify tools from both are available
    tools = agent.nodes["tools"].bound._tools_by_name
    assert "write_todos" in tools
    assert "write_file" in tools
```

### Pitfall 6: Not Testing Checkpointer Edge Cases

**Problem:**
```python
# Only testing basic persistence
def test_bad():
    agent = create_deep_agent(checkpointer=MemorySaver())
    result = agent.invoke({...}, config={"configurable": {"thread_id": "1"}})
    assert result
```

**Solution:**
```python
# Test edge cases
def test_good():
    checkpointer = MemorySaver()
    agent = create_deep_agent(checkpointer=checkpointer)

    # Test thread isolation
    r1 = agent.invoke({...}, config={"configurable": {"thread_id": "1"}})
    r2 = agent.invoke({...}, config={"configurable": {"thread_id": "2"}})
    # Verify r2 doesn't have r1's state

    # Test missing thread_id
    r3 = agent.invoke({...})  # No config
    # Should work (creates new thread)

    # Test state reconstruction
    state = agent.get_state(config={"configurable": {"thread_id": "1"}})
    assert state.values
```

### Pitfall 7: Not Testing Subgraph Isolation

**Problem:**
```python
# Assuming subagent has access to parent state
def test_bad():
    result = agent.invoke({"messages": [...], "files": {...}})
    # Expecting subagent to modify parent files directly
```

**Solution:**
```python
# Test isolation correctly
def test_good():
    # Subagent gets a copy of state
    # Changes are merged back through Command
    # Test that the merge happens correctly

    agent = create_deep_agent(subagents=[...])
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use subagent"}],
        "files": {"parent.txt": "content"}
    })

    # Subagent should receive files
    # Subagent changes should merge back
    # Test the task tool implementation
```

---

## Test Helpers and Utilities

### Helper 1: Mock Model for Deterministic Testing

```python
# File: tests/helpers/mock_model.py

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any

class MockToolCallingModel(BaseChatModel):
    """Mock model that returns predetermined tool calls"""

    def __init__(self, tool_calls: List[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.tool_calls = tool_calls or []
        self.call_count = 0

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1

        # Return tool calls on first call, finish on second
        if self.call_count == 1 and self.tool_calls:
            message = AIMessage(
                content="",
                tool_calls=self.tool_calls
            )
        else:
            message = AIMessage(content="Task completed")

        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "mock_tool_calling"

# Usage:
# mock_model = MockToolCallingModel(tool_calls=[
#     {"name": "write_file", "args": {"file_path": "test.txt", "content": "test"}, "id": "call_1"}
# ])
# agent = create_deep_agent(model=mock_model, ...)
```

### Helper 2: State Assertion Helpers

```python
# File: tests/helpers/assertions.py

def assert_state_schema_compatible(agent, expected_fields: List[str]):
    """Assert that agent has all expected state fields"""
    for field in expected_fields:
        assert field in agent.stream_channels, f"Missing state field: {field}"

def assert_tools_bound(agent, expected_tools: List[str]):
    """Assert that all expected tools are bound to agent"""
    bound_tools = agent.nodes["tools"].bound._tools_by_name
    for tool_name in expected_tools:
        assert tool_name in bound_tools, f"Tool not bound: {tool_name}"

def assert_state_transition(before_state: dict, after_state: dict, changed_fields: List[str]):
    """Assert that specific state fields changed"""
    for field in changed_fields:
        assert before_state.get(field) != after_state.get(field), \
            f"Expected {field} to change"

def assert_file_in_state(state: dict, filename: str, expected_content: str = None):
    """Assert that a file exists in state with optional content check"""
    assert "files" in state, "No files in state"
    assert filename in state["files"], f"File {filename} not in state"
    if expected_content:
        assert state["files"][filename] == expected_content, \
            f"File content mismatch for {filename}"
```

### Helper 3: Checkpoint Testing Helpers

```python
# File: tests/helpers/checkpoint.py

import uuid
from langgraph.checkpoint.memory import MemorySaver
from typing import Optional, Dict, Any

class CheckpointTestHelper:
    """Helper for testing checkpoint behavior"""

    def __init__(self, agent):
        self.agent = agent
        self.checkpointer = MemorySaver()
        self.agent.checkpointer = self.checkpointer
        self.thread_ids = {}

    def create_thread(self, name: str) -> str:
        """Create a named thread and return its ID"""
        thread_id = str(uuid.uuid4())
        self.thread_ids[name] = thread_id
        return thread_id

    def get_config(self, thread_name: str) -> Dict[str, Any]:
        """Get config for a named thread"""
        if thread_name not in self.thread_ids:
            self.create_thread(thread_name)
        return {"configurable": {"thread_id": self.thread_ids[thread_name]}}

    def invoke_in_thread(self, thread_name: str, input: dict) -> dict:
        """Invoke agent in a named thread"""
        config = self.get_config(thread_name)
        return self.agent.invoke(input, config=config)

    def get_thread_state(self, thread_name: str) -> dict:
        """Get current state of a named thread"""
        config = self.get_config(thread_name)
        state = self.agent.get_state(config)
        return state.values if state else {}

    def assert_threads_isolated(self, thread1: str, thread2: str):
        """Assert that two threads have isolated state"""
        state1 = self.get_thread_state(thread1)
        state2 = self.get_thread_state(thread2)

        # Messages should be different
        assert state1.get("messages") != state2.get("messages")

# Usage:
# helper = CheckpointTestHelper(agent)
# helper.invoke_in_thread("main", {"messages": [...]})
# helper.assert_threads_isolated("main", "secondary")
```

### Helper 4: Subagent Testing Helpers

```python
# File: tests/helpers/subagent.py

from typing import List, Dict, Any
from langchain_core.tools import tool

def create_test_subagent_config(
    name: str,
    description: str,
    tools: List = None,
    prompt: str = "Test subagent"
) -> Dict[str, Any]:
    """Create a test subagent configuration"""
    return {
        "name": name,
        "description": description,
        "prompt": prompt,
        "tools": tools or [],
        "model": "claude-3-5-sonnet-20240620"
    }

def create_marker_tool(marker: str):
    """Create a tool that marks when it's been called"""
    @tool
    def marker_tool(input: str) -> str:
        f"""Marker tool: {marker}"""
        return f"{marker}: {input}"

    marker_tool.name = f"{marker}_tool"
    return marker_tool

def assert_subagent_called(result: dict, subagent_name: str):
    """Assert that a specific subagent was called"""
    messages = result.get("messages", [])
    tool_messages = [msg for msg in messages if hasattr(msg, "tool_calls")]

    task_calls = []
    for msg in tool_messages:
        for tc in msg.tool_calls:
            if tc.get("name") == "task":
                task_calls.append(tc)

    assert any(
        tc.get("args", {}).get("subagent_type") == subagent_name
        for tc in task_calls
    ), f"Subagent {subagent_name} was not called"

# Usage:
# subagent = create_test_subagent_config("test_agent", "Test description")
# agent = create_deep_agent(subagents=[subagent])
# result = agent.invoke(...)
# assert_subagent_called(result, "test_agent")
```

---

## Coverage Gaps and Recommendations

### Current Coverage Analysis

Based on existing tests in `/home/user/deepagents/tests/`:

#### ✅ Currently Covered:
1. **Middleware structure** - Tools and state presence
2. **Basic agent creation** - Graph compilation
3. **Subagent configuration** - Creation and invocation
4. **HITL basics** - Interrupt and resume

#### ❌ NOT Covered (Critical Gaps):

1. **State Reducers**
   - No tests for `file_reducer` behavior
   - No tests for state merging across updates
   - No tests for Annotated field handling

2. **InjectedState Mechanics**
   - No tests verifying state injection works
   - No tests for `InjectedToolCallId` parameter
   - No tests for Command-based state updates

3. **Middleware Execution Flow**
   - No tests for `modify_model_request` execution
   - No tests for middleware ordering effects
   - No tests for prompt modification chain

4. **State Transitions**
   - No tests for state changes across graph steps
   - No tests for state persistence patterns
   - No tests for state validation

5. **Checkpointer Integration**
   - Only HITL test uses checkpointer
   - No tests for state reconstruction
   - No tests for thread isolation
   - No tests for checkpoint edge cases

6. **Subgraph Isolation**
   - No tests for state isolation between parent/subagent
   - No tests for state merging from subagents
   - No tests for subagent error propagation

7. **Graph Streaming**
   - No tests for streaming modes
   - No tests for stream channels
   - No tests for state updates during streaming

8. **Error Handling**
   - No tests for tool execution errors
   - No tests for state validation errors
   - No tests for graph execution failures

### Recommended Test Organization

```
tests/
├── unit/
│   ├── test_state_reducers.py          # NEW - State reducer logic
│   ├── test_middleware_classes.py      # NEW - Middleware methods
│   ├── test_tools_isolated.py          # NEW - Tool functions
│   └── test_state_schemas.py           # NEW - State schema validation
├── integration/
│   ├── test_graph_construction.py      # ENHANCED - Current test_middleware.py
│   ├── test_agent_execution.py         # ENHANCED - Current test_deepagents.py
│   ├── test_checkpointing.py           # NEW - Checkpoint behavior
│   ├── test_subagent_integration.py    # NEW - Subagent isolation/integration
│   ├── test_injected_state.py          # NEW - InjectedState in graph
│   └── test_streaming.py               # NEW - Graph streaming
├── e2e/
│   ├── test_deep_agent_workflows.py    # NEW - Complete workflows
│   ├── test_hitl_workflows.py          # ENHANCED - Current test_hitl.py
│   └── test_async_workflows.py         # NEW - Async agent tests
├── helpers/
│   ├── mock_model.py                   # NEW - Deterministic model
│   ├── assertions.py                   # NEW - Custom assertions
│   ├── checkpoint.py                   # NEW - Checkpoint helpers
│   └── subagent.py                     # NEW - Subagent helpers
└── utils.py                            # EXISTS - Keep current helpers
```

### Phase-Specific Additions

#### Phase 1: Foundation (Current + Critical Gaps)
- [x] Basic graph construction (EXISTS)
- [ ] State reducer unit tests (NEW)
- [ ] Middleware method tests (NEW)
- [ ] State schema validation (NEW)
- [ ] Basic checkpoint tests (NEW)

#### Phase 2: Integration
- [ ] InjectedState integration tests (NEW)
- [ ] Middleware execution flow (NEW)
- [ ] State transition tests (NEW)
- [ ] Subagent isolation tests (NEW)
- [ ] Enhanced HITL tests (ENHANCE)

#### Phase 3: Advanced
- [ ] Streaming tests (NEW)
- [ ] Error handling tests (NEW)
- [ ] Checkpoint edge cases (NEW)
- [ ] Async agent tests (NEW)
- [ ] Complex state scenarios (NEW)

#### Phase 4: E2E & Performance
- [ ] Complete workflow tests (NEW)
- [ ] Multi-agent scenarios (NEW)
- [ ] Performance benchmarks (NEW)
- [ ] Stress tests (NEW)

---

## Key Insights for LangGraph Testing

### 1. State is Central
- LangGraph's state management is the foundation
- Test reducers thoroughly - they control state merging
- Test Annotated fields separately from regular fields
- Test NotRequired fields with and without values

### 2. Middleware is Graph Modification
- Middleware doesn't just add features, it modifies the graph
- Test the graph AFTER middleware application
- Test middleware interaction, not just individual middleware
- Test `modify_model_request` is actually called

### 3. Tools Need Graph Context
- InjectedState only works in graph execution
- Tools with Command return values modify state
- Test tools in graph context, not in isolation
- Mock the model to ensure deterministic tool calls

### 4. Checkpointers Enable Statefulness
- Without checkpointer, no persistence
- Thread IDs are critical for isolation
- Test state reconstruction from checkpoints
- Test interrupt/resume patterns

### 5. Subgraphs are Isolated Execution
- Subagents get their own state copy
- State merges back through Command
- Test isolation and integration separately
- Errors may not propagate as expected

### 6. Streaming is Stateful
- Stream modes affect what you see
- Stream channels = accessible state fields
- Test different streaming modes
- Test state updates during streaming

---

## Conclusion

Testing LangGraph applications requires understanding the framework's state-centric, graph-based architecture. Key focus areas:

1. **State Management** - Reducers, schemas, transitions
2. **Middleware Integration** - Graph modification, stacking
3. **Tool Execution** - InjectedState, Command updates
4. **Checkpointing** - Persistence, isolation, reconstruction
5. **Subgraph Behavior** - Isolation, merging, errors
6. **Streaming** - Modes, channels, state updates

Current test coverage is primarily structural. The project needs significant expansion into behavioral testing, state transition testing, and edge case coverage.

Priority additions:
1. State reducer tests (file_reducer)
2. InjectedState integration tests
3. Checkpoint persistence tests
4. Subagent isolation tests
5. Middleware execution flow tests
6. Error handling tests

These additions will ensure the LangGraph-specific features of deepagents work correctly and robustly.
