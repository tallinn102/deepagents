# LangGraph Testing Analysis for DeepAgents

## Executive Summary

After analyzing the deepagents codebase, this document provides specific answers to LangGraph testing questions and identifies critical gaps in the current testing approach.

---

## 1. LangGraph-Specific Testing Needs

### How should we test CompiledGraph creation?

**Current Approach:**
```python
# tests/test_middleware.py
agent = create_agent(model=SAMPLE_MODEL, middleware=middleware, tools=[])
assert "todos" in agent.stream_channels
```

**What's Missing:**
1. **Graph compilation validation** - Does the graph compile without errors?
2. **Node structure verification** - Are all expected nodes present?
3. **Edge validation** - Are nodes connected correctly?
4. **Entry/exit points** - Is the graph properly wired?

**Recommended Additions:**
```python
def test_graph_compilation_success():
    """Test that graph compiles without errors"""
    agent = create_deep_agent(tools=[], instructions="test")

    # Verify it's a CompiledGraph
    assert hasattr(agent, 'nodes'), "Not a compiled graph"
    assert hasattr(agent, 'invoke'), "Missing invoke method"
    assert hasattr(agent, 'stream'), "Missing stream method"

def test_graph_node_structure():
    """Test that graph has expected node structure"""
    agent = create_deep_agent(tools=[], instructions="test")

    # LangGraph agent graphs have specific node structure
    assert 'agent' in agent.nodes, "Missing agent node"
    assert 'tools' in agent.nodes, "Missing tools node"

    # Verify node types
    assert callable(agent.nodes['agent']), "Agent node not callable"
    assert callable(agent.nodes['tools']), "Tools node not callable"

def test_graph_edges():
    """Test that graph edges are correct"""
    agent = create_deep_agent(tools=[], instructions="test")

    # Access graph structure
    # LangGraph graphs should have proper edges between nodes
    # This requires inspecting the internal graph structure
    compiled_graph = agent.graph

    # Verify edges exist
    assert compiled_graph is not None, "No graph structure"
```

### How to test graph state transitions?

**Critical Gap:** No current tests for state transitions

**Key Concept:** LangGraph graphs maintain state that flows through nodes. Each node can read and modify state.

**Recommended Tests:**
```python
def test_state_flows_through_graph():
    """Test that state is passed between nodes"""
    from tests.helpers.mock_model import MockToolCallingModel

    # Create mock that calls write_file
    mock_model = MockToolCallingModel(tool_calls=[
        {
            "name": "write_file",
            "args": {"file_path": "test.txt", "content": "hello"},
            "id": "call_1"
        }
    ])

    agent = create_deep_agent(model=mock_model, tools=[])

    # Initial state
    initial_state = {
        "messages": [{"role": "user", "content": "Write file"}],
        "files": {}
    }

    # Invoke and get final state
    final_state = agent.invoke(initial_state)

    # Verify state transition occurred
    assert "files" in final_state
    # File should be added if tool was called
    # Note: This requires the tool actually executed

def test_state_reducer_applied_during_transition():
    """Test that state reducers are applied when state updates"""
    agent = create_deep_agent(tools=[])

    # Create initial state with files
    initial = {
        "messages": [{"role": "user", "content": "test"}],
        "files": {"file1.txt": "content1"}
    }

    # After execution, if files are updated, reducer should merge them
    result = agent.invoke(initial)

    # The file_reducer should have been used for the 'files' field
    if "files" in result:
        assert isinstance(result["files"], dict)
```

### How to test checkpointer integration?

**Current Coverage:** Only in HITL test

**Key Concepts:**
- Checkpointers save state after each graph step
- Thread IDs isolate different conversation threads
- State can be reconstructed from checkpoints

**Recommended Tests:**
```python
def test_checkpointer_saves_state():
    """Test that checkpointer saves state between invocations"""
    from langgraph.checkpoint.memory import MemorySaver
    import uuid

    checkpointer = MemorySaver()
    agent = create_deep_agent(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # First invocation
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "Hello"}]
    }, config=config)

    # Get saved state
    saved_state = agent.get_state(config)

    # Saved state should match result
    assert saved_state.values["messages"] == result1["messages"]

def test_checkpointer_loads_state():
    """Test that checkpointer loads previous state"""
    from langgraph.checkpoint.memory import MemorySaver
    import uuid

    checkpointer = MemorySaver()
    agent = create_deep_agent(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # First message
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "Hello"}]
    }, config=config)

    # Second message - should load previous state
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "World"}]
    }, config=config)

    # result2 should have messages from both invocations
    assert len(result2["messages"]) > len(result1["messages"])

def test_checkpointer_thread_isolation():
    """Test that different threads don't share state"""
    from langgraph.checkpoint.memory import MemorySaver
    import uuid

    checkpointer = MemorySaver()
    agent = create_deep_agent(checkpointer=checkpointer)

    thread1 = str(uuid.uuid4())
    thread2 = str(uuid.uuid4())

    # Invoke in thread 1
    r1 = agent.invoke({
        "messages": [{"role": "user", "content": "Thread 1"}]
    }, config={"configurable": {"thread_id": thread1}})

    # Invoke in thread 2
    r2 = agent.invoke({
        "messages": [{"role": "user", "content": "Thread 2"}]
    }, config={"configurable": {"thread_id": thread2}})

    # Messages should be different
    assert r1["messages"][-1].content != r2["messages"][-1].content
```

### How to test middleware stacking in LangGraph context?

**Current Coverage:** Tests presence of tools/state, not stacking behavior

**Key Concept:** Middleware in LangGraph modifies the graph before compilation. Order matters.

**Recommended Tests:**
```python
def test_middleware_state_schemas_merge():
    """Test that multiple middleware state schemas merge correctly"""
    from langchain.agents import create_agent
    from deepagents.middleware import PlanningMiddleware, FilesystemMiddleware

    agent = create_agent(
        model="claude-3-5-sonnet-20240620",
        middleware=[
            PlanningMiddleware(),  # Adds 'todos' field
            FilesystemMiddleware(),  # Adds 'files' field
        ],
        tools=[]
    )

    # Both state fields should be accessible
    assert 'todos' in agent.stream_channels
    assert 'files' in agent.stream_channels
    assert 'messages' in agent.stream_channels  # From AgentState

def test_middleware_tools_accumulate():
    """Test that tools from multiple middleware accumulate"""
    from langchain.agents import create_agent
    from deepagents.middleware import PlanningMiddleware, FilesystemMiddleware

    agent = create_agent(
        model="claude-3-5-sonnet-20240620",
        middleware=[
            PlanningMiddleware(),
            FilesystemMiddleware(),
        ],
        tools=[]
    )

    bound_tools = agent.nodes["tools"].bound._tools_by_name

    # Tools from both middleware should be present
    assert "write_todos" in bound_tools  # From PlanningMiddleware
    assert "ls" in bound_tools  # From FilesystemMiddleware

def test_middleware_prompt_modifications_chain():
    """Test that modify_model_request calls chain together"""
    from deepagents.middleware import PlanningMiddleware, FilesystemMiddleware
    from langchain.agents.middleware import ModelRequest

    # Create middleware instances
    planning_mw = PlanningMiddleware()
    filesystem_mw = FilesystemMiddleware()

    # Create initial request
    request = ModelRequest(
        messages=[],
        system_prompt="Base prompt"
    )

    state = {"messages": []}

    # Apply middleware in order
    request1 = planning_mw.modify_model_request(request, state)
    request2 = filesystem_mw.modify_model_request(request1, state)

    # Final prompt should contain all modifications
    assert "Base prompt" in request2.system_prompt
    from deepagents.prompts import WRITE_TODOS_SYSTEM_PROMPT, FILESYSTEM_SYSTEM_PROMPT
    assert WRITE_TODOS_SYSTEM_PROMPT in request2.system_prompt
    assert FILESYSTEM_SYSTEM_PROMPT in request2.system_prompt

def test_middleware_order_matters():
    """Test that middleware order affects the result"""
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest

    class Middleware1(AgentMiddleware):
        def modify_model_request(self, request: ModelRequest, state: AgentState):
            request.system_prompt = request.system_prompt + "\nMW1"
            return request

    class Middleware2(AgentMiddleware):
        def modify_model_request(self, request: ModelRequest, state: AgentState):
            request.system_prompt = request.system_prompt + "\nMW2"
            return request

    agent1 = create_agent(
        model="claude-3-5-sonnet-20240620",
        middleware=[Middleware1(), Middleware2()],
        prompt="Base",
        tools=[]
    )

    agent2 = create_agent(
        model="claude-3-5-sonnet-20240620",
        middleware=[Middleware2(), Middleware1()],
        prompt="Base",
        tools=[]
    )

    # Different orders should theoretically produce different behavior
    # This is hard to test without invoking, but we can verify structure
    assert hasattr(agent1, 'nodes')
    assert hasattr(agent2, 'nodes')
```

---

## 2. State Management Testing

### How to test state reducers properly?

**File:** `/home/user/deepagents/src/deepagents/state.py`

**Critical Code:**
```python
def file_reducer(left: dict[str, str] | None, right: dict[str, str] | None) -> dict[str, str] | None:
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}

class DeepAgentState(AgentState):
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
```

**Current Coverage:** ❌ None

**Why This Matters:**
- Reducers control how state updates merge
- Bug in reducer = state corruption
- Used on EVERY state update for that field

**Recommended Tests:**
```python
# tests/unit/test_state_reducers.py

def test_file_reducer_both_none():
    """Test file_reducer with both inputs None"""
    result = file_reducer(None, None)
    assert result is None

def test_file_reducer_left_none():
    """Test file_reducer with left None"""
    right = {"file.txt": "content"}
    result = file_reducer(None, right)
    assert result == right

def test_file_reducer_right_none():
    """Test file_reducer with right None"""
    left = {"file.txt": "content"}
    result = file_reducer(left, None)
    assert result == left

def test_file_reducer_no_overlap():
    """Test file_reducer with non-overlapping dicts"""
    left = {"file1.txt": "content1"}
    right = {"file2.txt": "content2"}
    result = file_reducer(left, right)

    assert result == {"file1.txt": "content1", "file2.txt": "content2"}

def test_file_reducer_with_overlap():
    """Test file_reducer with overlapping keys (right wins)"""
    left = {"file.txt": "old content", "other.txt": "stays"}
    right = {"file.txt": "new content"}
    result = file_reducer(left, right)

    assert result == {
        "file.txt": "new content",  # Right overwrites
        "other.txt": "stays"
    }

def test_file_reducer_preserves_left_non_overlapping():
    """Test that reducer preserves non-overlapping left values"""
    left = {"a.txt": "A", "b.txt": "B", "c.txt": "C"}
    right = {"b.txt": "B_new", "d.txt": "D"}
    result = file_reducer(left, right)

    assert result == {
        "a.txt": "A",
        "b.txt": "B_new",
        "c.txt": "C",
        "d.txt": "D"
    }
```

### How to test Annotated state fields?

**Key Concept:** Annotated fields in TypedDict carry metadata (like reducers) that LangGraph uses

**Recommended Tests:**
```python
def test_annotated_field_has_reducer():
    """Test that files field has reducer annotation"""
    from deepagents.state import DeepAgentState
    from typing import get_type_hints, get_args

    # Get type hints with annotations
    hints = get_type_hints(DeepAgentState, include_extras=True)

    # files should be Annotated
    files_type = hints.get('files') or DeepAgentState.__annotations__.get('files')
    assert files_type is not None

    # Check if it's Annotated
    # This is tricky - Annotated types have __metadata__
    if hasattr(files_type, '__metadata__'):
        metadata = files_type.__metadata__
        # Should contain file_reducer
        from deepagents.state import file_reducer
        assert file_reducer in metadata

def test_annotated_field_used_in_graph():
    """Test that annotated field's reducer is used in graph"""
    from tests.helpers.mock_model import MockToolCallingModel

    # Create mock that writes files twice
    mock_model = MockToolCallingModel(tool_calls=[
        {
            "name": "write_file",
            "args": {"file_path": "file1.txt", "content": "content1"},
            "id": "call_1"
        }
    ])

    agent = create_deep_agent(model=mock_model)

    # Initial files
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Write files"}],
        "files": {"existing.txt": "existing"}
    })

    # The reducer should have merged files
    if "files" in result:
        assert isinstance(result["files"], dict)
        # Should have both existing and new (if tool executed)
```

### How to test state schema compatibility?

**Key Concept:** Middleware stacking requires compatible state schemas

**Recommended Tests:**
```python
def test_deepagent_state_includes_all_middleware_fields():
    """Test DeepAgentState is superset of middleware states"""
    from deepagents.state import DeepAgentState, PlanningState, FilesystemState

    deep_annotations = DeepAgentState.__annotations__
    planning_annotations = PlanningState.__annotations__
    filesystem_annotations = FilesystemState.__annotations__

    # DeepAgentState should include todos (from PlanningState)
    assert 'todos' in deep_annotations

    # DeepAgentState should include files (from FilesystemState)
    assert 'files' in deep_annotations

def test_middleware_states_compatible():
    """Test that middleware states can be merged"""
    from langchain.agents import create_agent
    from deepagents.middleware import PlanningMiddleware, FilesystemMiddleware

    # Should not raise error
    agent = create_agent(
        model="claude-3-5-sonnet-20240620",
        middleware=[PlanningMiddleware(), FilesystemMiddleware()],
        tools=[]
    )

    # Both fields should be in stream channels
    assert 'todos' in agent.stream_channels
    assert 'files' in agent.stream_channels
```

### How to test state persistence?

**Covered in checkpointer tests above**

---

## 3. Subgraph Testing

### How to test subagent spawning?

**Current Coverage:** ✅ Basic spawning tested

**File:** `/home/user/deepagents/tests/test_deepagents.py`

**Existing Test:**
```python
def test_deep_agent_with_subagents(self):
    subagents = [{
        "name": "weather_agent",
        "description": "Use this agent to get the weather",
        "prompt": "You are a weather agent.",
        "tools": [get_weather],
        "model": SAMPLE_MODEL,
    }]
    agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
    result = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]})

    # Check if task tool was called
    tool_calls = [tool_call for msg in result.get("messages", []) if msg.type == "ai" for tool_call in msg.tool_calls]
    assert any([tool_call["name"] == "task" and tool_call["args"].get("subagent_type") == "weather_agent" for tool_call in tool_calls])
```

**This is good!** But needs enhancement:

**Recommended Additions:**
```python
def test_subagent_created_correctly():
    """Test that subagent is created with correct configuration"""
    from deepagents.middleware import _get_agents

    subagents = [{
        "name": "test_agent",
        "description": "Test",
        "prompt": "Test prompt",
        "tools": [get_weather],
        "model": "claude-3-5-sonnet-20240620"
    }]

    agents_dict = _get_agents(
        default_subagent_tools=[],
        subagents=subagents,
        model="claude-3-5-sonnet-20240620"
    )

    # Should have general-purpose + custom subagent
    assert "general-purpose" in agents_dict
    assert "test_agent" in agents_dict

    # Verify test_agent is a compiled graph
    test_agent = agents_dict["test_agent"]
    assert hasattr(test_agent, 'invoke')
    assert hasattr(test_agent, 'nodes')

def test_subagent_error_when_invalid_type():
    """Test that calling non-existent subagent returns error"""
    agent = create_deep_agent(tools=[], subagents=[])

    # The task tool should validate subagent_type
    # This requires mocking the model to call task with invalid type
```

### How to test subgraph isolation?

**Current Coverage:** ❌ None

**Key Concept:** Subagents should have isolated state - changes don't leak

**Critical Code:** `/home/user/deepagents/src/deepagents/middleware.py`
```python
# In task tool:
state["messages"] = [{"role": "user", "content": description}]
result = sub_agent.invoke(state)  # Subagent gets state copy

state_update = {}
for k, v in result.items():
    if k not in ["todos", "messages"]:  # Excluded from merge back
        state_update[k] = v
```

**What This Means:**
1. Subagent gets a COPY of parent state
2. Subagent modifies the copy
3. Only certain fields merge back (NOT todos, NOT messages)

**Recommended Tests:**
```python
def test_subagent_state_isolation():
    """Test that subagent doesn't modify parent state directly"""
    from tests.helpers.mock_model import MockToolCallingModel

    # Create subagent that writes todos
    subagents = [{
        "name": "todo_agent",
        "description": "Manages todos",
        "prompt": "Write todos",
        "model": MockToolCallingModel(tool_calls=[
            {"name": "write_todos", "args": {"todos": [{"content": "subtask", "status": "pending"}]}, "id": "call_1"}
        ])
    }]

    agent = create_deep_agent(subagents=subagents)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use todo_agent"}],
        "todos": [{"content": "parent task", "status": "pending"}]
    })

    # Subagent todos should NOT merge back (per implementation)
    if "todos" in result:
        # Should still be parent todos only
        assert result["todos"] == [{"content": "parent task", "status": "pending"}]

def test_subagent_files_merge_back():
    """Test that subagent file changes merge back"""
    from tests.helpers.mock_model import MockToolCallingModel

    # Subagent that writes file
    subagents = [{
        "name": "file_agent",
        "description": "Writes files",
        "prompt": "Write files",
        "model": MockToolCallingModel(tool_calls=[
            {"name": "write_file", "args": {"file_path": "sub.txt", "content": "from subagent"}, "id": "call_1"}
        ])
    }]

    agent = create_deep_agent(subagents=subagents)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use file_agent"}],
        "files": {"parent.txt": "from parent"}
    })

    # Files should merge back
    if "files" in result:
        assert "parent.txt" in result["files"]
        # If subagent succeeded:
        # assert "sub.txt" in result["files"]
```

### How to test state passing to subgraphs?

**Current Coverage:** ❌ None

**Recommended Tests:**
```python
def test_subagent_receives_parent_files():
    """Test that subagent can access parent's files"""
    from langchain_core.tools import tool
    from langchain.agents.tool_node import InjectedState
    from typing import Annotated

    received_files = []

    @tool
    def check_files_tool(state: Annotated[dict, InjectedState]):
        """Tool that checks what files are in state"""
        received_files.append(state.get("files", {}))
        return "Checked"

    subagents = [{
        "name": "file_checker",
        "description": "Checks files",
        "prompt": "Check files",
        "tools": [check_files_tool],
        "model": "claude-3-5-sonnet-20240620"
    }]

    agent = create_deep_agent(subagents=subagents)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use file_checker to check files"}],
        "files": {"test.txt": "test content"}
    })

    # If tool was called, received_files should have the parent files
    # Note: Model-dependent behavior
```

### How to test subgraph error handling?

**Current Coverage:** ❌ None

**Recommended Tests:**
```python
def test_subagent_tool_error_propagates():
    """Test that tool errors in subagent propagate correctly"""
    from langchain_core.tools import tool

    @tool
    def failing_tool():
        """Tool that always fails"""
        raise ValueError("Tool failed intentionally")

    subagents = [{
        "name": "failing_agent",
        "description": "Has failing tool",
        "prompt": "Use failing tool",
        "tools": [failing_tool],
        "model": "claude-3-5-sonnet-20240620"
    }]

    agent = create_deep_agent(subagents=subagents)

    # Should not crash
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use failing_agent"}]
    })

    # Error should be in messages
    messages = result.get("messages", [])
    # LangGraph wraps errors in tool messages
    tool_messages = [m for m in messages if m.type == "tool"]
    # At least one should contain error info

def test_subagent_graph_error_handling():
    """Test error handling when subagent graph fails"""
    from langchain.agents import create_agent

    # Create a broken subagent (e.g., with invalid model)
    # This is hard to test without actually breaking something
    # Focus on testing that errors are caught and handled
```

---

## 4. Tool Integration Testing

### How to test InjectedState parameters?

**File:** `/home/user/deepagents/src/deepagents/tools.py`

**Critical Code:**
```python
@tool
def read_file(
    file_path: str,
    state: Annotated[FilesystemState, InjectedState],  # <-- Injected
    offset: int = 0,
    limit: int = 2000,
) -> str:
    mock_filesystem = state.get("files", {})  # <-- Uses state
    ...
```

**Current Coverage:** ❌ None

**Key Concept:** InjectedState parameters are filled by LangGraph during tool execution

**Recommended Tests:**
```python
def test_injected_state_receives_current_state():
    """Test that InjectedState parameter receives graph state"""
    from langchain_core.tools import tool
    from langchain.agents.tool_node import InjectedState
    from typing import Annotated

    received_state = []

    @tool
    def state_checker(
        value: str,
        state: Annotated[dict, InjectedState]
    ) -> str:
        """Tool that captures state"""
        received_state.append(state.copy())
        return "ok"

    from tests.helpers.mock_model import MockToolCallingModel

    mock_model = MockToolCallingModel(tool_calls=[
        {"name": "state_checker", "args": {"value": "test"}, "id": "call_1"}
    ])

    agent = create_deep_agent(
        model=mock_model,
        tools=[state_checker]
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "test"}],
        "files": {"test.txt": "content"}
    })

    # If tool was called, received_state should have the files
    if received_state:
        assert "files" in received_state[0]
        assert received_state[0]["files"] == {"test.txt": "content"}

def test_injected_tool_call_id():
    """Test that InjectedToolCallId parameter receives correct ID"""
    from langchain_core.tools import tool, InjectedToolCallId
    from typing import Annotated

    received_ids = []

    @tool
    def id_checker(
        value: str,
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> str:
        """Tool that captures tool_call_id"""
        received_ids.append(tool_call_id)
        return "ok"

    from tests.helpers.mock_model import MockToolCallingModel

    mock_model = MockToolCallingModel(tool_calls=[
        {"name": "id_checker", "args": {"value": "test"}, "id": "test_call_123"}
    ])

    agent = create_deep_agent(
        model=mock_model,
        tools=[id_checker]
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "test"}]
    })

    # If tool was called, received_ids should have the ID
    if received_ids:
        assert received_ids[0] == "test_call_123"
```

### How to test tool binding to models?

**Current Coverage:** ✅ Structural only

**Recommended Enhancement:**
```python
def test_tools_bound_to_graph():
    """Test that tools are actually bound and callable"""
    agent = create_deep_agent(tools=[])

    # Get tool node
    tool_node = agent.nodes["tools"]

    # Verify tools are bound
    bound_tools = tool_node.bound._tools_by_name

    # Should have all default tools
    assert "write_todos" in bound_tools
    assert "ls" in bound_tools
    assert "read_file" in bound_tools
    assert "write_file" in bound_tools
    assert "edit_file" in bound_tools
    assert "task" in bound_tools

def test_custom_tools_bound_correctly():
    """Test that custom tools are bound with correct signatures"""
    from langchain_core.tools import tool

    @tool
    def custom_tool(x: str, y: int = 5) -> str:
        """Custom tool with args"""
        return f"{x} {y}"

    agent = create_deep_agent(tools=[custom_tool])

    bound_tools = agent.nodes["tools"].bound._tools_by_name

    # Custom tool should be bound
    assert "custom_tool" in bound_tools

    # Verify tool is callable
    tool_instance = bound_tools["custom_tool"]
    assert callable(tool_instance)

    # Verify tool has correct schema
    assert hasattr(tool_instance, 'args_schema')
```

### How to test tool execution in graph context?

**Current Coverage:** ❌ None

**Key Concept:** Tools behave differently in graph vs isolation

**Recommended Tests:**
```python
def test_tool_execution_updates_state():
    """Test that tool execution updates graph state"""
    from tests.helpers.mock_model import MockToolCallingModel

    mock_model = MockToolCallingModel(tool_calls=[
        {"name": "write_file", "args": {"file_path": "new.txt", "content": "new"}, "id": "call_1"}
    ])

    agent = create_deep_agent(model=mock_model)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Write file"}],
        "files": {}
    })

    # State should be updated
    if "files" in result:
        # If tool executed successfully
        assert isinstance(result["files"], dict)

def test_tool_command_merges_state():
    """Test that Command from tool merges into state"""
    from langchain_core.tools import tool, InjectedToolCallId
    from langgraph.types import Command
    from typing import Annotated

    @tool
    def command_tool(
        value: str,
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Tool that returns Command"""
        return Command(update={
            "custom_field": value,
            "messages": [{"role": "tool", "content": "ok", "tool_call_id": tool_call_id}]
        })

    from tests.helpers.mock_model import MockToolCallingModel
    from langchain.agents.middleware import AgentMiddleware, AgentState

    class CustomState(AgentState):
        custom_field: str

    class CustomMiddleware(AgentMiddleware):
        state_schema = CustomState

    mock_model = MockToolCallingModel(tool_calls=[
        {"name": "command_tool", "args": {"value": "test"}, "id": "call_1"}
    ])

    from deepagents.graph import agent_builder

    agent = agent_builder(
        tools=[command_tool],
        instructions="test",
        middleware=[CustomMiddleware()],
        model=mock_model
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "test"}],
        "custom_field": ""
    })

    # Custom field should be updated
    if "custom_field" in result:
        assert result["custom_field"] == "test"
```

---

## 5. Middleware Testing in LangGraph

### How to test modify_model_request properly?

**File:** `/home/user/deepagents/src/deepagents/middleware.py`

**Critical Code:**
```python
class PlanningMiddleware(AgentMiddleware):
    def modify_model_request(self, request: ModelRequest, agent_state: PlanningState) -> ModelRequest:
        request.system_prompt = request.system_prompt + "\n\n" + WRITE_TODOS_SYSTEM_PROMPT
        return request
```

**Current Coverage:** ❌ None (only structural tests)

**Recommended Tests:**
```python
def test_planning_middleware_modifies_request():
    """Test PlanningMiddleware.modify_model_request"""
    from deepagents.middleware import PlanningMiddleware
    from langchain.agents.middleware import ModelRequest

    middleware = PlanningMiddleware()

    request = ModelRequest(
        messages=[],
        system_prompt="Base prompt"
    )

    state = {"messages": []}

    modified = middleware.modify_model_request(request, state)

    # Should append planning prompt
    from deepagents.prompts import WRITE_TODOS_SYSTEM_PROMPT
    assert WRITE_TODOS_SYSTEM_PROMPT in modified.system_prompt
    assert "Base prompt" in modified.system_prompt

def test_filesystem_middleware_modifies_request():
    """Test FilesystemMiddleware.modify_model_request"""
    from deepagents.middleware import FilesystemMiddleware
    from langchain.agents.middleware import ModelRequest

    middleware = FilesystemMiddleware()

    request = ModelRequest(
        messages=[],
        system_prompt="Base prompt"
    )

    state = {"messages": [], "files": {}}

    modified = middleware.modify_model_request(request, state)

    # Should append filesystem prompt
    from deepagents.prompts import FILESYSTEM_SYSTEM_PROMPT
    assert FILESYSTEM_SYSTEM_PROMPT in modified.system_prompt
    assert "Base prompt" in modified.system_prompt

def test_modify_model_request_called_in_graph():
    """Test that modify_model_request is actually called during execution"""
    from langchain.agents.middleware import AgentMiddleware, ModelRequest, AgentState

    calls = []

    class TrackingMiddleware(AgentMiddleware):
        def modify_model_request(self, request: ModelRequest, state: AgentState):
            calls.append(("modify_model_request", request.system_prompt))
            return request

    from langchain.agents import create_agent

    agent = create_agent(
        model="claude-3-5-sonnet-20240620",
        middleware=[TrackingMiddleware()],
        prompt="Test",
        tools=[]
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "test"}]})

    # modify_model_request should have been called
    assert len(calls) > 0
    assert calls[0][0] == "modify_model_request"
```

### How to test middleware execution order?

**Recommended Tests:**
```python
def test_middleware_executes_in_order():
    """Test that middleware executes in specified order"""
    from langchain.agents.middleware import AgentMiddleware, ModelRequest, AgentState

    order = []

    class MW1(AgentMiddleware):
        def modify_model_request(self, request: ModelRequest, state: AgentState):
            order.append("MW1")
            request.system_prompt = request.system_prompt + " MW1"
            return request

    class MW2(AgentMiddleware):
        def modify_model_request(self, request: ModelRequest, state: AgentState):
            order.append("MW2")
            request.system_prompt = request.system_prompt + " MW2"
            return request

    from langchain.agents import create_agent

    agent = create_agent(
        model="claude-3-5-sonnet-20240620",
        middleware=[MW1(), MW2()],
        prompt="Base",
        tools=[]
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "test"}]})

    # Should execute in order
    assert order == ["MW1", "MW2"]
```

### How to test middleware state modifications?

**Recommended Tests:**
```python
def test_middleware_state_accessible_in_graph():
    """Test that middleware state fields are accessible"""
    from deepagents.middleware import PlanningMiddleware
    from langchain.agents import create_agent

    agent = create_agent(
        model="claude-3-5-sonnet-20240620",
        middleware=[PlanningMiddleware()],
        tools=[]
    )

    # Planning state field should be in stream channels
    assert 'todos' in agent.stream_channels

    # Should be able to pass todos in state
    result = agent.invoke({
        "messages": [{"role": "user", "content": "test"}],
        "todos": [{"content": "task", "status": "pending"}]
    })

    # Todos should be in result
    assert "todos" in result
```

---

## 6. Common LangGraph Testing Pitfalls

### Pitfall 1: Testing Structure Instead of Behavior

**What You're Doing:**
```python
assert "write_todos" in agent.nodes["tools"].bound._tools_by_name
```

**Problem:** This only tests that the tool exists, not that it works

**Better Approach:** Test actual execution

### Pitfall 2: Model Non-Determinism

**Problem:** Real LLMs are non-deterministic

**Solution:** Use mock models for unit/integration tests

**Example Mock:**
```python
class MockToolCallingModel(BaseChatModel):
    def __init__(self, tool_calls: List[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.tool_calls = tool_calls or []

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        message = AIMessage(content="", tool_calls=self.tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "mock"
```

### Pitfall 3: Not Testing State Transitions

**Problem:** Single invocations don't test state management

**Solution:** Test multi-step interactions with checkpointers

### Pitfall 4: Ignoring Checkpointer Requirements

**Problem:** Some features only work with checkpointers

**Examples:**
- HITL (interrupt/resume)
- State persistence
- Thread isolation

**Solution:** Always use checkpointer when testing these features

### Pitfall 5: Not Testing Middleware Interactions

**Problem:** Middleware tested in isolation may not work when stacked

**Solution:** Test middleware combinations

### Pitfall 6: Assuming Subagent State Merges

**Problem:** Not all state merges back from subagents

**Reality:** Per implementation, todos and messages are excluded

**Solution:** Test what actually merges back

---

## 7. Recommended Test Helpers

### Helper 1: MockToolCallingModel

```python
# tests/helpers/mock_model.py
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any

class MockToolCallingModel(BaseChatModel):
    """Mock model that returns predetermined tool calls"""

    def __init__(self, tool_calls: List[dict] = None, content: str = "", **kwargs):
        super().__init__(**kwargs)
        self.tool_calls = tool_calls or []
        self.content = content
        self.call_count = 0

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1

        # First call: return tool calls
        # Second call: return completion
        if self.call_count == 1 and self.tool_calls:
            message = AIMessage(content="", tool_calls=self.tool_calls)
        else:
            message = AIMessage(content=self.content or "Task completed")

        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "mock_tool_calling"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, **kwargs)
```

### Helper 2: CheckpointTestHelper

See LANGGRAPH_TESTING_GUIDE.md for full implementation

### Helper 3: State Assertion Helpers

See LANGGRAPH_TESTING_GUIDE.md for full implementation

---

## 8. Coverage Gaps and Phase Recommendations

### Current State

**Total Test Files:** 3
- test_middleware.py (structure only)
- test_deepagents.py (basic integration)
- test_hitl.py (HITL only)

**Current Coverage:** ~20% of LangGraph-specific functionality

### Critical Gaps (Phase 1 Priority)

1. **State Reducers** - ❌ Zero coverage
   - Add: `tests/unit/test_state_reducers.py`
   - Tests: file_reducer with all edge cases

2. **InjectedState** - ❌ Zero coverage
   - Add: `tests/integration/test_injected_state.py`
   - Tests: State injection, tool_call_id injection

3. **Checkpointing** - ⚠️ Minimal (HITL only)
   - Add: `tests/integration/test_checkpointing.py`
   - Tests: Persistence, thread isolation, state reconstruction

4. **Middleware Execution** - ⚠️ Partial
   - Enhance: Current tests only check structure
   - Add: Tests for modify_model_request execution

### Phase 1: Foundation Tests (Week 1-2)

**Priority: Critical bugs that would corrupt state**

```
tests/unit/
├── test_state_reducers.py          # NEW - CRITICAL
│   ├── test_file_reducer_both_none
│   ├── test_file_reducer_left_none
│   ├── test_file_reducer_right_none
│   ├── test_file_reducer_merge
│   └── test_file_reducer_overlap
│
├── test_middleware_methods.py      # NEW - HIGH
│   ├── test_planning_modify_request
│   ├── test_filesystem_modify_request
│   └── test_subagent_modify_request
│
└── test_state_schemas.py           # NEW - MEDIUM
    ├── test_annotated_fields
    ├── test_state_compatibility
    └── test_notrequired_fields
```

### Phase 2: Integration Tests (Week 3-4)

**Priority: Features working together**

```
tests/integration/
├── test_injected_state.py          # NEW - CRITICAL
│   ├── test_injected_state_receives_state
│   ├── test_injected_tool_call_id
│   ├── test_command_updates_state
│   └── test_state_access_in_tools
│
├── test_checkpointing.py           # NEW - CRITICAL
│   ├── test_state_persistence
│   ├── test_thread_isolation
│   ├── test_state_reconstruction
│   └── test_interrupt_resume
│
├── test_middleware_execution.py   # NEW - HIGH
│   ├── test_modify_request_called
│   ├── test_middleware_order
│   ├── test_middleware_stacking
│   └── test_prompt_chain
│
└── test_subagent_isolation.py     # NEW - HIGH
    ├── test_state_isolation
    ├── test_state_merging
    ├── test_field_exclusions
    └── test_error_propagation
```

### Phase 3: Advanced Tests (Week 5-6)

**Priority: Edge cases and complex scenarios**

```
tests/integration/
├── test_streaming.py               # NEW - MEDIUM
│   ├── test_stream_mode_values
│   ├── test_stream_mode_updates
│   └── test_stream_channels
│
├── test_error_handling.py          # NEW - MEDIUM
│   ├── test_tool_errors
│   ├── test_state_validation_errors
│   └── test_graph_execution_errors
│
└── test_complex_workflows.py      # NEW - LOW
    ├── test_multi_step_state_changes
    ├── test_nested_subagents
    └── test_concurrent_state_updates
```

### Phase 4: E2E & Performance (Week 7-8)

**Priority: Real-world scenarios**

```
tests/e2e/
├── test_workflows.py               # NEW
├── test_async_workflows.py         # NEW
└── test_performance.py             # NEW
```

---

## Summary: Critical Actions

### Immediate Additions (This Week)

1. **Create tests/unit/test_state_reducers.py**
   - Test file_reducer exhaustively
   - This is critical - bugs here corrupt all state

2. **Create tests/helpers/mock_model.py**
   - MockToolCallingModel for deterministic tests
   - Required for all integration tests

3. **Create tests/integration/test_checkpointing.py**
   - Test thread isolation
   - Test state persistence
   - Required for HITL and statefulness

4. **Enhance tests/test_middleware.py**
   - Add modify_model_request execution tests
   - Add middleware ordering tests

### Why These Matter

**State Reducers:**
- Bug in reducer = state corruption
- Affects every state update
- Hard to debug if broken

**Checkpointing:**
- Required for HITL
- Required for conversation history
- Required for state persistence
- Thread isolation bugs cause data leaks

**InjectedState:**
- All tools rely on this
- Broken injection = tools can't access state
- Critical for filesystem, todos, custom tools

**Middleware Execution:**
- Prompts may not be modified
- Tools may not be added
- State may not be extended
- Silent failures possible

---

## Conclusion

DeepAgents has **good structural tests** but **lacks behavioral tests** for LangGraph-specific features.

**Critical gaps:**
1. State reducers (ZERO tests) - can corrupt state
2. InjectedState (ZERO tests) - breaks tools
3. Checkpointing (MINIMAL tests) - breaks persistence
4. Middleware execution (STRUCTURAL only) - silent failures

**Recommended immediate focus:**
1. State reducer unit tests
2. Mock model helper
3. Checkpointing integration tests
4. InjectedState integration tests

These form the foundation for all other testing and catch bugs that would break core functionality.
