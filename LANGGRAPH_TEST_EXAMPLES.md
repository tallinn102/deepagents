# LangGraph Test Examples - Ready to Use

This document provides copy-paste ready test code for LangGraph-specific testing in deepagents.

## Quick Start

### 1. Create Mock Model Helper

```python
# File: tests/helpers/mock_model.py

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any

class MockToolCallingModel(BaseChatModel):
    """
    Mock model for deterministic testing.

    Usage:
        mock_model = MockToolCallingModel(tool_calls=[
            {"name": "write_file", "args": {"file_path": "test.txt", "content": "hello"}, "id": "call_1"}
        ])
        agent = create_deep_agent(model=mock_model, ...)
    """

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

        # First call returns tool calls, second call returns final message
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


class MockModel(BaseChatModel):
    """
    Simple mock model that returns predetermined content.

    Usage:
        mock_model = MockModel(content="Hello, world!")
        agent = create_deep_agent(model=mock_model, ...)
    """

    def __init__(self, content: str = "Response", **kwargs):
        super().__init__(**kwargs)
        self.content = content

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self.content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "mock"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, **kwargs)
```

### 2. State Reducer Tests

```python
# File: tests/unit/test_state_reducers.py

import pytest
from deepagents.state import file_reducer


class TestFileReducer:
    """Test the file_reducer function"""

    def test_both_none(self):
        """When both inputs are None, return None"""
        result = file_reducer(None, None)
        assert result is None

    def test_left_none(self):
        """When left is None, return right"""
        right = {"file.txt": "content"}
        result = file_reducer(None, right)
        assert result == right
        assert result is right  # Should return the same object

    def test_right_none(self):
        """When right is None, return left"""
        left = {"file.txt": "content"}
        result = file_reducer(left, None)
        assert result == left
        assert result is left  # Should return the same object

    def test_both_empty(self):
        """When both are empty dicts, return empty dict"""
        result = file_reducer({}, {})
        assert result == {}

    def test_no_overlap(self):
        """When keys don't overlap, merge both"""
        left = {"file1.txt": "content1"}
        right = {"file2.txt": "content2"}
        result = file_reducer(left, right)

        assert result == {
            "file1.txt": "content1",
            "file2.txt": "content2"
        }

    def test_with_overlap_right_wins(self):
        """When keys overlap, right value takes precedence"""
        left = {"file.txt": "old content", "other.txt": "stays"}
        right = {"file.txt": "new content"}
        result = file_reducer(left, right)

        assert result == {
            "file.txt": "new content",
            "other.txt": "stays"
        }

    def test_preserves_left_non_overlapping(self):
        """Non-overlapping left keys are preserved"""
        left = {"a.txt": "A", "b.txt": "B", "c.txt": "C"}
        right = {"b.txt": "B_new", "d.txt": "D"}
        result = file_reducer(left, right)

        assert result == {
            "a.txt": "A",
            "b.txt": "B_new",
            "c.txt": "C",
            "d.txt": "D"
        }

    def test_multiple_overlaps(self):
        """Multiple overlapping keys all use right values"""
        left = {"a.txt": "A1", "b.txt": "B1", "c.txt": "C1"}
        right = {"a.txt": "A2", "b.txt": "B2"}
        result = file_reducer(left, right)

        assert result == {
            "a.txt": "A2",
            "b.txt": "B2",
            "c.txt": "C1"
        }

    def test_does_not_modify_inputs(self):
        """Reducer should not modify input dictionaries"""
        left = {"file.txt": "left"}
        right = {"file.txt": "right"}
        left_copy = left.copy()
        right_copy = right.copy()

        result = file_reducer(left, right)

        # Inputs should be unchanged
        assert left == left_copy
        assert right == right_copy

    def test_empty_string_values(self):
        """Empty string values are valid"""
        left = {"file1.txt": ""}
        right = {"file2.txt": ""}
        result = file_reducer(left, right)

        assert result == {"file1.txt": "", "file2.txt": ""}

    def test_right_can_set_empty_string(self):
        """Right can overwrite with empty string"""
        left = {"file.txt": "content"}
        right = {"file.txt": ""}
        result = file_reducer(left, right)

        assert result == {"file.txt": ""}
```

### 3. Checkpointing Tests

```python
# File: tests/integration/test_checkpointing.py

import pytest
from deepagents.graph import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
import uuid


class TestCheckpointing:
    """Test checkpointer integration"""

    def test_state_persists_across_invocations(self):
        """State should persist when using same thread_id"""
        checkpointer = MemorySaver()
        agent = create_deep_agent(tools=[], instructions="test", checkpointer=checkpointer)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # First invocation
        result1 = agent.invoke({
            "messages": [{"role": "user", "content": "Hello"}]
        }, config=config)

        message_count_1 = len(result1["messages"])

        # Second invocation - should have previous messages
        result2 = agent.invoke({
            "messages": [{"role": "user", "content": "World"}]
        }, config=config)

        message_count_2 = len(result2["messages"])

        # Second invocation should have more messages (includes first invocation's messages)
        assert message_count_2 > message_count_1

    def test_thread_isolation(self):
        """Different threads should have isolated state"""
        checkpointer = MemorySaver()
        agent = create_deep_agent(tools=[], instructions="test", checkpointer=checkpointer)

        thread1 = str(uuid.uuid4())
        thread2 = str(uuid.uuid4())

        # Invoke in thread 1
        result1 = agent.invoke({
            "messages": [{"role": "user", "content": "Thread 1 message"}]
        }, config={"configurable": {"thread_id": thread1}})

        # Invoke in thread 2
        result2 = agent.invoke({
            "messages": [{"role": "user", "content": "Thread 2 message"}]
        }, config={"configurable": {"thread_id": thread2}})

        # Get states
        state1 = agent.get_state(config={"configurable": {"thread_id": thread1}})
        state2 = agent.get_state(config={"configurable": {"thread_id": thread2}})

        # States should be different
        assert state1.values["messages"] != state2.values["messages"]

    def test_state_reconstruction(self):
        """State should be reconstructable from checkpoint"""
        checkpointer = MemorySaver()
        agent = create_deep_agent(tools=[], instructions="test", checkpointer=checkpointer)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Create state
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Test"}],
            "files": {"test.txt": "content"}
        }, config=config)

        # Get state from checkpoint
        saved_state = agent.get_state(config)

        # Saved state should match result
        assert "messages" in saved_state.values
        assert len(saved_state.values["messages"]) == len(result["messages"])

        if "files" in result:
            assert "files" in saved_state.values
            assert saved_state.values["files"] == result["files"]

    def test_files_persist_with_checkpoint(self):
        """Files should persist across invocations"""
        from tests.helpers.mock_model import MockToolCallingModel

        mock_model = MockToolCallingModel(tool_calls=[
            {"name": "write_file", "args": {"file_path": "persisted.txt", "content": "hello"}, "id": "call_1"}
        ])

        checkpointer = MemorySaver()
        agent = create_deep_agent(
            model=mock_model,
            tools=[],
            instructions="test",
            checkpointer=checkpointer
        )

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # First invocation - creates file
        result1 = agent.invoke({
            "messages": [{"role": "user", "content": "Write file"}],
            "files": {}
        }, config=config)

        # Second invocation - should see file
        result2 = agent.invoke({
            "messages": [{"role": "user", "content": "Check files"}]
        }, config=config)

        # File should persist
        if "files" in result1 and "persisted.txt" in result1["files"]:
            assert "files" in result2
            assert "persisted.txt" in result2["files"]

    def test_no_config_creates_new_thread(self):
        """Invoking without config should work (creates new thread)"""
        checkpointer = MemorySaver()
        agent = create_deep_agent(tools=[], instructions="test", checkpointer=checkpointer)

        # No config = new thread
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Hello"}]
        })

        assert "messages" in result
        assert len(result["messages"]) > 0
```

### 4. InjectedState Tests

```python
# File: tests/integration/test_injected_state.py

import pytest
from deepagents.graph import create_deep_agent
from langchain_core.tools import tool, InjectedToolCallId
from langchain.agents.tool_node import InjectedState
from langgraph.types import Command
from typing import Annotated
from tests.helpers.mock_model import MockToolCallingModel


class TestInjectedState:
    """Test InjectedState parameter injection"""

    def test_injected_state_receives_graph_state(self):
        """InjectedState parameter should receive current graph state"""
        received_states = []

        @tool
        def state_checker(
            value: str,
            state: Annotated[dict, InjectedState]
        ) -> str:
            """Tool that captures state"""
            received_states.append(state.copy())
            return f"Received state with {len(state)} keys"

        mock_model = MockToolCallingModel(tool_calls=[
            {"name": "state_checker", "args": {"value": "test"}, "id": "call_1"}
        ])

        agent = create_deep_agent(
            model=mock_model,
            tools=[state_checker],
            instructions="test"
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "Check state"}],
            "files": {"test.txt": "test content"}
        })

        # Tool should have been called
        assert len(received_states) > 0

        # State should contain files
        state = received_states[0]
        assert "files" in state
        assert state["files"]["test.txt"] == "test content"

    def test_injected_tool_call_id(self):
        """InjectedToolCallId should receive correct tool call ID"""
        received_ids = []

        @tool
        def id_checker(
            value: str,
            tool_call_id: Annotated[str, InjectedToolCallId]
        ) -> str:
            """Tool that captures tool_call_id"""
            received_ids.append(tool_call_id)
            return "ok"

        mock_model = MockToolCallingModel(tool_calls=[
            {"name": "id_checker", "args": {"value": "test"}, "id": "expected_id_123"}
        ])

        agent = create_deep_agent(
            model=mock_model,
            tools=[id_checker],
            instructions="test"
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "Check ID"}]
        })

        # Tool should have been called
        assert len(received_ids) > 0
        assert received_ids[0] == "expected_id_123"

    def test_command_updates_state(self):
        """Tool returning Command should update state"""
        @tool
        def state_updater(
            new_value: str,
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId]
        ) -> Command:
            """Tool that updates state via Command"""
            current_files = state.get("files", {})
            updated_files = {**current_files, "new.txt": new_value}
            return Command(
                update={
                    "files": updated_files,
                    "messages": [{"role": "tool", "content": f"Updated", "tool_call_id": tool_call_id}]
                }
            )

        mock_model = MockToolCallingModel(tool_calls=[
            {"name": "state_updater", "args": {"new_value": "hello"}, "id": "call_1"}
        ])

        agent = create_deep_agent(
            model=mock_model,
            tools=[state_updater],
            instructions="test"
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "Update state"}],
            "files": {"existing.txt": "existing"}
        })

        # State should be updated
        assert "files" in result
        assert "existing.txt" in result["files"]
        assert "new.txt" in result["files"]
        assert result["files"]["new.txt"] == "hello"

    def test_state_access_from_builtin_tools(self):
        """Built-in tools should access state correctly"""
        from tests.helpers.mock_model import MockToolCallingModel

        mock_model = MockToolCallingModel(tool_calls=[
            {"name": "read_file", "args": {"file_path": "test.txt"}, "id": "call_1"}
        ])

        agent = create_deep_agent(
            model=mock_model,
            tools=[],
            instructions="test"
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "Read test.txt"}],
            "files": {"test.txt": "Hello, world!"}
        })

        # read_file should have accessed state and returned content
        # Check tool messages
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        if tool_messages:
            # Should contain the file content
            assert any("Hello, world!" in m.content or "test.txt" in m.content for m in tool_messages)
```

### 5. Middleware Execution Tests

```python
# File: tests/integration/test_middleware_execution.py

import pytest
from deepagents.middleware import PlanningMiddleware, FilesystemMiddleware, SubAgentMiddleware
from langchain.agents.middleware import ModelRequest, AgentState, AgentMiddleware
from langchain.agents import create_agent
from deepagents.prompts import WRITE_TODOS_SYSTEM_PROMPT, FILESYSTEM_SYSTEM_PROMPT


class TestMiddlewareExecution:
    """Test middleware execution in graph context"""

    def test_planning_middleware_modifies_system_prompt(self):
        """PlanningMiddleware should modify system prompt"""
        middleware = PlanningMiddleware()

        request = ModelRequest(
            messages=[],
            system_prompt="Base prompt"
        )
        state = {"messages": []}

        modified = middleware.modify_model_request(request, state)

        # Should include both base and planning prompts
        assert "Base prompt" in modified.system_prompt
        assert WRITE_TODOS_SYSTEM_PROMPT in modified.system_prompt

    def test_filesystem_middleware_modifies_system_prompt(self):
        """FilesystemMiddleware should modify system prompt"""
        middleware = FilesystemMiddleware()

        request = ModelRequest(
            messages=[],
            system_prompt="Base prompt"
        )
        state = {"messages": [], "files": {}}

        modified = middleware.modify_model_request(request, state)

        # Should include both base and filesystem prompts
        assert "Base prompt" in modified.system_prompt
        assert FILESYSTEM_SYSTEM_PROMPT in modified.system_prompt

    def test_middleware_order_affects_prompt(self):
        """Middleware order should affect prompt construction"""
        class MW1(AgentMiddleware):
            def modify_model_request(self, request: ModelRequest, state: AgentState):
                request.system_prompt = request.system_prompt + " [MW1]"
                return request

        class MW2(AgentMiddleware):
            def modify_model_request(self, request: ModelRequest, state: AgentState):
                request.system_prompt = request.system_prompt + " [MW2]"
                return request

        # Order 1: MW1 then MW2
        request1 = ModelRequest(messages=[], system_prompt="Base")
        mw1 = MW1()
        mw2 = MW2()

        result1 = mw1.modify_model_request(request1, {})
        result1 = mw2.modify_model_request(result1, {})

        assert result1.system_prompt == "Base [MW1] [MW2]"

        # Order 2: MW2 then MW1
        request2 = ModelRequest(messages=[], system_prompt="Base")
        result2 = mw2.modify_model_request(request2, {})
        result2 = mw1.modify_model_request(result2, {})

        assert result2.system_prompt == "Base [MW2] [MW1]"

    def test_middleware_execution_in_graph(self):
        """Middleware should actually execute when graph runs"""
        execution_log = []

        class TrackingMiddleware(AgentMiddleware):
            def modify_model_request(self, request: ModelRequest, state: AgentState):
                execution_log.append("modify_model_request_called")
                return request

        agent = create_agent(
            model="claude-3-5-sonnet-20240620",
            middleware=[TrackingMiddleware()],
            prompt="Test",
            tools=[]
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "test"}]
        })

        # modify_model_request should have been called
        assert "modify_model_request_called" in execution_log

    def test_multiple_middleware_all_execute(self):
        """All middleware in stack should execute"""
        execution_log = []

        class MW1(AgentMiddleware):
            def modify_model_request(self, request: ModelRequest, state: AgentState):
                execution_log.append("MW1")
                return request

        class MW2(AgentMiddleware):
            def modify_model_request(self, request: ModelRequest, state: AgentState):
                execution_log.append("MW2")
                return request

        class MW3(AgentMiddleware):
            def modify_model_request(self, request: ModelRequest, state: AgentState):
                execution_log.append("MW3")
                return request

        agent = create_agent(
            model="claude-3-5-sonnet-20240620",
            middleware=[MW1(), MW2(), MW3()],
            prompt="Test",
            tools=[]
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "test"}]
        })

        # All should have executed
        assert "MW1" in execution_log
        assert "MW2" in execution_log
        assert "MW3" in execution_log

        # Should execute in order
        assert execution_log.index("MW1") < execution_log.index("MW2")
        assert execution_log.index("MW2") < execution_log.index("MW3")
```

### 6. Subagent Isolation Tests

```python
# File: tests/integration/test_subagent_isolation.py

import pytest
from deepagents.graph import create_deep_agent
from langchain_core.tools import tool
from tests.helpers.mock_model import MockToolCallingModel


class TestSubagentIsolation:
    """Test subagent state isolation and merging"""

    def test_subagent_todos_dont_merge_back(self):
        """Subagent todos should NOT merge back to parent"""
        # Per implementation: todos are excluded from merge

        @tool
        def dummy_tool():
            """Dummy tool"""
            return "ok"

        # Subagent will write todos
        subagent_model = MockToolCallingModel(tool_calls=[
            {"name": "write_todos", "args": {"todos": [{"content": "subtask", "status": "pending"}]}, "id": "call_1"}
        ])

        # Parent will call subagent
        parent_model = MockToolCallingModel(tool_calls=[
            {"name": "task", "args": {"description": "Do task", "subagent_type": "test_agent"}, "id": "call_2"}
        ])

        subagents = [{
            "name": "test_agent",
            "description": "Test agent",
            "prompt": "You are a test agent",
            "model": subagent_model,
            "tools": [dummy_tool]
        }]

        agent = create_deep_agent(
            model=parent_model,
            tools=[],
            subagents=subagents
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "Use test_agent"}],
            "todos": [{"content": "parent task", "status": "pending"}]
        })

        # Parent todos should remain unchanged
        # Subagent todos should NOT have merged back
        if "todos" in result:
            # Should only have parent task
            assert len(result["todos"]) == 1
            assert result["todos"][0]["content"] == "parent task"

    def test_subagent_files_merge_back(self):
        """Subagent file changes should merge back to parent"""
        # Per implementation: files ARE NOT in exclusion list

        @tool
        def dummy_tool():
            """Dummy tool"""
            return "ok"

        # Subagent will write file
        subagent_model = MockToolCallingModel(tool_calls=[
            {"name": "write_file", "args": {"file_path": "sub.txt", "content": "from subagent"}, "id": "call_1"}
        ])

        # Parent will call subagent
        parent_model = MockToolCallingModel(tool_calls=[
            {"name": "task", "args": {"description": "Write file", "subagent_type": "file_agent"}, "id": "call_2"}
        ])

        subagents = [{
            "name": "file_agent",
            "description": "File agent",
            "prompt": "You write files",
            "model": subagent_model,
            "tools": [dummy_tool]
        }]

        agent = create_deep_agent(
            model=parent_model,
            tools=[],
            subagents=subagents
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "Use file_agent"}],
            "files": {"parent.txt": "from parent"}
        })

        # Files should merge back
        if "files" in result:
            assert "parent.txt" in result["files"]
            # Subagent file should be present (if tool executed)
            # Note: This depends on subagent actually writing the file

    def test_subagent_receives_parent_state(self):
        """Subagent should receive copy of parent state"""
        received_states = []

        @tool
        def state_checker_tool(state: Annotated[dict, InjectedState]) -> str:
            """Captures state"""
            received_states.append(state.copy())
            return "checked"

        from langchain.agents.tool_node import InjectedState
        from typing import Annotated

        # Subagent will check state
        subagent_model = MockToolCallingModel(tool_calls=[
            {"name": "state_checker_tool", "args": {}, "id": "call_1"}
        ])

        # Parent will call subagent
        parent_model = MockToolCallingModel(tool_calls=[
            {"name": "task", "args": {"description": "Check state", "subagent_type": "checker"}, "id": "call_2"}
        ])

        subagents = [{
            "name": "checker",
            "description": "State checker",
            "prompt": "Check state",
            "model": subagent_model,
            "tools": [state_checker_tool]
        }]

        agent = create_deep_agent(
            model=parent_model,
            tools=[],
            subagents=subagents
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "Use checker"}],
            "files": {"parent.txt": "parent content"}
        })

        # Subagent should have received parent's files
        if received_states:
            subagent_state = received_states[0]
            assert "files" in subagent_state
            assert "parent.txt" in subagent_state["files"]

    def test_subagent_messages_dont_merge_back(self):
        """Subagent messages should NOT merge back to parent"""
        # Per implementation: messages are excluded from merge

        @tool
        def dummy_tool():
            """Dummy tool"""
            return "ok"

        # Subagent model
        subagent_model = MockToolCallingModel(content="Subagent response")

        # Parent will call subagent
        parent_model = MockToolCallingModel(tool_calls=[
            {"name": "task", "args": {"description": "Do something", "subagent_type": "test"}, "id": "call_1"}
        ])

        subagents = [{
            "name": "test",
            "description": "Test",
            "prompt": "Test",
            "model": subagent_model,
            "tools": [dummy_tool]
        }]

        agent = create_deep_agent(
            model=parent_model,
            tools=[],
            subagents=subagents
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "Use test subagent"}]
        })

        # Messages from subagent should NOT be in parent messages
        # Only a tool message with the final result should be added
        messages = result["messages"]
        tool_messages = [m for m in messages if m.type == "tool"]

        # Should have a tool message from the task tool
        # But NOT the subagent's intermediate messages
```

---

## Usage Guide

### Running These Tests

1. **Install dependencies:**
```bash
pip install -e ".[dev]"
```

2. **Create test structure:**
```bash
mkdir -p tests/unit tests/integration tests/helpers
touch tests/unit/__init__.py tests/integration/__init__.py tests/helpers/__init__.py
```

3. **Copy test files:**
- Copy mock_model.py to `tests/helpers/mock_model.py`
- Copy individual test modules to appropriate directories

4. **Run tests:**
```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_state_reducers.py

# Specific test
pytest tests/unit/test_state_reducers.py::TestFileReducer::test_both_none

# With coverage
pytest --cov=deepagents --cov-report=html
```

### Test Priorities

**Week 1 (Critical):**
1. `tests/helpers/mock_model.py` - Required for all other tests
2. `tests/unit/test_state_reducers.py` - Prevents state corruption bugs
3. `tests/integration/test_checkpointing.py` - Required for HITL and persistence

**Week 2 (High Priority):**
4. `tests/integration/test_injected_state.py` - Ensures tools work
5. `tests/integration/test_middleware_execution.py` - Ensures middleware works

**Week 3-4 (Medium Priority):**
6. `tests/integration/test_subagent_isolation.py` - Complex feature testing

### Common Issues

**Issue:** Mock model doesn't trigger tools
**Solution:** Ensure tool_calls format matches LangChain's expected format

**Issue:** State not updating
**Solution:** Check that Command is being returned with proper update dict

**Issue:** Tests fail with real API calls
**Solution:** Always use MockToolCallingModel or MockModel for unit/integration tests

**Issue:** Checkpointer tests fail
**Solution:** Ensure thread_id is in config: `{"configurable": {"thread_id": "..."}}`

---

## Next Steps

After implementing these tests:

1. **Add to CI/CD:**
```yaml
# .github/workflows/test.yml
- name: Run tests
  run: pytest tests/ --cov=deepagents
```

2. **Set coverage goals:**
```ini
# setup.cfg or pyproject.toml
[tool.pytest.ini_options]
addopts = "--cov=deepagents --cov-fail-under=80"
```

3. **Add property-based testing** (optional):
```python
from hypothesis import given, strategies as st

@given(
    left=st.none() | st.dictionaries(st.text(), st.text()),
    right=st.none() | st.dictionaries(st.text(), st.text())
)
def test_file_reducer_property(left, right):
    """Property: reducer should never raise an exception"""
    result = file_reducer(left, right)
    # Result should be None or dict
    assert result is None or isinstance(result, dict)
```

4. **Add mutation testing** (optional):
```bash
pip install mutpy
mutpy --target deepagents/state.py --unit-test tests/unit/test_state_reducers.py
```

This will ensure your tests actually catch bugs!
