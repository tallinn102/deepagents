# Contributing to Deep Agents

Thank you for your interest in contributing to Deep Agents! This project aims to make it easy to create powerful, deep agents that can plan, use tools, spawn sub-agents, and work with file systems. We welcome contributions from the community and appreciate your help in making this project better.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)
- [Questions and Support](#questions-and-support)

## Getting Started

Before you start contributing, please:

1. Read the [README.md](README.md) to understand what Deep Agents does and how it works
2. Check the [existing issues](https://github.com/langchain-ai/deepagents/issues) to see if your idea or bug has already been reported
3. For major changes, open an issue first to discuss your proposed changes

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git

### Setting Up Your Development Environment

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/YOUR-USERNAME/deepagents.git
   cd deepagents
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package in editable mode with dev dependencies:**

   ```bash
   pip install -e ".[dev]"
   ```

   This installs:
   - The main package dependencies (langgraph, langchain, langchain-anthropic, langgraph-prebuilt)
   - Development dependencies (pytest, pytest-cov, build, twine)

4. **Verify your installation:**

   ```bash
   python -c "from deepagents import create_deep_agent; print('Installation successful!')"
   ```

### Project Structure

```
deepagents/
├── src/deepagents/        # Main source code
│   ├── __init__.py        # Public API exports
│   ├── graph.py           # Agent graph creation
│   ├── middleware.py      # Middleware components
│   ├── model.py           # Model configuration
│   ├── prompts/           # System prompts (organized by type)
│   ├── state.py           # State management
│   ├── tools.py           # Built-in tools
│   └── types.py           # Type definitions
├── tests/                 # Test files
├── examples/              # Example implementations
└── pyproject.toml         # Project configuration
```

## Code Style Guidelines

### General Principles

- **Clarity over cleverness**: Write code that is easy to understand and maintain
- **Follow Python conventions**: Adhere to PEP 8 style guidelines
- **Be consistent**: Match the existing code style in the project

### Type Hints

We strongly encourage the use of type hints for all function signatures:

```python
from typing import List, Optional, Dict, Any

def create_subagent(
    name: str,
    description: str,
    tools: List[Any],
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Create a subagent configuration."""
    ...
```

### Docstrings

All public functions, classes, and modules should have clear docstrings:

```python
def create_deep_agent(tools: list, instructions: str, **kwargs):
    """Create a deep agent with planning, file system, and sub-agent capabilities.

    Args:
        tools: List of functions or LangChain tool objects the agent can use
        instructions: Custom instructions to prepend to the system prompt
        **kwargs: Additional configuration options (subagents, model, middleware, etc.)

    Returns:
        A compiled LangGraph agent ready to be invoked

    Example:
        >>> agent = create_deep_agent([my_tool], "You are a helpful assistant")
        >>> result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
    """
    ...
```

### Code Formatting

While we don't currently enforce a specific formatter, please:
- Use 4 spaces for indentation (no tabs)
- Keep line length reasonable (aim for 88-100 characters, but be flexible)
- Use meaningful variable and function names
- Add comments for complex logic

### Imports

Organize imports in the following order:
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
import os
from typing import List, Optional

from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

from deepagents.state import DeepAgentState
from deepagents.tools import create_file_tools
```

## Testing Guidelines

### Running Tests

Run the test suite using pytest:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=deepagents --cov-report=html

# Run specific test file
pytest tests/test_deepagents.py

# Run specific test function
pytest tests/test_deepagents.py::test_function_name
```

### Writing Tests

When adding new features or fixing bugs, please include tests:

1. **Unit tests** for individual functions and components
2. **Integration tests** for end-to-end workflows
3. **Edge case tests** for error handling and boundary conditions

Example test structure:

```python
import pytest
from deepagents import create_deep_agent

def test_basic_agent_creation():
    """Test that a basic agent can be created successfully."""
    def dummy_tool(query: str) -> str:
        """A dummy tool for testing."""
        return f"Result: {query}"

    agent = create_deep_agent(
        tools=[dummy_tool],
        instructions="You are a test agent"
    )

    assert agent is not None
    # Add more assertions

def test_agent_with_invalid_config():
    """Test that invalid configurations raise appropriate errors."""
    with pytest.raises(ValueError):
        create_deep_agent(tools=[], instructions="")
```

### Test Coverage

- Aim for high test coverage on new code (80%+ is a good target)
- Don't sacrifice test quality for coverage numbers
- Focus on testing behavior, not implementation details

## Pull Request Process

### Before Submitting

1. **Update tests**: Ensure all tests pass and add tests for new functionality
2. **Update documentation**: Update README.md or other docs if needed
3. **Check your changes**: Review your own code first
4. **Run the test suite**: Make sure all tests pass locally

### Submitting Your PR

1. **Create a descriptive title**: Use a clear, concise title that describes the change
   - Good: "Add support for custom file system backends"
   - Bad: "Fix bug" or "Updates"

2. **Write a detailed description**: Include:
   - What changes you made and why
   - Any relevant issue numbers (e.g., "Fixes #123")
   - Testing you performed
   - Any breaking changes or migration notes

3. **Keep PRs focused**: One PR should address one concern
   - Feature additions should be separate from bug fixes
   - Refactoring should be separate from new features

4. **Be responsive**: Respond to review comments promptly and professionally

### PR Template

```markdown
## Description
Brief description of what this PR does

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2

## Testing
How was this tested?

## Breaking Changes
Any breaking changes? (Yes/No)
If yes, describe migration path

## Related Issues
Fixes #(issue number)
```

### Review Process

1. A maintainer will review your PR
2. They may request changes or ask questions
3. Once approved, a maintainer will merge your PR
4. Your contribution will be included in the next release!

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. While we don't have a formal CODE_OF_CONDUCT.md yet, we expect all contributors to:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

Unacceptable behavior includes harassment, trolling, insults, or other unprofessional conduct.

## Questions and Support

### Getting Help

- **Documentation**: Check the [README.md](README.md) first
- **Issues**: Search [existing issues](https://github.com/langchain-ai/deepagents/issues)
- **New Issue**: Open a new issue if you can't find an answer
- **Discussions**: Use GitHub Discussions for general questions

### Reporting Bugs

When reporting bugs, please include:
- Python version
- deepagents version
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Full error traceback (if applicable)

### Suggesting Features

For feature requests:
- Explain the use case and why it's valuable
- Describe how it might work
- Consider if it fits the project's scope and goals

## Recognition

Contributors will be recognized in:
- The CHANGELOG.md for their contributions
- GitHub's contributor graph
- Special mentions for significant contributions

## License

By contributing to Deep Agents, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Deep Agents! Your efforts help make this project better for everyone.
