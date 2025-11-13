# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CONTRIBUTING.md with comprehensive contribution guidelines and development setup
- CHANGELOG.md to track project changes
- Comprehensive docstrings to all classes and functions following Google-style format
- Module-level docstrings to all Python files in src/deepagents/
- Type hints to all 9 functions that were missing return types
- Type hint to `model` parameter in SubAgentMiddleware
- New API Reference section to README.md documenting all public APIs
- New Troubleshooting section to README.md with common issues and solutions
- New Contributing section to README.md with development setup guide

### Changed
- Refactored `prompts.py` into organized `prompts/` package with separate modules for system prompts and tool descriptions
- Improved README.md with better code examples and inline comments
- Updated README.md Installation section with development setup and optional dependencies
- Fixed parameter names in `file_reducer()` from `l, r` to `left, right` for better readability
- Marked "Code cleanliness" roadmap item as complete
- Updated project structure documentation to reflect prompts/ directory

### Removed
- Incomplete TODO comment in middleware.py (functionality verified as working)

### Fixed

### Security

## [0.0.9] - 2025-09-25

### Added
- Prompt caching support for improved performance
- Support for passing custom state keys to subagents

### Changed
- Updated to latest LangChain version (v1)
- Improved default deep agent performance
- Updated documentation to remove deprecated `create_react_agent` references

### Fixed
- Bug that prevented custom state keys from being passed to subagents correctly

## [0.0.8] - 2025-09-24

### Changed
- Updated LangChain dependency to latest version
- Migration to LangChain v1

### Fixed
- InjectedState-related issues

## [0.0.7] - 2025-09-18

### Added
- Ability to filter main agent tools
- Support for excluding specific built-in tools

### Changed
- Improved default system prompt to heavily discourage calling todo tool multiple times in parallel
- General purpose subagent now only receives base prompt (not full system prompt)

### Fixed
- General purpose task now correctly gets state schema to pass back file updates

## [0.0.6] - 2025-09-10

### Added
- Support for custom subagents with pre-built LangGraph graphs
- `CustomSubAgent` type for using existing agent graphs as subagents

### Fixed
- Human-in-the-loop interrupt issues
- Post-model hook passing to subagents

## [0.0.5] - 2025-09-01

### Added
- Async support with `async_create_deep_agent` function
- Support for async tools (useful for MCP tools)

### Changed
- Improved human-in-the-loop interrupt configuration
- Enhanced tool filtering for built-in tools

## [0.0.4] - 2025-08-22

### Added
- Human-in-the-loop interrupt configuration support via `tool_configs` parameter
- `HumanInTheLoopConfig` for specifying approval workflows

### Changed
- Improved interrupt handling and configuration options

## [0.0.3] - 2025-08-19

### Added
- Checkpointer support for deep agent creation
- Config schema for better agent configuration
- Support for async tools at subagent level

### Changed
- Subagents now use `ainvoke()` to properly support async tools

### Fixed
- Return type annotations for various functions
- Tool message updates in command handling

## [0.0.2] - 2025-07-30

### Added
- Initial public release of Deep Agents
- Core functionality including:
  - `create_deep_agent` function for agent creation
  - Built-in planning tool (TodoWrite)
  - Virtual file system with `ls`, `read_file`, `write_file`, `edit_file` tools
  - Sub-agent spawning capability
  - General-purpose subagent by default
  - Custom subagent configuration support
- Model configuration support (default: Claude Sonnet 4)
- Middleware support for extending functionality
- Comprehensive system prompt based on Claude Code
- Example implementations:
  - Research agent with Tavily search
  - Ollama model integration example
- MCP (Model Context Protocol) support via langchain-mcp-adapters
- Documentation and README

### Changed
- Project renamed from original name to "Deep Agents"

## [0.0.1] - 2025-07-27

### Added
- Initial prototype and project structure
- Basic agent framework using LangGraph
- MIT License

---

## Version History Notes

### Understanding Version Numbers

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.1.0): New functionality in a backward-compatible manner
- **PATCH** version (0.0.1): Backward-compatible bug fixes

### Pre-1.0.0 Releases

As this project is currently in active development (version 0.0.x), we may introduce breaking changes in minor version updates. Once we reach version 1.0.0, we will strictly adhere to semantic versioning.

### How to Contribute to the Changelog

When submitting a pull request, please add an entry to the `[Unreleased]` section under the appropriate category:

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed in upcoming releases
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes
- **Security**: Security fixes or improvements

Example:
```markdown
## [Unreleased]

### Added
- New feature X for doing Y (#PR_NUMBER)

### Fixed
- Bug in feature Z that caused W (#PR_NUMBER)
```

Maintainers will move entries from `[Unreleased]` to a new version section during releases.

---

[Unreleased]: https://github.com/langchain-ai/deepagents/compare/v0.0.9...HEAD
[0.0.9]: https://github.com/langchain-ai/deepagents/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/langchain-ai/deepagents/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/langchain-ai/deepagents/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/langchain-ai/deepagents/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/langchain-ai/deepagents/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/langchain-ai/deepagents/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/langchain-ai/deepagents/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/langchain-ai/deepagents/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/langchain-ai/deepagents/releases/tag/v0.0.1
