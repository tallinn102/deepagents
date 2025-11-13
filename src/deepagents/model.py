"""Default model configuration for DeepAgents.

This module provides the default language model instance used across the DeepAgents framework.
"""

from langchain_anthropic import ChatAnthropic


def get_default_model() -> ChatAnthropic:
    """Get the default ChatAnthropic model instance.

    Returns:
        ChatAnthropic: A configured ChatAnthropic instance with Claude Sonnet 4 model
            and 64000 max tokens.
    """
    return ChatAnthropic(model_name="claude-sonnet-4-20250514", max_tokens=64000)
