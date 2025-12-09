"""
Custom LLM wrapper that sanitizes messages for Docker Model Runner compatibility.

Docker Model Runner rejects message lists with 2 or more consecutive assistant messages.
This wrapper merges consecutive assistant messages before sending to the model.
"""
import os
from typing import Any, Dict, List, Optional

import litellm
from crewai import LLM


def merge_consecutive_assistant_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive assistant messages to avoid Docker Model Runner rejection."""
    if not messages:
        return messages

    merged = []
    i = 0
    while i < len(messages):
        msg = messages[i].copy()

        # If this is an assistant message, check for consecutive ones
        if msg.get("role") == "assistant":
            content_parts = []

            # Get content from this message
            if isinstance(msg.get("content"), str):
                content_parts.append(msg["content"])
            elif isinstance(msg.get("content"), list):
                # Handle multi-part content
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        content_parts.append(part)

            # Check for consecutive assistant messages
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "assistant":
                next_msg = messages[j]
                if isinstance(next_msg.get("content"), str):
                    content_parts.append(next_msg["content"])
                elif isinstance(next_msg.get("content"), list):
                    for part in next_msg["content"]:
                        if isinstance(part, dict) and part.get("type") == "text":
                            content_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            content_parts.append(part)
                j += 1

            # Merge all content parts
            msg["content"] = "\n\n".join(filter(None, content_parts))
            merged.append(msg)
            i = j
        else:
            merged.append(msg)
            i += 1

    return merged


def get_custom_llm() -> LLM:
    """Create a custom LLM that works with Docker Model Runner."""
    base_url = os.getenv("OPENAI_BASE_URL", "")
    model_name = os.getenv("OPENAI_MODEL_NAME", "")

    # Use hosted_vllm prefix for proper OpenAI-compatible handling
    return LLM(
        model=model_name,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
    )


# Monkey-patch litellm to sanitize messages
_original_completion = litellm.completion


def patched_completion(*args, **kwargs):
    """Wrapper that sanitizes messages before sending to LiteLLM."""
    if "messages" in kwargs:
        kwargs["messages"] = merge_consecutive_assistant_messages(kwargs["messages"])
    return _original_completion(*args, **kwargs)


def install_message_sanitizer():
    """Install the message sanitizer patch."""
    litellm.completion = patched_completion
