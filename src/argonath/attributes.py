"""OpenInference attribute extraction from Anthropic API requests/responses.

Preserves the full structure of content arrays for proper Phoenix display.
"""

import json
from typing import Any


def request_attributes(body: dict[str, Any]) -> dict[str, Any]:
    """Extract OpenInference attributes from an Anthropic Messages API request.

    Preserves content array structure using indexed attribute names.
    System prompt becomes message index 0, conversation messages follow.
    """
    attrs: dict[str, Any] = {}

    # === Core LLM attributes ===
    model = body.get("model", "unknown")
    attrs["openinference.span.kind"] = "LLM"
    attrs["llm.system"] = "anthropic"
    attrs["llm.model_name"] = model

    # gen_ai.* for compatibility
    attrs["gen_ai.system"] = "anthropic"
    attrs["gen_ai.request.model"] = model

    # === Invocation parameters ===
    attrs["llm.invocation_parameters"] = json.dumps({
        "max_tokens": body.get("max_tokens", 0),
        "temperature": body.get("temperature", 1.0),
        "stream": body.get("stream", False),
    })

    # === Input messages ===
    msg_idx = 0

    # System prompt first (index 0)
    system = body.get("system")
    if system:
        attrs[f"llm.input_messages.{msg_idx}.message.role"] = "system"
        if isinstance(system, str):
            attrs[f"llm.input_messages.{msg_idx}.message.content"] = system
        elif isinstance(system, list):
            # System prompt with content array (cache_control blocks, etc.)
            _add_content_array(attrs, f"llm.input_messages.{msg_idx}.message", system)
        msg_idx += 1

    # Conversation messages
    for msg in body.get("messages", []):
        role = msg.get("role", "unknown")
        content = msg.get("content")
        prefix = f"llm.input_messages.{msg_idx}.message"

        attrs[f"{prefix}.role"] = role

        if isinstance(content, str):
            attrs[f"{prefix}.content"] = content
        elif isinstance(content, list):
            _add_content_array(attrs, prefix, content)

        msg_idx += 1

    # === Tools (if provided) ===
    tools = body.get("tools", [])
    for tool_idx, tool in enumerate(tools):
        prefix = f"llm.tools.{tool_idx}.tool"
        attrs[f"{prefix}.json_schema"] = json.dumps(tool)
        if "name" in tool:
            attrs[f"{prefix}.name"] = tool["name"]
        if "description" in tool:
            attrs[f"{prefix}.description"] = tool.get("description", "")

    # === input.value for Phoenix UI (last real user message) ===
    last_user_text = _find_last_user_text(body.get("messages", []))
    if last_user_text:
        attrs["input.value"] = last_user_text[:2000]
        attrs["input.mime_type"] = "text/plain"

    return attrs


def response_attributes(body: dict[str, Any]) -> dict[str, Any]:
    """Extract OpenInference attributes from an Anthropic Messages API response."""
    attrs: dict[str, Any] = {}

    # === Token counts ===
    usage = body.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    attrs["llm.token_count.prompt"] = input_tokens
    attrs["llm.token_count.completion"] = output_tokens
    attrs["llm.token_count.total"] = input_tokens + output_tokens

    # gen_ai.* for compatibility
    attrs["gen_ai.usage.input_tokens"] = input_tokens
    attrs["gen_ai.usage.output_tokens"] = output_tokens

    # Cache metrics (Anthropic-specific)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    if cache_read:
        attrs["llm.token_count.prompt_details.cache_read"] = cache_read
    if cache_write:
        attrs["llm.token_count.prompt_details.cache_write"] = cache_write

    # === Output message ===
    content_blocks = body.get("content", [])
    prefix = "llm.output_messages.0.message"

    attrs[f"{prefix}.role"] = "assistant"

    # Separate text content from tool calls
    text_parts = []
    tool_call_idx = 0

    for block in content_blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type == "text":
            text_parts.append(block.get("text", ""))

        elif block_type == "tool_use":
            tc_prefix = f"{prefix}.tool_calls.{tool_call_idx}.tool_call"
            attrs[f"{tc_prefix}.id"] = block.get("id", "")
            attrs[f"{tc_prefix}.function.name"] = block.get("name", "")
            attrs[f"{tc_prefix}.function.arguments"] = json.dumps(block.get("input", {}))
            tool_call_idx += 1

    # Set content as joined text (tool calls are in tool_calls)
    if text_parts:
        full_text = "\n".join(text_parts)
        attrs[f"{prefix}.content"] = full_text

        # output.value for Phoenix UI
        attrs["output.value"] = full_text[:2000]
        attrs["output.mime_type"] = "text/plain"

    # === Stop reason ===
    stop_reason = body.get("stop_reason")
    if stop_reason:
        attrs["llm.stop_reason"] = stop_reason

    return attrs


def _add_content_array(attrs: dict[str, Any], prefix: str, content: list) -> None:
    """Add content array items as indexed OpenInference attributes.

    Handles: text, image, tool_use, tool_result blocks.
    """
    # Track tool calls separately (they go in message.tool_calls, not contents)
    tool_call_idx = 0
    content_idx = 0

    for block in content:
        if not isinstance(block, dict):
            # Plain string in array
            c_prefix = f"{prefix}.contents.{content_idx}.message_content"
            attrs[f"{c_prefix}.type"] = "text"
            attrs[f"{c_prefix}.text"] = str(block)
            content_idx += 1
            continue

        block_type = block.get("type")

        if block_type == "text":
            c_prefix = f"{prefix}.contents.{content_idx}.message_content"
            attrs[f"{c_prefix}.type"] = "text"
            attrs[f"{c_prefix}.text"] = block.get("text", "")
            content_idx += 1

        elif block_type == "image":
            c_prefix = f"{prefix}.contents.{content_idx}.message_content"
            attrs[f"{c_prefix}.type"] = "image"
            # Handle both URL and base64 source types
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                attrs[f"{c_prefix}.image.image.url"] = f"data:{media_type};base64,{data[:100]}..."
            elif source.get("type") == "url":
                attrs[f"{c_prefix}.image.image.url"] = source.get("url", "")
            content_idx += 1

        elif block_type == "tool_use":
            # Tool uses in assistant messages go to tool_calls
            tc_prefix = f"{prefix}.tool_calls.{tool_call_idx}.tool_call"
            attrs[f"{tc_prefix}.id"] = block.get("id", "")
            attrs[f"{tc_prefix}.function.name"] = block.get("name", "")
            attrs[f"{tc_prefix}.function.arguments"] = json.dumps(block.get("input", {}))
            tool_call_idx += 1

        elif block_type == "tool_result":
            # Tool results in user messages
            c_prefix = f"{prefix}.contents.{content_idx}.message_content"
            attrs[f"{c_prefix}.type"] = "tool_result"
            attrs[f"{prefix}.tool_call_id"] = block.get("tool_use_id", "")

            # Extract text from tool result content
            result_content = block.get("content", "")
            if isinstance(result_content, str):
                attrs[f"{c_prefix}.text"] = result_content[:1000]
            elif isinstance(result_content, list):
                # Tool result can have its own content array
                texts = []
                for item in result_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                attrs[f"{c_prefix}.text"] = "\n".join(texts)[:1000]
            content_idx += 1


def _find_last_user_text(messages: list) -> str:
    """Find the last real user text (not tool results or system injections)."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            # Skip hook-injected content
            if content.startswith("<system-reminder>"):
                continue
            if "LOOM_METADATA" in content or "EAVESDROP_METADATA" in content:
                continue
            return content

        if isinstance(content, list):
            # Find text blocks that aren't system injections
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text.startswith("<system-reminder>"):
                        continue
                    if "LOOM_METADATA" in text or "EAVESDROP_METADATA" in text:
                        continue
                    return text

    return ""
