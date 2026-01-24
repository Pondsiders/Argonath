"""Attribute extraction from Anthropic API requests/responses.

Emits gen_ai.* semantic conventions for Logfire LLM observability.
"""

import json
from typing import Any


def _anthropic_content_to_gen_ai_parts(content: Any) -> list[dict]:
    """Convert Anthropic content to gen_ai parts format.

    gen_ai format: [{"type": "text", "content": "..."}, {"type": "tool_call", ...}]
    """
    if isinstance(content, str):
        return [{"type": "text", "content": content}]

    if not isinstance(content, list):
        return []

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append({"type": "text", "content": block})
        elif isinstance(block, dict):
            block_type = block.get("type")

            if block_type == "text":
                parts.append({"type": "text", "content": block.get("text", "")})

            elif block_type == "tool_use":
                parts.append({
                    "type": "tool_call",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {})
                })

            elif block_type == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    # Extract text from nested content
                    texts = [item.get("text", "") for item in result_content
                             if isinstance(item, dict) and item.get("type") == "text"]
                    result_content = "\n".join(texts)
                parts.append({
                    "type": "tool_call_response",
                    "id": block.get("tool_use_id", ""),
                    "result": result_content[:2000] if isinstance(result_content, str) else str(result_content)[:2000]
                })

            elif block_type == "image":
                # Note image presence without embedding data
                parts.append({"type": "image", "content": "[image]"})

    return parts


def _anthropic_system_to_gen_ai(system: Any) -> list[dict]:
    """Convert Anthropic system prompt to gen_ai system_instructions format."""
    if isinstance(system, str):
        return [{"type": "text", "content": system}]

    if isinstance(system, list):
        instructions = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                instructions.append({"type": "text", "content": block.get("text", "")})
            elif isinstance(block, str):
                instructions.append({"type": "text", "content": block})
        return instructions

    return []


def _anthropic_tools_to_gen_ai(tools: list) -> list[dict]:
    """Convert Anthropic tools to gen_ai tool_definitions format."""
    definitions = []
    for tool in tools:
        definitions.append({
            "type": "function",
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {})
        })
    return definitions


def _is_tool_result_message(content: Any) -> bool:
    """Check if message content is a tool result."""
    if isinstance(content, list):
        return any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        )
    return False


def request_attributes(body: dict[str, Any]) -> dict[str, Any]:
    """Extract gen_ai.* attributes from an Anthropic Messages API request."""
    attrs: dict[str, Any] = {}

    model = body.get("model", "unknown")

    # === gen_ai.* semantic conventions ===
    attrs["gen_ai.operation.name"] = "chat"
    attrs["gen_ai.provider.name"] = "anthropic"
    attrs["gen_ai.request.model"] = model

    # Request parameters
    if "max_tokens" in body:
        attrs["gen_ai.request.max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        attrs["gen_ai.request.temperature"] = body["temperature"]
    if "top_p" in body:
        attrs["gen_ai.request.top_p"] = body["top_p"]
    if "top_k" in body:
        attrs["gen_ai.request.top_k"] = body["top_k"]
    if "stop_sequences" in body:
        attrs["gen_ai.request.stop_sequences"] = body["stop_sequences"]

    # System instructions - full computed system prompt
    system = body.get("system")
    if system:
        attrs["gen_ai.system_instructions"] = json.dumps(_anthropic_system_to_gen_ai(system))

    # Input messages - CURRENT TURN ONLY (last user message)
    # Each trace captures one API call. If you want full history, look at previous traces.
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if not _is_tool_result_message(content):
                attrs["gen_ai.input.messages"] = json.dumps([{
                    "role": "user",
                    "parts": _anthropic_content_to_gen_ai_parts(content)
                }])
                break

    # Tool definitions - full schemas
    tools = body.get("tools", [])
    if tools:
        attrs["gen_ai.tool.definitions"] = json.dumps(_anthropic_tools_to_gen_ai(tools))

    return attrs


def response_attributes(body: dict[str, Any]) -> dict[str, Any]:
    """Extract gen_ai.* attributes from an Anthropic Messages API response."""
    attrs: dict[str, Any] = {}

    # Token counts
    usage = body.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # gen_ai.* semantic conventions
    attrs["gen_ai.usage.input_tokens"] = input_tokens
    attrs["gen_ai.usage.output_tokens"] = output_tokens

    # llm.token_count.* (for Phoenix/OpenInference and Logfire list view display)
    attrs["llm.token_count.prompt"] = input_tokens
    attrs["llm.token_count.completion"] = output_tokens
    attrs["llm.token_count.total"] = input_tokens + output_tokens

    # Response identification
    if "id" in body:
        attrs["gen_ai.response.id"] = body["id"]
    if "model" in body:
        attrs["gen_ai.response.model"] = body["model"]

    # Cache metrics (Anthropic-specific, but useful)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    if cache_read:
        attrs["gen_ai.anthropic.cache_read_tokens"] = cache_read
    if cache_write:
        attrs["gen_ai.anthropic.cache_write_tokens"] = cache_write

    # Stop reason
    stop_reason = body.get("stop_reason")
    if stop_reason:
        attrs["gen_ai.response.finish_reasons"] = [stop_reason]

    # Output messages
    content_blocks = body.get("content", [])
    output_parts = _anthropic_content_to_gen_ai_parts(content_blocks)
    if output_parts:
        attrs["gen_ai.output.messages"] = json.dumps([{
            "role": "assistant",
            "parts": output_parts,
            "finish_reason": stop_reason or "stop"
        }])

    return attrs
