"""gen_ai.* attribute extraction from Anthropic API requests/responses.

Follows OpenTelemetry Semantic Conventions for GenAI:
https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md

Key design decisions:
- System prompt goes in gen_ai.system_instructions (separate from input.messages)
- gen_ai.input.messages contains the current TURN only (from last user text message onward)
- Turn boundary = user message with text content (not tool_result)
"""

import json
from typing import Any


def request_attributes(body: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
    """Extract gen_ai.* attributes from an Anthropic Messages API request."""
    attrs: dict[str, Any] = {}

    model = body.get("model", "unknown")

    # === Required attributes ===
    attrs["gen_ai.operation.name"] = "chat"
    attrs["gen_ai.provider.name"] = "anthropic"
    attrs["gen_ai.request.model"] = model

    # === Conditionally required ===
    if session_id:
        attrs["gen_ai.conversation.id"] = session_id

    # === Recommended request parameters ===
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

    # === System instructions (separate from input.messages) ===
    system = body.get("system")
    if system:
        attrs["gen_ai.system_instructions"] = _format_system_instructions(system)

    # === Input messages (current turn only) ===
    messages = body.get("messages", [])
    turn_messages = _extract_current_turn(messages)
    if turn_messages:
        attrs["gen_ai.input.messages"] = _format_input_messages(turn_messages)

    return attrs


def response_attributes(body: dict[str, Any]) -> dict[str, Any]:
    """Extract gen_ai.* attributes from an Anthropic Messages API response."""
    attrs: dict[str, Any] = {}

    # === Usage ===
    usage = body.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    if input_tokens:
        attrs["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens:
        attrs["gen_ai.usage.output_tokens"] = output_tokens

    # === Response metadata ===
    if "id" in body:
        attrs["gen_ai.response.id"] = body["id"]
    if "model" in body:
        attrs["gen_ai.response.model"] = body["model"]

    # === Finish reason ===
    stop_reason = body.get("stop_reason")
    if stop_reason:
        attrs["gen_ai.response.finish_reasons"] = [stop_reason]

    # === Output type ===
    content_blocks = body.get("content", [])
    has_text = any(b.get("type") == "text" for b in content_blocks if isinstance(b, dict))
    has_tool = any(b.get("type") == "tool_use" for b in content_blocks if isinstance(b, dict))

    if has_tool:
        attrs["gen_ai.output.type"] = "json"  # Tool calls are structured
    elif has_text:
        attrs["gen_ai.output.type"] = "text"

    # === Output messages ===
    if content_blocks:
        attrs["gen_ai.output.messages"] = _format_output_messages(content_blocks, stop_reason)

    return attrs


def _extract_current_turn(messages: list) -> list:
    """Extract the current turn from the message history.

    Returns:
    1. ALWAYS message[0] (CLAUDE.md from SDK) — this is crucial context
    2. Everything after the last assistant message (all user messages with
       system-reminders, prompts, memories, Intro memorables)

    This captures:
    - The SDK boilerplate that shapes behavior
    - All system-reminders injected by hooks
    - The user's actual prompt
    - Injected memories
    - Intro's memorables
    """
    if not messages:
        return []

    result = []

    # Always include message[0] — that's CLAUDE.md from the SDK
    if messages:
        result.append(messages[0])

    # Find the last assistant message
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    # Include everything after the last assistant message
    if last_assistant_idx is not None:
        result.extend(messages[last_assistant_idx + 1:])
    elif len(messages) > 1:
        # No assistant message found — include everything except [0] (already added)
        result.extend(messages[1:])

    return result


def _format_system_instructions(system: str | list) -> str:
    """Format system prompt as gen_ai.system_instructions.

    Returns JSON string per spec: array of {"type": "text", "content": "..."}
    """
    if isinstance(system, str):
        return json.dumps([{"type": "text", "content": system}])

    # Already a list (cache_control blocks, etc.)
    parts = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append({"type": "text", "content": block.get("text", "")})
        elif isinstance(block, str):
            parts.append({"type": "text", "content": block})

    return json.dumps(parts)


def _format_input_messages(messages: list) -> str:
    """Format messages as gen_ai.input.messages.

    Returns JSON string per spec: array of {"role": "...", "parts": [...]}
    """
    formatted = []

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content")

        parts = []

        if isinstance(content, str):
            parts.append({"type": "text", "content": content})

        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")

                if block_type == "text":
                    text = block.get("text", "")
                    parts.append({"type": "text", "content": text})

                elif block_type == "tool_use":
                    parts.append({
                        "type": "tool_call",
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    })

                elif block_type == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        # Extract text from result content array
                        texts = [
                            item.get("text", "")
                            for item in result_content
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        result_content = "\n".join(texts)

                    parts.append({
                        "type": "tool_call_response",
                        "id": block.get("tool_use_id", ""),
                        "result": result_content[:1000] if result_content else "",
                    })

        if parts:  # Only add if there are actual parts
            formatted.append({"role": role, "parts": parts})

    return json.dumps(formatted)


def _format_output_messages(content_blocks: list, stop_reason: str | None = None) -> str:
    """Format response as gen_ai.output.messages.

    Returns JSON string per spec: array of {"role": "assistant", "parts": [...], "finish_reason": "..."}
    """
    parts = []

    for block in content_blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type == "text":
            parts.append({"type": "text", "content": block.get("text", "")})

        elif block_type == "tool_use":
            parts.append({
                "type": "tool_call",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": block.get("input", {}),
            })

    message = {"role": "assistant", "parts": parts}
    if stop_reason:
        message["finish_reason"] = stop_reason

    return json.dumps([message])
