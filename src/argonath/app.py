"""The Argonath - FastAPI application.

Pure observation proxy. Witnesses all traffic, emits perfect LLM traces.
No transformation, just watching.
"""

import json
import logging
import os
import time

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx

from pondside.telemetry import init, get_tracer
from opentelemetry import trace as otel_trace
from opentelemetry.context import attach, detach
from opentelemetry.trace import Status, StatusCode, set_span_in_context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .attributes import request_attributes, response_attributes

# Initialize telemetry
init("argonath")

logger = logging.getLogger(__name__)
tracer = get_tracer()

ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com")

# Reusable HTTP client
_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    """Get or create the HTTP client."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=ANTHROPIC_API_URL,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )
    return _client


async def close_client():
    """Close the HTTP client."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


app = FastAPI(
    title="The Argonath",
    description="Everything that passes through the Gates is witnessed.",
)


@app.on_event("shutdown")
async def shutdown():
    await close_client()


def _filter_request_headers(headers: dict) -> dict:
    """Filter headers for forwarding to Anthropic."""
    # Headers to forward
    forward = {}
    for key, value in headers.items():
        key_lower = key.lower()
        # Skip hop-by-hop and host headers
        if key_lower in ("host", "connection", "keep-alive", "transfer-encoding",
                         "te", "trailers", "upgrade", "proxy-authorization",
                         "proxy-connection", "content-length"):
            continue
        # Skip our custom correlation headers (don't send to Anthropic)
        if key_lower in ("x-session-id", "traceparent", "tracestate"):
            continue
        forward[key] = value
    return forward


def _filter_response_headers(headers: dict) -> dict:
    """Filter headers for returning to client."""
    skip = {"transfer-encoding", "content-encoding", "content-length", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in skip}


def _parse_sse_response(chunks: list[bytes]) -> dict | None:
    """Parse SSE chunks to extract final response data.

    Returns a dict with usage, content, and tool_use info.
    """
    full_text = b"".join(chunks).decode("utf-8", errors="replace")

    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0
    text_parts: list[str] = []
    tool_uses: list[dict] = []
    current_tool: dict | None = None
    stop_reason = None

    for line in full_text.split("\n"):
        if not line.startswith("data: "):
            continue
        try:
            data = json.loads(line[6:])
            event_type = data.get("type")

            if event_type == "message_start":
                message = data.get("message", {})
                usage = message.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)
                cache_write = usage.get("cache_creation_input_tokens", 0)

            elif event_type == "content_block_start":
                content_block = data.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    current_tool = {
                        "type": "tool_use",
                        "id": content_block.get("id", ""),
                        "name": content_block.get("name", ""),
                        "input_json": "",
                    }

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text_parts.append(delta.get("text", ""))
                elif delta.get("type") == "input_json_delta" and current_tool:
                    current_tool["input_json"] += delta.get("partial_json", "")

            elif event_type == "content_block_stop":
                if current_tool:
                    try:
                        current_tool["input"] = json.loads(current_tool["input_json"]) if current_tool["input_json"] else {}
                    except json.JSONDecodeError:
                        current_tool["input"] = {}
                    del current_tool["input_json"]
                    tool_uses.append(current_tool)
                    current_tool = None

            elif event_type == "message_delta":
                usage = data.get("usage", {})
                output_tokens = usage.get("output_tokens", 0)
                stop_reason = data.get("delta", {}).get("stop_reason")

        except json.JSONDecodeError:
            continue

    if input_tokens or output_tokens or text_parts or tool_uses:
        content = []
        if text_parts:
            content.append({"type": "text", "text": "".join(text_parts)})
        content.extend(tool_uses)

        return {
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
            },
            "content": content,
            "stop_reason": stop_reason,
        }

    return None


async def _stream_and_capture(upstream_response, chunks_list: list):
    """Stream response while capturing chunks."""
    async for chunk in upstream_response.aiter_bytes():
        chunks_list.append(chunk)
        yield chunk


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def handle_request(request: Request, path: str):
    """Proxy all requests to Anthropic, witnessing and recording."""

    start_time_ns = time.time_ns()
    body_bytes = await request.body()
    headers = dict(request.headers)

    # === Extract correlation info from headers ===
    session_id = headers.get("x-session-id")
    traceparent = headers.get("traceparent")
    loom_pattern = headers.get("x-loom-pattern")

    # Get parent context if traceparent provided
    parent_context = None
    if traceparent:
        carrier = {"traceparent": traceparent}
        parent_context = TraceContextTextMapPropagator().extract(carrier=carrier)

    # === Determine if this is a messages endpoint ===
    is_messages = request.method == "POST" and "messages" in path

    # Only create LLM spans for actual messages calls, not for batch/logging endpoints
    if not is_messages:
        # Just proxy through without tracing
        client = await get_client()
        forward_headers = _filter_request_headers(headers)
        upstream_response = await client.request(
            method=request.method,
            url=f"/{path}",
            headers=forward_headers,
            content=body_bytes,
            params=dict(request.query_params),
        )
        response_headers = _filter_response_headers(dict(upstream_response.headers))
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    # === Messages endpoint - parse and trace ===
    request_body = None
    model_name = "unknown"

    try:
        request_body = json.loads(body_bytes)
        model_name = request_body.get("model", "unknown")
    except json.JSONDecodeError:
        pass

    # === Create wrapper span for infrastructure visibility ===
    # This span is parented to the incoming traceparent and goes to Logfire
    # (no openinference.span.kind attribute). All log messages inside it
    # will be properly parented. The LLM span is created separately inside.
    wrapper_span = tracer.start_span(
        "argonath: /v1/messages",
        context=parent_context,
    )
    wrapper_span.set_attribute("model", model_name)
    if session_id:
        wrapper_span.set_attribute("session.id", session_id[:8])

    # Make wrapper span the current context so logs are parented to it
    wrapper_context = set_span_in_context(wrapper_span)
    wrapper_token = attach(wrapper_context)

    logger.info(f"Processing: model={model_name}, session={session_id[:8] if session_id else 'none'}")

    # === Create the LLM span (goes to Phoenix via openinference.span.kind) ===
    # Pattern + model makes it easy to tell who's talking (alpha, iota, etc.)
    span_prefix = f"{loom_pattern} " if loom_pattern else ""
    llm_span = tracer.start_span(
        name=f"{span_prefix}{model_name}",
        kind=otel_trace.SpanKind.CLIENT,
        start_time=start_time_ns,
        # Parent to current context (wrapper_span)
    )

    try:
        # Set request attributes on LLM span
        if request_body:
            req_attrs = request_attributes(request_body)
            for key, value in req_attrs.items():
                llm_span.set_attribute(key, value)

        # Session correlation
        if session_id:
            llm_span.set_attribute("session.id", session_id)

        # Pattern identification (who's talking: alpha, iota, etc.)
        if loom_pattern:
            llm_span.set_attribute("loom.pattern", loom_pattern)

        # === Forward to Anthropic ===
        client = await get_client()
        forward_headers = _filter_request_headers(headers)

        upstream_response = await client.request(
            method=request.method,
            url=f"/{path}",
            headers=forward_headers,
            content=body_bytes,
            params=dict(request.query_params),
        )

        status_code = upstream_response.status_code
        content_type = upstream_response.headers.get("content-type", "")
        response_headers = _filter_response_headers(dict(upstream_response.headers))

        llm_span.set_attribute("http.status_code", status_code)
        wrapper_span.set_attribute("http.status_code", status_code)

        if "text/event-stream" in content_type:
            # === Streaming response ===
            chunks: list[bytes] = []

            # Capture context for the generator
            captured_wrapper_token = wrapper_token
            captured_wrapper_context = wrapper_context

            async def stream_with_span():
                # Re-attach context for the generator
                stream_token = attach(captured_wrapper_context)
                try:
                    async for chunk in _stream_and_capture(upstream_response, chunks):
                        yield chunk

                    # After streaming, parse and record attributes
                    response_body = _parse_sse_response(chunks)

                    if response_body:
                        resp_attrs = response_attributes(response_body)
                        for key, value in resp_attrs.items():
                            llm_span.set_attribute(key, value)

                    if status_code >= 400:
                        llm_span.set_status(Status(StatusCode.ERROR, f"HTTP {status_code}"))
                        wrapper_span.set_status(Status(StatusCode.ERROR, f"HTTP {status_code}"))
                    else:
                        llm_span.set_status(Status(StatusCode.OK))
                        wrapper_span.set_status(Status(StatusCode.OK))

                    logger.info(f"Witnessed: model={model_name}, session={session_id[:8] if session_id else 'none'}, status={status_code}")

                finally:
                    llm_span.end()
                    wrapper_span.end()
                    try:
                        detach(stream_token)
                    except ValueError:
                        pass  # Cross-context detach, harmless
                    # Don't detach captured_wrapper_token here - it belongs to the
                    # original request context which is gone by streaming time.
                    # Attempting to detach it causes ValueError spam in logs.

            # Return streaming response - generator handles span cleanup
            return StreamingResponse(
                stream_with_span(),
                status_code=status_code,
                headers=response_headers,
                media_type="text/event-stream",
            )

        else:
            # === Non-streaming response ===
            response_content = upstream_response.content

            try:
                response_body = json.loads(response_content)
                resp_attrs = response_attributes(response_body)
                for key, value in resp_attrs.items():
                    llm_span.set_attribute(key, value)
            except json.JSONDecodeError:
                pass

            if status_code >= 400:
                llm_span.set_status(Status(StatusCode.ERROR, f"HTTP {status_code}"))
                wrapper_span.set_status(Status(StatusCode.ERROR, f"HTTP {status_code}"))
            else:
                llm_span.set_status(Status(StatusCode.OK))
                wrapper_span.set_status(Status(StatusCode.OK))

            logger.info(f"Witnessed: model={model_name}, session={session_id[:8] if session_id else 'none'}, status={status_code}")

            llm_span.end()
            wrapper_span.end()
            detach(wrapper_token)

            return Response(
                content=response_content,
                status_code=status_code,
                headers=response_headers,
            )

    except Exception as e:
        llm_span.record_exception(e)
        llm_span.set_status(Status(StatusCode.ERROR, str(e)))
        llm_span.end()
        wrapper_span.record_exception(e)
        wrapper_span.set_status(Status(StatusCode.ERROR, str(e)))
        wrapper_span.end()
        detach(wrapper_token)
        logger.error(f"Argonath error: {e}")
        raise
