# The Argonath

> Everything that passes through the Gates is witnessed.

Pure observation proxy for Anthropic API calls. Emits perfect OpenInference LLM traces to Phoenix. No transformation, just watching.

## What It Does

The Argonath sits between your client and Anthropic. It:

1. **Witnesses** every request and response
2. **Extracts** full OpenInference attributes, preserving content array structure
3. **Emits** LLM spans to Phoenix via OTLP
4. **Proxies** traffic transparently (no modification)

## Correlation

Pass these headers for proper trace correlation:

- `x-session-id`: Groups traces by conversation session
- `traceparent`: W3C trace context for distributed tracing

The Loom can promote these from request body metadata before forwarding here.

## Attributes Emitted

### From Request
- `openinference.span.kind`: "LLM"
- `llm.system`: "anthropic"
- `llm.model_name`: The model being called
- `llm.input_messages.{idx}.message.role/content`: Full message history
- `llm.input_messages.{idx}.message.contents.{cidx}.*`: Multimodal content blocks
- `llm.tools.{idx}.tool.*`: Tool definitions
- `input.value`: Last real user message (for Phoenix UI)

### From Response
- `llm.token_count.prompt/completion/total`: Token usage
- `llm.token_count.prompt_details.cache_read/cache_write`: Anthropic cache metrics
- `llm.output_messages.0.message.*`: Assistant response
- `llm.output_messages.0.message.tool_calls.{idx}.*`: Tool calls
- `output.value`: Response text (for Phoenix UI)
- `session.id`: From x-session-id header

## Running

```bash
docker compose up -d
```

Or for development:

```bash
cd /Pondside/Basement/Argonath
uv run uvicorn argonath.app:app --reload --port 8081
```

## Architecture

```
Client → [The Loom] → The Argonath → Anthropic
              ↓              ↓
         transforms      witnesses
              ↓              ↓
           Logfire       Phoenix
```

The Loom mutates (system prompt, memories, etc.). The Argonath just watches and tells the story.

---

*Named for Tolkien's Pillars of the Kings. Two massive statues flanking the Anduin, hands raised in warning and greeting. Everything that passes through the Gates is witnessed.*
