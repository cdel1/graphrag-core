# `llm/` — INTERFACE (BB1 supporting)

**Protocol:** `LLMClient`
**Source:** [`graphrag_core/interfaces.py`](../interfaces.py) lines 72–90
**Default implementations:** [`anthropic.py`](anthropic.py), [`openai.py`](openai.py), [`base.py`](base.py)
**Vocabulary:** see `tessera/CONTEXT.md`

---

## `LLMClient`

Provider-agnostic interface to a chat-completion-style LLM. Used by `LLMExtractionEngine`, `LLMCurationLayer`, and (in Lacuna) the `GenerationLLM` Protocol for report writing.

### Interface

```python
async def complete(
    self,
    messages: list[dict[str, str]],
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str: ...

async def complete_json(
    self,
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> BaseModel: ...
```

### Contracts

- **`complete`** returns the raw text response. No JSON parsing, no schema validation.
- **`complete_json`** returns a *validated* Pydantic model instance. Implementations must:
  - Pass the JSON Schema (or equivalent) to the provider via structured-output mode if available.
  - Validate the response against `schema` client-side regardless of provider claims.
  - Raise on validation failure — never return a partial / loosely-typed object.
- **`temperature=0.0` is the documented default** for graphrag-core's extraction path. Callers may override but should justify.
- **`max_tokens`** is the output cap, not the prompt cap. Implementations should not silently truncate; if `max_tokens` is exceeded the response is whatever the provider returned at the cutoff.
- **`messages` format follows the OpenAI convention** (`[{"role": "user", "content": "..."}, ...]`). Implementations adapt to provider-specific shapes internally.

### Error modes

- Provider rate limit / 5xx → propagate provider exception; caller chooses retry.
- Provider 4xx (auth, payload) → propagate; caller fixes config or input.
- `complete_json` validation failure → raise `ValidationError` (Pydantic). Caller may catch and retry with adjusted prompt.

### Performance invariants

- Latency dominated by provider call (often 1–30s).
- No graph I/O.
- No per-call setup cost beyond authentication (cached at instance level).

### Structured-output note

graphrag-core specifically uses `strict: False` in the OpenAI client because the extraction schema includes `dict[str, Any]` properties on `ExtractedNode.properties`, which is incompatible with OpenAI's `strict: True` mode. Pydantic validates client-side instead. See `repos/lacuna/CLAUDE.md` "Gotchas."

### Reference impls

- `AnthropicLLMClient` — uses Anthropic's tool-use loop for structured output.
- `OpenAILLMClient` — uses `response_format={"type": "json_schema"}` when `complete_json` is called.

Both implement the Protocol identically from the caller's perspective. Lacuna swaps providers via env var (`LLM_PROVIDER`), no code change.

---

## Implementation skeleton (new provider)

```python
class MyProviderLLMClient:
    def __init__(self, api_key: str, model: str = "default"):
        self._client = MyProviderSDK(api_key)
        self._model = model

    async def complete(self, messages, system=None, temperature=0.0, max_tokens=4096):
        response = await self._client.chat(
            model=self._model,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content

    async def complete_json(self, messages, schema, system=None, temperature=0.0, max_tokens=4096):
        # 1. Convert schema -> provider's structured-output format if available.
        # 2. Call provider.
        # 3. Parse response text as JSON.
        # 4. Validate against `schema` (Pydantic).
        # 5. Return validated instance.
        ...
```

### Test checklist

- `complete`: returns string, never None.
- `complete_json`: returns validated model instance.
- `complete_json`: malformed provider output → `ValidationError`.
- Temperature `0.0` is the documented default.
- Mock provider: same input twice → same output (within provider determinism).
