# Ollama compaction proxy

Native, server-side **context compaction** for locally hosted Ollama models —
on both provider wire protocols, consumable unmodified by the official SDKs:

- **Anthropic Messages** (`/v1/messages`) — the `compact-2026-01-12` beta
  contract: `context_management.edits` with `compact_20260112`, the
  `compaction` content block, `compaction_delta` streaming, `stop_reason:
  "compaction"`, `usage.iterations`.
- **OpenAI Responses** (`/v1/responses`) — `context_management:
  [{"type":"compaction","compact_threshold":N}]`, the `compaction` output
  item, the `compaction_trigger` input item, and the standalone
  `POST /v1/responses/compact` endpoint.

The proxy fronts Ollama's own compat endpoints (Ollama ≥ 0.32 serves both
dialects natively), so it is a *compaction middleware*, not a protocol
translator: requests that need no compaction work pass through byte-for-byte.

```
Anthropic SDK ──┐                          ┌── /v1/messages ──────┐
OpenAI SDK ─────┼──► compaction proxy ─────┼── /v1/responses ─────┼──► Ollama
Agents SDK ─────┘        :8082             └── /v1/chat/completions┘   :11434
                           │
                           └──► ollama-tokenizer-service :8081  (rendered counts)
```

## Repository layout

Two independent Go modules — the proxy, and the tokenizer primitive it counts
with. They deploy as separate systemd units on the same host and talk over
loopback.

| Path | Module | Port | What it is |
|---|---|---|---|
| `.` | `github.com/andyjmorgan/ollama-compaction-proxy` | 8082 | This proxy: compaction on the Anthropic and OpenAI wire protocols |
| `tokenizer-service/` | `.../tokenizer-service` | 8081 | [Exact token counts](tokenizer-service/README.md) from the model's own tokenizer |

Each builds and tests from its own directory (`go test ./...`); the nested
module is not part of the root module's package tree.

## How compaction works

1. Each request is inspected. A round-tripped compaction block/item is
   **reconstituted**: the compacted history is replaced by the carried summary
   before the model sees anything.
2. The post-substitution request is **counted using the model tokenizer** (via the tokenizer
   service, after dialect normalization and model-specific rendering).
3. If the request opts into compaction and the count crosses the trigger, the
   history is **summarized through Ollama** (`COMPACT_MODEL`, or the request's
   model) and the request is rebuilt around the summary.
4. The response carries the compaction state in the provider's native shape.
   The client's ordinary append-output-to-input loop round-trips it; the proxy
   holds **no state** — everything rides the message contract.

State integrity: `encrypted_content` is `base64url(JSON ‖ HMAC-SHA256)`.
Signed, not encrypted — the payload is the client's own conversation summary;
what matters is rejecting tampered/foreign state. Policy per dialect:

- **Anthropic**: bad blob falls back to the block's client-visible plaintext
  `content`; if that is also null, the block is a no-op (Anthropic's own
  failed-compaction semantic) and full history is kept.
- **OpenAI**: bad blob → 400. `encrypted_content` is the *only* carrier of the
  compacted context there; silently dropping it would produce amnesiac answers.

## Statelessness contract

The proxy stores nothing. `previous_response_id` and `conversation` are
rejected with a clear 400; `store` is accepted and ignored. Point the SDKs at
the proxy and chain turns client-side:

```python
# OpenAI / Agents SDK
client = OpenAI(base_url="http://spark:8082/v1", api_key="unused")
r = client.responses.create(model="gemma4:e4b", store=False, input=history,
    extra_body={"context_management": [{"type": "compaction", "compact_threshold": 8000}]})
history += [item.model_dump(exclude_unset=True) for item in r.output]  # round-trips the compaction item

# Anthropic SDK
client = anthropic.Anthropic(base_url="http://spark:8082", api_key="unused")
r = client.beta.messages.create(model="gemma4:e4b", max_tokens=1000,
    betas=["compact-2026-01-12"],
    context_management={"edits": [{"type": "compact_20260112",
                                   "trigger": {"type": "input_tokens", "value": 8000}}]},
    messages=history)
history.append({"role": "assistant", "content": [b.model_dump(exclude_unset=True) for b in r.content]})
```

Claude Code / Claude Agent SDK use **client-side** compaction, a different
contract from the server-side compact edit above. Opt in with
`X-Claude-Compaction: native` and configure `CLAUDE_MODEL_POLICY_FILE` (see
`deploy/claude-models.json`). The proxy counts each Messages request using the
Anthropic dialect and returns a standard `prompt is too long` error at that
model's budget. Claude performs its own reactive summary and continues the
same session, including native subagents. Recognized summary requests can use
the physical window minus the output reserve. The proxy adds a generic exact-fact
retention requirement to the native summary instruction and counts that augmented
request. `X-Ollama-Thinking: disabled` explicitly restores disabled thinking when
Claude omits its thinking field for unfamiliar model names. For summary requests
only, use `X-Ollama-Summary-Thinking: disabled` instead; ordinary tasks then keep
their model default. Counting outages fail closed on
this opt-in path; requests without the header retain their previous behavior.

`GET /v1/compaction/models` (also `/v1/models` when accessed directly) exposes
the configured serving windows, output limits and compaction budgets. In Claude
2.1.269, generic gateway `/v1/models` discovery is only a picker facility and
ignores these context fields. The SDK application must read the explicit
policy endpoint and configure the runtime. The process-global window cannot
represent mixed Qwen/Gemma limits; the proxy applies each model's own budget.

Validated application and full test harness:
`/home/localuser/source/anthropic-agent-sdk-example`. Runtime compatibility is
pinned to Claude Code 2.1.269; rerun the native SDK compaction acceptance test
before upgrading. Summary recognition is compatibility logic, not an auth
boundary, and never bypasses the physical context reserve.

The Agents SDK's `OpenAIResponsesCompactionSession` works against
`/v1/responses/compact` — note its *client-side* model-name gate accepts only
`gpt-*`-style names (`gpt-oss:20b` passes; alias other models with
`ollama cp` if needed).

## Endpoints

| Endpoint | Behavior |
|---|---|
| `GET /v1/compaction/models` | Configured native Claude serving policies (also GET /v1/models directly) |
| `POST /v1/messages` | Anthropic dialect + compaction |
| `POST /v1/messages/count_tokens` | Proxy-implemented (Ollama has none); substitution-aware |
| `POST /v1/responses` | OpenAI dialect + compaction |
| `POST /v1/responses/compact` | Standalone client-driven compaction |
| `POST /v1/chat/completions` | Byte passthrough + stream watchdog (no compaction) |
| `GET /health` | Proxy + Ollama + tokenizer reachability |

## What the proxy fixes vs passes through

**Fixed** (they break SDKs or the contract):
- `content: null` messages (Ollama 400s) — normalized.
- Swallowed mid-stream Ollama errors — the proxy tracks terminal events and
  synthesizes a proper `error` / `response.failed` frame instead of letting
  the SDK hang on a silently truncated stream.
- Missing `count_tokens` endpoint — implemented.
- `math/rand` outer response IDs — re-minted from crypto/rand on rewrite turns.

**Passed through, documented** (upstream Ollama bugs, better fixed there):
- `/v1/responses` streams suppress text output once any tool call appears.
- `output_index` collides for multiple streamed tool calls.
- Streaming vs non-streaming inner item IDs differ for the same request.
- `usage.input_tokens_details.cached_tokens` / `reasoning_tokens` hardcoded 0.
- `message_start.usage.input_tokens` is a `len/4` estimate; `message_delta`'s
  value is authoritative (SDKs already treat it that way).

**Counting caveat**: these are rendering-based estimates, not a promise of
zero drift. Anthropic requests must use `X-Tokenizer-Dialect: anthropic` at the
tokenizer service (the proxy sets it). Generic flattening loses tool schemas
and call/result structure. Qwen 3.6 plain-text parity was exact in the lab;
small residual differences remain for complex tool-bearing prompts. Old
Qwen models using unsupported Jinja templates can still fall back to
concatenation; inspect the tokenizer's render tier before enabling a policy.

## Configuration

| Env | Default | |
|---|---|---|
| `LISTEN_ADDR` | `:8082` | |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | |
| `TOKENIZER_URL` | `http://127.0.0.1:8081` | ollama-tokenizer-service |
| `CLAUDE_MODEL_POLICY_FILE` | *(unset)* | JSON map of actual Ollama model IDs to context_window, max_output_tokens, compact_at_input_tokens; native Claude opt-in only |
| `COMPACT_MODEL` | *(request model)* | dedicated summarizer model |
| `MODEL_PREFIX` | `agent/` | routing prefix stripped from model names; empty disables |
| `COMPACT_HMAC_KEY` | *(random per boot + warning)* | persist it — deploy generates one in `/etc/ollama-compaction-proxy/env` |
| `COMPACT_DEFAULT_TRIGGER_ANTHROPIC` | `150000` | when the edit has no trigger |
| `COMPACT_DEFAULT_THRESHOLD_OPENAI` | `200000` | when the entry has no threshold |
| `COMPACT_MIN_TRIGGER` | `1024` | floor on client-supplied triggers |
| `MAX_BODY_BYTES` | `67108864` | |
| `SUMMARIZE_TIMEOUT` | `5m` | |
| `UPSTREAM_TIMEOUT` | `10m` | non-streaming calls |
| `LOG_LEVEL` | `info` | |

Logging is structured JSON (journald). Requests log model, token counts,
trigger decisions, summarizer spend, and stream health — **never** message
content, summaries, or blobs.

## Gateway integration (SlipSpace)

The proxy is wired into the SlipSpace gateway as the `agent-compaction`
provider: any model prefixed `agent/` (e.g. `agent/gemma4:e4b`) routes here on
the chat, responses, and messages protocols, and the proxy strips the prefix
before resolving the model — so every model the Spark's Ollama serves gets the
compaction lane with zero per-model config. `/v1/messages/count_tokens` and
`/v1/responses/compact` ride a passthrough family on the same provider.

```python
client = OpenAI(base_url="https://sluice.donkeywork.dev/v1", api_key="<slipspace key>")
r = client.responses.create(model="agent/gemma4:e4b", store=False, ...)
```

Caveat: the OpenAI **Agents SDK** parses `provider/` prefixes in model strings
itself and rejects `agent/...` — pass a Model object to bypass its resolver:
`Agent(model=OpenAIResponsesModel(model="agent/gemma4:e4b", openai_client=client))`.

## Development

```console
go build ./... && go test ./... -race     # hermetic; scripted fake Ollama + tokenizer
```

SDK-fidelity smoke tests (real SDKs, real models, run against a deployed
proxy):

```console
PROXY_URL=http://192.168.69.28:8082 MODEL=gemma4:e4b \
  <anthropic-venv>/bin/python sdk-tests/anthropic_smoke.py
PROXY_URL=http://192.168.69.28:8082 MODEL=gemma4:e4b \
  <openai-venv>/bin/python sdk-tests/openai_smoke.py
PROXY_URL=http://192.168.69.28:8082 MODEL=gemma4:e4b \
  <openai-venv>/bin/python sdk-tests/agents_smoke.py
```

Wire types come from [slipspace-gateway](https://github.com/andyjmorgan/slipspace-gateway)'s
`protocols/` packages (pinned at a pseudo-version — the repo's v2 tags predate
a `/v2` module path and are not resolvable). The block registries there are
sealed; incoming compaction blocks are read via the `UnknownBlock` passthrough
and outgoing ones are constructed locally.

## Deployment

```console
./deploy/deploy.sh          # arm64 cross-compile → scp → systemd on the Spark
```

Runs as `ollama-compaction-proxy.service` beside `ollama.service` and
`ollama-tokenizer.service`, as the `ollama` user, HTTP-only. The deploy script
generates a persistent `COMPACT_HMAC_KEY` on first install so compaction blobs
survive restarts.
