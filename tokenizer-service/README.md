# Ollama token count service

Estimates how many input tokens a request will cost against a locally hosted
Ollama model, using that model's own tokenizer.

It does one thing. It does not compact context, drop messages, summarize,
route, or run inference — those are consumers of this primitive.

```console
$ curl -s localhost:8081/count-tokens -d '{
    "model": "gemma4:e4b",
    "messages": [{"role": "user", "content": "Explain Kubernetes."}],
    "temperature": 0.7
  }'
{"model":"gemma4:e4b","tokens":19,"estimated":true}
```

## API

`POST /count-tokens` takes arbitrary JSON. The only required field is `model`.

Prompt-bearing fields are recognized and counted: `messages`, `input`,
`instructions`, `system`, `prompt`, `tools`. `think` is also read, because it
decides whether a thinking block is rendered into the prompt. Everything else —
`temperature`, `stream`, `max_tokens`, `metadata`, and any field this service
has never heard of — is ignored, never rejected.

| Status | Body | Cause |
|---|---|---|
| 200 | `{"model","tokens","estimated":true}` | counted |
| 400 | `{"error":"model is required"}` | no usable `model` |
| 400 | `{"error":"invalid JSON"}` | body is not a JSON object |
| 404 | `{"error":"model not found"}` | model not installed in Ollama |
| 500 | `{"error":"..."}` | tokenizer could not be built |

`estimated` is always `true`. Even where drift measures zero today, this is not
an authoritative inference count, and consumers needing context-window safety
should apply their own margin.

`GET /health` returns `{"status":"ok"}` and checks process liveness only. It
never loads a tokenizer or contacts Ollama.

## How it works

Everything comes from the installed model. There is no per-model table to
maintain, and no tokenizer is downloaded from Hugging Face — that would be a
second source of truth able to drift from the artifact actually being served.

1. **Model metadata** — one `POST /api/show` with `verbose:true` per model
   yields the chat template, the `RENDERER` directive, the default system
   prompt, the capabilities, and the full GGUF tokenizer vocabulary. For gemma4
   that response is about 12MB, so it is cached for the process lifetime.
2. **Rendering** — three tiers, highest fidelity first:
   - the model's Go-native renderer, via Ollama's `model/renderers`
     (`RENDERER gemma4`, `RENDERER glimmer`, …)
   - the model's Go chat template, via Ollama's `template` package
   - plain concatenation of the extracted text
   A tier that is unavailable or errors falls to the next one, so a count is
   always returned. The tier used is logged.
3. **Tokenizing** — the vocabulary is handed to Ollama's `tokenizer` package:
   byte-pair, SentencePiece, or WordPiece, selected from
   `tokenizer.ggml.model`. No tokenization algorithm is reimplemented here.

Tokenization is CPU work. The service never loads weights into VRAM and never
runs generation to obtain a count.

### Caching

One in-memory cache keyed by exact model ID, holding both the rendering
metadata and the built tokenizer. Concurrent first requests for a model share a
single load. Failures are not cached, so a model pulled after a 404 works on
the next request.

Budget roughly **75MB per cached model** once the vocabulary's lookup maps are
built — measured at 154MB RSS on the Spark with `gemma4:e4b` (262k tokens,
515k merges) and `muse-glimmer` (202k tokens, 440k merges) both cached. With a
bounded set of installed models this stays well inside the unit's 4G cap.

There is no eviction, no invalidation, and no persistence. Restarting empties
the cache, which is fine.

## Accuracy

Measured against Ollama's real `prompt_eval_count`, on the Spark, across the
five request shapes in the parity harness:

| Model | Renders via | Worst drift |
|---|---|---|
| `gemma4:e4b` | renderer `gemma4` | **0.0%** |
| `gemma4:26b` | renderer `gemma4` | **0.0%** |
| `muse-glimmer:latest` | renderer `glimmer` | **0.0%** |
| `gpt-oss:20b` | Go template | **0.0%** |
| `qwen2.5-coder:7b` | Go template | **-0.5%** (one token, tools shape) |
| `qwen3:4b-instruct-2507` | Jinja template — see below | **-64.7%** |

### Known gap: Jinja chat templates

Ollama 0.32.8 stores some models' chat templates in Jinja rather than Go, and
renders those **inside llama.cpp** — it has no Go Jinja engine to reuse. Such
templates fail `template.Parse` here, so those models fall to plain
concatenation and are undercounted by 30-65%.

Of the models installed on the Spark this affects `qwen3:4b-instruct-2507`
only; the Muse and Gemma models this service was built for are unaffected. A
model on this path is visible in the logs as `render_tier=concat` with a
`render_fallback` explaining why.

Closing it means adding a Go Jinja engine (for example `gonja`) and accepting
the compatibility risk around Jinja features these templates use — `namespace`,
slice syntax such as `messages[::-1]`, and the `tojson` filter. That is a real
dependency, so it is deliberately not in this version.

## Configuration

| Variable | Default | |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `LISTEN_ADDR` | `:8080` | listen address (the unit sets `:8081`) |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warn`, `error` |

## Logging

Structured JSON to stderr, one line per request: model, token count, tokenizer
cache hit/miss, tokenization algorithm, render tier, whether thinking was on,
duration, and any media-block count or render fallback.

Prompt, message, and tool content are **never logged** at any level. Only
shapes, counts, and timings.

## Development

```console
go test ./... -race          # unit tests, no Ollama needed
go build ./...
```

The unit tests run against a fake Ollama with a synthetic byte-level
vocabulary, so they need neither a running Ollama nor model weights.

### Parity harness

```console
OLLAMA_URL=http://192.168.69.28:11434 go test -tags parity ./internal/parity/ -v
PARITY_MODELS=gemma4:e4b,muse-glimmer:latest go test -tags parity ./internal/parity/ -v
```

This loads real models and prints the drift table above. It **never fails on
non-zero drift** — its job is to characterize the estimator, not to gate it.

It obtains actual counts with a `num_predict: 0` chat call, reading
`prompt_eval_count`. That is a measurement for the harness only; the service
itself must never get a count that way.

When drift appears, diff the rendered prompt against Ollama's own using the
`_debug_render_only` request option. That option is underscore-prefixed and
unstable, so it is a debugging aid only — production counts never go through
it.

## Deployment

Runs as a systemd daemon on the Spark, beside Ollama:

```console
./deploy/deploy.sh              # defaults to 192.168.69.28
```

Cross-compiles a static `linux/arm64` binary, installs it and the unit, and
waits for `/health`. The Spark has no Go toolchain, and none is needed.

```console
ssh 192.168.69.28 'systemctl status ollama-tokenizer'
ssh 192.168.69.28 'journalctl -u ollama-tokenizer -f'
```

The unit runs as the `ollama` user with no filesystem access to model blobs —
everything arrives over HTTP.
