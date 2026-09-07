#!/usr/bin/env python3
"""SDK-fidelity smoke test: anthropic-python against the compaction proxy.

Usage: PROXY_URL=http://192.168.69.28:8082 MODEL=gemma4:e4b python anthropic_smoke.py
"""
import os
import sys

import anthropic

PROXY = os.environ.get("PROXY_URL", "http://192.168.69.28:8082")
MODEL = os.environ.get("MODEL", "gemma4:e4b")

client = anthropic.Anthropic(base_url=PROXY, api_key="not-needed")

FILLER = (
    "The gateway routes traffic by protocol and binds upstreams by model. "
    "Records spool to disk as ndjson and ship to object storage out of band. "
    "The circuit breaker sheds load from flapping upstreams automatically. "
) * 30  # ~ enough tokens to cross a 1024 trigger with history

def ok(name):
    print(f"  PASS {name}")

def main():
    print("== anthropic-python smoke ==")

    # 1. Plain create (passthrough path).
    r = client.messages.create(
        model=MODEL, max_tokens=40,
        messages=[{"role": "user", "content": "Say OK and nothing else."}],
    )
    assert any(b.type == "text" and b.text for b in r.content), "no text"
    assert r.usage.input_tokens > 0
    ok("plain create")

    # 2. Streaming create (passthrough path).
    with client.messages.stream(
        model=MODEL, max_tokens=40,
        messages=[{"role": "user", "content": "Say OK and nothing else."}],
    ) as stream:
        final = stream.get_final_message()
    assert any(b.type == "text" and b.text for b in final.content), "no text"
    ok("plain stream")

    # 3. count_tokens (proxy-implemented; Ollama has no such endpoint).
    n = client.messages.count_tokens(
        model=MODEL,
        messages=[{"role": "user", "content": "Explain Kubernetes."}],
    )
    assert n.input_tokens > 0
    ok(f"count_tokens ({n.input_tokens} tokens)")

    # 4. Compaction end-to-end via the beta surface.
    history = [
        {"role": "user", "content": "Context dump: " + FILLER},
        {"role": "assistant", "content": "Understood. I have absorbed the context."},
        {"role": "user", "content": "Summarize our state in one sentence."},
    ]
    r = client.beta.messages.create(
        model=MODEL, max_tokens=400,
        betas=["compact-2026-01-12"],
        context_management={"edits": [{
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 1024},
        }]},
        messages=history,
    )
    kinds = [b.type for b in r.content]
    assert kinds[0] == "compaction", f"expected compaction block first, got {kinds}"
    block = r.content[0]
    assert block.content, "compaction block has no summary"
    assert block.encrypted_content, "compaction block has no blob"
    assert any(i.type == "compaction" for i in (r.usage.iterations or [])), "no compaction iteration"
    ok("compaction triggered; block + iterations present")

    # 5. Round-trip: append the whole content array back, continue, and check
    #    the counted context dropped.
    before = client.beta.messages.count_tokens(
        model=MODEL, betas=["compact-2026-01-12"], messages=history,
    ).input_tokens

    history.append({"role": "assistant", "content": [b.model_dump(exclude_unset=True) for b in r.content]})
    history.append({"role": "user", "content": "And what did I first ask about?"})

    after = client.beta.messages.count_tokens(
        model=MODEL, betas=["compact-2026-01-12"], messages=history,
    ).input_tokens
    assert after < before, f"post-compaction count {after} not below pre {before}"
    ok(f"round-trip count drop ({before} -> {after})")

    r2 = client.beta.messages.create(
        model=MODEL, max_tokens=400,
        betas=["compact-2026-01-12"],
        thinking={"type": "disabled"},
        messages=history,
    )
    assert any(b.type == "text" and b.text for b in r2.content), "no text in continuation"
    assert all(b.type != "compaction" for b in r2.content), "no new compaction expected"
    ok("continuation after round-trip")

    # 6. Streaming compaction: accumulator must reconstruct the block.
    stream_history = [
        {"role": "user", "content": "Context dump: " + FILLER},
        {"role": "assistant", "content": "Absorbed."},
        {"role": "user", "content": "One-line status?"},
    ]
    with client.beta.messages.stream(
        model=MODEL, max_tokens=400,
        betas=["compact-2026-01-12"],
        context_management={"edits": [{
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 1024},
        }]},
        messages=stream_history,
    ) as stream:
        final = stream.get_final_message()
    kinds = [b.type for b in final.content]
    assert kinds[0] == "compaction", f"streamed content kinds: {kinds}"
    assert final.content[0].content, "streamed block has no summary"
    ok("streaming compaction reconstructed by SDK accumulator")

    # 7. pause_after_compaction: block-only, stop_reason compaction.
    r3 = client.beta.messages.create(
        model=MODEL, max_tokens=400,
        betas=["compact-2026-01-12"],
        context_management={"edits": [{
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 1024},
            "pause_after_compaction": True,
        }]},
        messages=stream_history,
    )
    assert r3.stop_reason == "compaction", f"stop_reason={r3.stop_reason}"
    assert len(r3.content) == 1 and r3.content[0].type == "compaction"
    ok("pause_after_compaction")

    print("anthropic smoke: ALL PASS")

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"  FAIL: {e}")
        sys.exit(1)
