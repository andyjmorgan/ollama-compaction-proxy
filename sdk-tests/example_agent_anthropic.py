#!/usr/bin/env python3
"""Example agent: Anthropic SDK against the compaction proxy.

A tool-using research agent with native server-side compaction enabled. Each
tool call returns a large document, inflating the context; when the input
crosses the trigger, the proxy compacts and the agent keeps going — the only
client-side obligation is the standard append-response-content loop.

Usage:
  PROXY_URL=http://192.168.69.28:8082 MODEL=gemma4:e4b TRIGGER=2048 \
    python example_agent_anthropic.py
"""
import os
import sys
import time

import anthropic

PROXY = os.environ.get("PROXY_URL", "http://192.168.69.28:8082")
MODEL = os.environ.get("MODEL", "gemma4:e4b")
TRIGGER = int(os.environ.get("TRIGGER", "30000"))
DOC_REPEAT = int(os.environ.get("DOC_REPEAT", "400"))

client = anthropic.Anthropic(base_url=PROXY, api_key=os.environ.get("API_KEY", "not-needed"), timeout=600)

# ---------------------------------------------------------------- the tool --

DOCUMENTS = {
    "architecture": "The platform has three tiers. " + (
        "The gateway tier terminates TLS, authenticates tenants, and routes by protocol. "
        "The service tier runs stateless workers behind a queue with at-least-once delivery. "
        "The storage tier pairs Postgres for transactions with object storage for blobs. "
    ) * DOC_REPEAT,
    "incidents": "Incident history follows. " + (
        "INC-2214: queue saturation after a deploy doubled message size; fixed by batch caps. "
        "INC-2215: certificate expiry on the internal mesh; fixed by automated rotation. "
        "INC-2219: slow drain during failover traced to sticky connections; fixed by jitter. "
    ) * DOC_REPEAT,
    "roadmap": "Roadmap items follow. " + (
        "Q3 delivers multi-region failover with active-passive Postgres and object replication. "
        "Q4 delivers the plugin marketplace with signed bundles and a review pipeline. "
        "Next year targets usage-based billing with hourly aggregation windows. "
    ) * DOC_REPEAT,
}

TOOLS = [{
    "name": "get_document",
    "description": "Fetch an internal document by name. Available: architecture, incidents, roadmap.",
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": list(DOCUMENTS)}},
        "required": ["name"],
    },
}]

def run_tool(name, tool_input):
    if name == "get_document":
        return DOCUMENTS.get(tool_input.get("name", ""), "unknown document")
    return f"unknown tool {name}"

# ------------------------------------------------------------- agent loop --

CONTEXT_MANAGEMENT = {"edits": [{
    "type": "compact_20260112",
    "trigger": {"type": "input_tokens", "value": TRIGGER},
}]}

def count(history):
    return client.beta.messages.count_tokens(
        model=MODEL, betas=["compact-2026-01-12"], messages=history,
    ).input_tokens

def raw_count(history):
    """What the request would cost with NO compaction: the client-side history
    with compaction blocks stripped, i.e. every document ever fetched."""
    stripped = []
    for m in history:
        c = m.get("content")
        if isinstance(c, list):
            c = [b for b in c if b.get("type") != "compaction"]
            if not c:
                continue
            m = {"role": m["role"], "content": c}
        stripped.append(m)
    return count(stripped)

def turn(history, text):
    """One user turn, running the tool loop to completion."""
    history.append({"role": "user", "content": text})
    compactions = 0

    while True:
        print(f"    [raw history: {raw_count(history)} tok | model will see: {count(history)} tok]")
        t0 = time.monotonic()
        r = client.beta.messages.create(
            model=MODEL, max_tokens=600,
            thinking={"type": "disabled"},
            betas=["compact-2026-01-12"],
            context_management=CONTEXT_MANAGEMENT,
            tools=TOOLS,
            messages=history,
        )

        wall = time.monotonic() - t0
        print(f"    [call {wall:.1f}s | model processed {r.usage.input_tokens} in / {r.usage.output_tokens} out]")

        # The whole content array goes back verbatim — this is the entire
        # compaction protocol from the client's side.
        history.append({
            "role": "assistant",
            "content": [b.model_dump(exclude_unset=True) for b in r.content],
        })

        for b in r.content:
            if b.type == "compaction":
                compactions += 1
                it = next((i for i in (r.usage.iterations or []) if i.type == "compaction"), None)
                spent = f" (summarizer read {it.input_tokens} tok, wrote {it.output_tokens} tok)" if it else ""
                print(f"    ** COMPACTED{spent}")
                print(f"       summary: {b.content[:120]}...")

        if r.stop_reason != "tool_use":
            texts = [b.text for b in r.content if b.type == "text"]
            return " ".join(texts).strip(), compactions

        results = []
        for b in r.content:
            if b.type == "tool_use":
                print(f"    -> tool {b.name}({b.input})")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": run_tool(b.name, b.input),
                })
        history.append({"role": "user", "content": results})

def main():
    print(f"== anthropic example agent ==  model={MODEL} trigger={TRIGGER}")
    history = []
    total_compactions = 0

    prompts = [
        "Remember this: the release codeword is MERIDIAN. Confirm in one short sentence.",
        "Fetch the architecture document and name its three tiers in one sentence.",
        "Fetch the incidents document and name one incident cause in one sentence.",
        "Fetch the roadmap document and name the Q3 deliverable in one sentence.",
        "What is the release codeword? Answer with the codeword only.",
    ]

    for i, p in enumerate(prompts, 1):
        print(f"\n[{i}] USER: {p[:80]}")
        answer, c = turn(history, p)
        total_compactions += c
        print(f"    AGENT: {answer[:200]}")

    print(f"\nfinal: raw history {raw_count(history)} tok vs model-effective {count(history)} tok; "
          f"compactions: {total_compactions}")
    assert total_compactions >= 1, "context blast never triggered compaction"
    assert "MERIDIAN" in answer.upper(), f"codeword lost across compaction: {answer!r}"
    print("RESULT: compaction fired and the codeword survived. PASS")

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
