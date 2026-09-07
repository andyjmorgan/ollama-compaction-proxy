#!/usr/bin/env python3
"""SDK-fidelity smoke test: openai-python against the compaction proxy.

Usage: PROXY_URL=http://192.168.69.28:8082 MODEL=gemma4:e4b python openai_smoke.py
"""
import os
import sys

from openai import OpenAI

PROXY = os.environ.get("PROXY_URL", "http://192.168.69.28:8082")
MODEL = os.environ.get("MODEL", "gemma4:e4b")

client = OpenAI(base_url=PROXY + "/v1", api_key="not-needed")

FILLER = (
    "The gateway routes traffic by protocol and binds upstreams by model. "
    "Records spool to disk as ndjson and ship to object storage out of band. "
    "The circuit breaker sheds load from flapping upstreams automatically. "
) * 30

def ok(name):
    print(f"  PASS {name}")

def user_msg(text):
    return {"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": text}]}

def main():
    print("== openai-python smoke ==")

    # 1. Plain create (passthrough).
    r = client.responses.create(model=MODEL, input="Say OK and nothing else.", store=False)
    assert r.output_text
    assert r.usage.input_tokens > 0
    ok("plain create")

    # 2. Streaming create (passthrough).
    text = ""
    with client.responses.stream(model=MODEL, input="Say OK and nothing else.", store=False) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                text += event.delta
        final = stream.get_final_response()
    assert final.output_text
    ok("plain stream")

    # 3. Statelessness contract.
    try:
        client.responses.create(model=MODEL, input="hi", previous_response_id="resp_x", store=False)
        raise AssertionError("previous_response_id should 400")
    except Exception as e:
        assert "stateless" in str(e), str(e)
    ok("previous_response_id rejected")

    # 4. Compaction via context_management.
    history = [
        user_msg("Context dump: " + FILLER),
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Understood, context absorbed."}]},
        user_msg("Summarize our state in one sentence."),
    ]
    r = client.responses.create(
        model=MODEL, store=False,
        input=history,
        extra_body={"context_management": [{"type": "compaction", "compact_threshold": 1024}]},
    )
    items = [item.type for item in r.output]
    assert items[0] == "compaction", f"expected compaction first, got {items}"
    comp = r.output[0]
    assert comp.encrypted_content, "compaction item has no encrypted_content"
    assert r.output_text, "no generation text alongside compaction"
    ok("compaction item emitted")

    # 5. Round-trip: append output to input, continue.
    history += [item.model_dump(exclude_unset=True) for item in r.output]
    history.append(user_msg("And what did I first send you?"))
    r2 = client.responses.create(model=MODEL, store=False, input=history)
    assert r2.output_text
    assert all(item.type != "compaction" for item in r2.output)
    ok("round-trip continuation")

    # 6. Streaming compaction: the item must arrive via output_item events and
    #    the accumulator must reconstruct the final response.
    stream_history = [
        user_msg("Context dump: " + FILLER),
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Absorbed."}]},
        user_msg("One-line status?"),
    ]
    saw_item = False
    with client.responses.stream(
        model=MODEL, store=False, input=stream_history,
        extra_body={"context_management": [{"type": "compaction", "compact_threshold": 1024}]},
    ) as stream:
        for event in stream:
            if event.type == "response.output_item.done" and event.item.type == "compaction":
                saw_item = True
        final = stream.get_final_response()
    assert saw_item, "compaction item never streamed"
    assert final.output[0].type == "compaction"
    assert final.output_text
    ok("streaming compaction reconstructed by SDK accumulator")

    # 7. Standalone /v1/responses/compact.
    compacted = client.responses.compact(
        model=MODEL,
        input=[
            user_msg("user question one"),
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "assistant answer one"}]},
            user_msg("user question two"),
        ],
    )
    assert compacted.object == "response.compaction"
    types = [item.type for item in compacted.output]
    assert types[-1] == "compaction", types
    assert types[:-1] == ["message", "message"], types
    ok("responses.compact endpoint")

    print("openai smoke: ALL PASS")

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"  FAIL: {e}")
        sys.exit(1)
