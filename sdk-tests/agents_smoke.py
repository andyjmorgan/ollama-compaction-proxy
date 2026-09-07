#!/usr/bin/env python3
"""SDK-fidelity smoke test: openai-agents SDK against the compaction proxy.

Usage: PROXY_URL=http://192.168.69.28:8082 MODEL=gemma4:e4b python agents_smoke.py
"""
import asyncio
import os
import sys

from agents import Agent, ModelSettings, Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from agents.memory.openai_responses_compaction_session import OpenAIResponsesCompactionSession
from agents.memory.sqlite_session import SQLiteSession
from openai import AsyncOpenAI

PROXY = os.environ.get("PROXY_URL", "http://192.168.69.28:8082")
MODEL = os.environ.get("MODEL", "gemma4:e4b")

FILLER = (
    "The gateway routes traffic by protocol and binds upstreams by model. "
    "Records spool to disk as ndjson and ship to object storage out of band. "
) * 60

def ok(name):
    print(f"  PASS {name}")

async def main():
    print("== openai-agents smoke ==")
    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=PROXY + "/v1", api_key="not-needed")
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("responses")

    agent = Agent(
        name="assistant",
        instructions="You are terse. Answer in one sentence.",
        model=MODEL,
        model_settings=ModelSettings(store=False),
    )

    # 1. Basic run + stateless chaining via to_input_list().
    result = await Runner.run(agent, "Say OK and nothing else.")
    assert result.final_output
    followup = result.to_input_list() + [{"role": "user", "content": "Now say DONE."}]
    result2 = await Runner.run(agent, followup)
    assert result2.final_output
    ok("Runner + to_input_list chaining")

    # 2. Server-side compaction through ModelSettings.context_management.
    compacting = Agent(
        name="assistant",
        instructions="You are terse. Answer in one sentence.",
        model=MODEL,
        model_settings=ModelSettings(
            store=False,
            context_management=[{"type": "compaction", "compact_threshold": 1024}],
        ),
    )
    history = [
        {"role": "user", "content": "Context dump: " + FILLER},
        {"role": "assistant", "content": "Absorbed."},
        {"role": "user", "content": "One-line status?"},
    ]
    result3 = await Runner.run(compacting, history)
    assert result3.final_output
    items = result3.to_input_list()
    kinds = [i.get("type") for i in items if isinstance(i, dict)]
    assert "compaction" in kinds, f"no compaction item in run items: {kinds}"
    ok("ModelSettings.context_management compaction item round-trips")

    # 3. Continue from the compacted history — the SDK replays the item
    #    verbatim and the proxy reconstitutes it.
    followup3 = items + [{"role": "user", "content": "What was my first message about?"}]
    result4 = await Runner.run(compacting, followup3)
    assert result4.final_output
    ok("continuation over replayed compaction item")

    # 4. OpenAIResponsesCompactionSession against /v1/responses/compact.
    # The SDK gates this class on OpenAI-looking model names (gpt-*), a purely
    # client-side check — gpt-oss:20b passes it and is a real local model.
    SESSION_MODEL = os.environ.get("SESSION_MODEL", "gpt-oss:20b")
    session_agent = Agent(
        name="assistant",
        instructions="You are terse. Answer in one sentence.",
        model=SESSION_MODEL,
        model_settings=ModelSettings(store=False),
    )
    inner = SQLiteSession(session_id="smoke", db_path=":memory:")
    session = OpenAIResponsesCompactionSession(
        session_id="smoke", underlying_session=inner, client=client, model=SESSION_MODEL,
        compaction_mode="input",
    )
    r1 = await Runner.run(session_agent, "Remember: the codeword is HELIOTROPE. Say OK.", session=session)
    assert r1.final_output
    for i in range(3):
        await Runner.run(session_agent, f"Filler question {i}: say OK.", session=session)
    # Force the session's client-driven compaction.
    await session.run_compaction({"force": True})
    compacted_items = await inner.get_items()
    kinds = [i.get("type") for i in compacted_items if isinstance(i, dict)]
    assert "compaction" in kinds, f"session did not store a compaction item: {kinds}"
    r5 = await Runner.run(session_agent, "What is the codeword?", session=session)
    assert "HELIOTROPE" in (r5.final_output or "").upper(), f"codeword lost: {r5.final_output}"
    ok("OpenAIResponsesCompactionSession compact + recall")

    print("openai-agents smoke: ALL PASS")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"  FAIL: {e}")
        sys.exit(1)
