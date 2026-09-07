#!/usr/bin/env python3
"""Example agent: OpenAI Agents SDK against the compaction proxy.

A tool-using agent with server-side compaction enabled via
ModelSettings.context_management. Session history accumulates across turns;
when the input crosses the threshold, the proxy compacts mid-run and the
compaction item lands in the session, shrinking every later request.

Usage:
  PROXY_URL=http://192.168.69.28:8082 MODEL=gemma4:e4b THRESHOLD=2048 \
    python example_agent_openai.py
"""
import asyncio
import os
import sys
import time

from agents import (
    Agent, ModelSettings, OpenAIResponsesModel, Runner, function_tool,
    set_default_openai_api, set_default_openai_client, set_tracing_disabled,
)
from agents.memory.sqlite_session import SQLiteSession
from openai import AsyncOpenAI

PROXY = os.environ.get("PROXY_URL", "http://192.168.69.28:8082")
MODEL = os.environ.get("MODEL", "gemma4:e4b")
THRESHOLD = int(os.environ.get("THRESHOLD", "30000"))
DOC_REPEAT = int(os.environ.get("DOC_REPEAT", "400"))

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

@function_tool
def get_document(name: str) -> str:
    """Fetch an internal document by name. Available: architecture, incidents, roadmap."""
    print(f"    -> tool get_document({name!r})")
    return DOCUMENTS.get(name, "unknown document")

async def session_stats(session):
    items = await session.get_items()
    kinds = [i.get("type") for i in items if isinstance(i, dict)]
    return len(items), kinds.count("compaction")

async def main():
    print(f"== openai-agents example agent ==  model={MODEL} threshold={THRESHOLD}")
    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=PROXY + "/v1", api_key=os.environ.get("API_KEY", "not-needed"), timeout=600)
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("responses")

    agent = Agent(
        name="research-agent",
        instructions=(
            "You are a terse research agent. Use get_document when asked about "
            "documents. Answer in one short sentence."
        ),
        # A Model object bypasses the Agents SDK's own provider-prefix parser
        # (it would otherwise reject "agent/..." as an unknown provider).
        model=OpenAIResponsesModel(model=MODEL, openai_client=client),
        tools=[get_document],
        model_settings=ModelSettings(
            store=False,
            context_management=[{"type": "compaction", "compact_threshold": THRESHOLD}],
        ),
    )
    session = SQLiteSession(session_id="example", db_path=":memory:")

    prompts = [
        "Remember this: the release codeword is MERIDIAN. Confirm in one short sentence.",
        "Fetch the architecture document and name its three tiers.",
        "Fetch the incidents document and name one incident cause.",
        "Fetch the roadmap document and name the Q3 deliverable.",
        "What is the release codeword? Answer with the codeword only.",
    ]

    answer = ""
    for i, p in enumerate(prompts, 1):
        print(f"\n[{i}] USER: {p[:80]}")
        t0 = time.monotonic()
        result = await Runner.run(agent, p, session=session)
        wall = time.monotonic() - t0
        answer = result.final_output or ""
        n_items, n_compactions = await session_stats(session)
        # usage from raw responses = tokens the model ACTUALLY processed
        # (Ollama prompt_eval), the ground truth that trimming happened.
        processed = [f"{rr.usage.input_tokens}in/{rr.usage.output_tokens}out"
                     for rr in result.raw_responses if rr.usage]
        print(f"    AGENT: {answer[:200]}")
        print(f"    [turn {wall:.1f}s | model processed: {', '.join(processed)}]")
        print(f"    [session: {n_items} items, {n_compactions} compaction item(s)]")

    _, n_compactions = await session_stats(session)
    assert n_compactions >= 1, "context blast never triggered compaction"
    assert "MERIDIAN" in answer.upper(), f"codeword lost across compaction: {answer!r}"
    print(f"\nRESULT: {n_compactions} compaction(s) fired and the codeword survived. PASS")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
