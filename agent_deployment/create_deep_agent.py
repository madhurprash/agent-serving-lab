import os
import sys
from dotenv import load_dotenv
from typing import Literal

from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

load_dotenv()

# Enable verbose logging to see what's happening
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Tools ----------
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# ---------- Sub-agents ----------
sub_research_prompt = """You are a dedicated researcher. Conduct deep research on the given subtopic
and return a detailed, self-contained answer. Only your FINAL answer is forwarded."""
research_sub_agent = {
    "name": "research-agent",
    "description": "In-depth researcher for a single subtopic at a time.",
    "prompt": sub_research_prompt,
    # IMPORTANT: tool names as strings
    "tools": ["internet_search"],
}

sub_critique_prompt = """You are a dedicated editor. Critique the report in `final_report.md`
against the question in `question.txt`. Be thorough and actionable."""
critique_sub_agent = {
    "name": "critique-agent",
    "description": "Critiques the final report for clarity, coverage, structure.",
    "prompt": sub_critique_prompt,
}

# ---------- Main agent instructions ----------
research_instructions = """You are an expert researcher. Write a polished report in Markdown.
Keep a running `question.txt`. Use `internet_search` to gather evidence.
Write your final output to `final_report.md`. Cite sources as [Title](URL).
Structure with #, ##, ### headings and include a Sources section."""

# ---------- Model (your local vLLM OpenAI-compatible server) ----------
logger.info(f"🔗 Connecting to vLLM server at: {os.environ.get('OPENAI_BASE_URL', 'not set')}")
logger.info(f"🎯 Using model: openai/gpt-oss-20b")

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0.1,
    max_tokens=2048,
    # rely on env:
    #   OPENAI_BASE_URL=http://127.0.0.1:8000/v1
    #   OPENAI_API_KEY=sk-local-demo
    # Hint to keep "thinking" short so it doesn't consume all tokens
    extra_body={"reasoning": {"effort": "low"}},
    verbose=True,  # Enable verbose mode
)

agent = create_deep_agent(
    tools=[internet_search],   # callable is fine here
    instructions=research_instructions,
    subagents=[critique_sub_agent, research_sub_agent],
    model=llm,
).with_config({"recursion_limit": 1000})

if __name__ == "__main__":
    question = "Compare NVIDIA L40S vs L40 for vLLM inference (throughput, VRAM fit, FP8 KV cache). Include sources."

    print("=" * 80)
    print("🤖 Starting Deep Agent Research...")
    print("=" * 80)
    print(f"\n📋 Question: {question}\n")
    print("=" * 80)

    step_counter = 0

    # Stream the agent execution to see real-time progress
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values"
    ):
        if "messages" in chunk and len(chunk["messages"]) > 0:
            last_msg = chunk["messages"][-1]
            msg_type = getattr(last_msg, "type", last_msg.__class__.__name__)
            content = getattr(last_msg, "content", "")

            step_counter += 1

            # Show different types of messages with clear formatting
            if msg_type == "ai":
                print(f"\n🧠 [Step {step_counter}] AI Agent:")
                print("-" * 80)
                if isinstance(content, str):
                    print(content[:1000])  # Show first 1000 chars
                    if len(content) > 1000:
                        print(f"\n... (truncated, total {len(content)} chars)")
                else:
                    print(content)

                # Show tool calls if present
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print("\n🔧 Tool Calls:")
                    for tc in last_msg.tool_calls:
                        print(f"  - {tc.get('name', 'unknown')}: {tc.get('args', {})}")

            elif msg_type == "tool":
                print(f"\n🔨 [Step {step_counter}] Tool Result:")
                print("-" * 80)
                if isinstance(content, str):
                    print(content[:500])  # Show first 500 chars of tool output
                    if len(content) > 500:
                        print(f"\n... (truncated, total {len(content)} chars)")
                else:
                    print(content)

            elif msg_type == "human":
                print(f"\n👤 [Step {step_counter}] Human:")
                print("-" * 80)
                print(content)

            print()

    print("\n" + "=" * 80)
    print("✅ Agent execution complete!")
    print("=" * 80)

    # If the agent wrote a report, show it
    if os.path.exists("final_report.md"):
        print("\n📄 === final_report.md ===\n")
        with open("final_report.md", "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("\n⚠️  (no final_report.md found yet — the agent may need another iteration)")
