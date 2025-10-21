"""
Script to run benchmarking using the deep agent with questions from the dataset.
This script:
1. Loads the benchmarking dataset
2. Creates the deep agent
3. Runs the deep agent for each question with reference URLs
4. Saves the results (question, reference URLs, ground truth, generated answer) to a DataFrame
"""

import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import Dict, Any, List, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    print("\n" + "="*80)
    print("LOADING CONFIGURATION")
    print("="*80)
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Configuration loaded from: {config_path}")
    print(f"  - Dataset: {config['benchmarking']['dataset_path']}")
    print(f"  - Samples to process: {config['benchmarking']['num_samples']}")
    print(f"  - Output directory: {config['benchmarking']['output_dir']}")
    return config


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the benchmarking dataset"""
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    dataset_path = Path(__file__).parent.parent / config['benchmarking']['dataset_path']

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")

    if dataset_path.suffix == '.parquet':
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == '.csv':
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

    print(f"✓ Dataset loaded successfully!")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Columns: {df.columns.tolist()}")
    return df


def create_deep_agent():
    """Create the deep agent with all necessary tools and configuration"""
    print("\n" + "="*80)
    print("CREATING DEEP AGENT")
    print("="*80)

    from tavily import TavilyClient
    from langchain_openai import ChatOpenAI
    from deepagents import create_deep_agent

    # Setup Tavily client for web search
    print("Setting up Tavily client for web search...")
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

    # Setup sub-agents
    print("Setting up sub-agents...")
    sub_research_prompt = """You are a dedicated researcher. Conduct deep research on the given subtopic
and return a detailed, self-contained answer. Only your FINAL answer is forwarded. Use the internet_search for searching on the reference URLs."""

    research_sub_agent = {
        "name": "research-agent",
        "description": "In-depth researcher for a single subtopic at a time.",
        "prompt": sub_research_prompt,
        "tools": ["internet_search"],
    }

    sub_critique_prompt = """You are a dedicated editor. Critique the report in `final_report.md`
against the question in `question.txt`. Be thorough and actionable."""

    critique_sub_agent = {
        "name": "critique-agent",
        "description": "Critiques the final report for clarity, coverage, structure.",
        "prompt": sub_critique_prompt,
    }

    # Setup main agent instructions
    research_instructions = """You are an expert researcher. Your task is to answer questions based on provided documents.

Your final answer should be a concise sentence in the following format:
Final Answer: put your answer here.

It's critical your answer is concise and follows the format strictly.

VERY IMPORTANT: You may only use the Reference URLs to anwer the question and nothing else. Do not search on your own, just use the reference URLs from the user question."""

    # Setup LLM connection to local vLLM server
    print(f"Connecting to vLLM server at: {os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')}")
    print(f"Using model: openai/gpt-oss-20b")

    llm = ChatOpenAI(
        model="openai/gpt-oss-20b",
        temperature=0.1,
        max_tokens=8192,
        base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "sk-local-demo"),
        verbose=True,
    )

    # Create the deep agent
    print("Creating deep agent with tools and sub-agents...")
    agent = create_deep_agent(
        tools=[internet_search],
        instructions=research_instructions,
        subagents=[critique_sub_agent, research_sub_agent],
        model=llm,
    ).with_config({"recursion_limit": 1000})

    print("✓ Deep agent created successfully!")
    return agent


def format_reference_urls(urls_str: str) -> str:
    """Format the reference URLs into a readable string for the prompt"""
    try:
        # URLs are stored as a JSON string of lists
        urls_data = json.loads(urls_str)

        # Flatten the list and extract URLs
        formatted_urls = []
        for item in urls_data:
            if isinstance(item, list) and len(item) > 0:
                formatted_urls.append(item[0])  # Get the URL from [url, content] pairs

        # Create a formatted string
        if formatted_urls:
            return "\n".join([f"- {url}" for url in formatted_urls]), len(formatted_urls)
        return "No reference URLs provided", 0
    except Exception as e:
        logger.error(f"Error formatting URLs: {e}")
        return str(urls_str), 0


def run_agent_on_question(question: str, reference_urls: str, agent, sample_num: int, total_samples: int) -> str:
    """Run the agent on a single question with reference URLs"""

    print("\n" + "="*80)
    print(f"PROCESSING SAMPLE {sample_num}/{total_samples}")
    print("="*80)

    # Display the question
    print(f"\n📋 QUESTION:")
    print("-"*80)
    print(question)
    print("-"*80)

    # Format and display reference URLs
    formatted_urls, num_urls = format_reference_urls(reference_urls)
    print(f"\n🔗 REFERENCE URLs ({num_urls} total):")
    print("-"*80)
    print(formatted_urls)  # Show first 500 chars
    print("-"*80)

    # Create the full prompt
    full_prompt = f"""Question: {question}

Reference URLs to use:
{formatted_urls}

Please answer the question based on the information from these reference URLs."""

    print(f"\n🤖 RUNNING AGENT...")
    print("-"*80)

    try:
        # Stream the agent execution
        final_answer = ""
        step_count = 0

        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": full_prompt}]},
            stream_mode="values"
        ):
            if "messages" in chunk and len(chunk["messages"]) > 0:
                last_msg = chunk["messages"][-1]
                msg_type = getattr(last_msg, "type", last_msg.__class__.__name__)
                content = getattr(last_msg, "content", "")

                step_count += 1

                if msg_type == "ai":
                    print(f"\n  [Step {step_count}] AI Agent response...")
                    if isinstance(content, str):
                        final_answer = content
                        # Show full response
                        print(f"    Content:\n{content}")

                    # Show tool calls if present
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"    Tool calls: {[tc.get('name', 'unknown') for tc in last_msg.tool_calls]}")
                        for tc in last_msg.tool_calls:
                            print(f"      - {tc.get('name', 'unknown')}: {tc.get('args', {})}")

                elif msg_type == "tool":
                    print(f"  [Step {step_count}] Tool execution result received")
                    # Show full tool result content
                    if isinstance(content, str):
                        print(f"    Tool result:\n{content}")
                    else:
                        print(f"    Tool result: {content}")

        print("-"*80)
        print(f"✓ Agent execution complete! (Total steps: {step_count})")

        # Display final answer
        print(f"\n💡 FINAL ANSWER:")
        print("-"*80)
        print(final_answer)
        print("-"*80)

        return final_answer

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        logger.error(f"Error running agent: {e}", exc_info=True)
        print(f"\n❌ ERROR occurred: {error_msg}")
        return error_msg


def save_results(results_df: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Save results DataFrame to output directory"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_dir = Path(__file__).parent.parent / config['benchmarking']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config['benchmarking']['output_filename']}_{timestamp}"

    # Save as both parquet and CSV
    parquet_path = output_dir / f"{filename}.parquet"
    csv_path = output_dir / f"{filename}.csv"

    results_df.to_parquet(parquet_path, index=False)
    results_df.to_csv(csv_path, index=False)

    print(f"✓ Results saved successfully!")
    print(f"  - Parquet: {parquet_path}")
    print(f"  - CSV: {csv_path}")

    return parquet_path


def main():
    """Main benchmarking execution"""
    print("\n" + "="*80)
    print("🚀 STARTING BENCHMARKING RUN")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    config = load_config()
    benchmark_config = config['benchmarking']

    # Load dataset
    df = load_dataset(config)

    # Select samples to run
    num_samples = benchmark_config['num_samples']
    if num_samples == -1:
        samples_df = df
        print(f"\n📊 Processing ALL {len(samples_df)} samples from dataset")
    else:
        samples_df = df.head(num_samples)
        print(f"\n📊 Processing first {len(samples_df)} samples from dataset")

    # Create the deep agent
    agent = create_deep_agent()

    # Prepare results list
    results = []

    print("\n" + "="*80)
    print("STARTING AGENT PROCESSING")
    print("="*80)

    # Run agent on each sample
    for idx, row in samples_df.iterrows():
        question = row[benchmark_config['question_column']]
        reference_urls = row[benchmark_config['reference_column']]
        ground_truth = row[benchmark_config['ground_truth_column']]

        # Run agent
        generated_answer = run_agent_on_question(
            question,
            reference_urls,
            agent,
            idx + 1,
            len(samples_df)
        )

        # Display ground truth for comparison
        print(f"\n📚 GROUND TRUTH:")
        print("-"*80)
        print(ground_truth)
        print("-"*80)

        # Store results
        results.append({
            'sample_id': idx,
            'question': question,
            'reference_urls': reference_urls,
            'ground_truth': ground_truth,
            'generated_answer': generated_answer,
            'timestamp': datetime.now().isoformat()
        })

        print(f"\n✅ Sample {idx + 1}/{len(samples_df)} completed!")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_path = save_results(results_df, config)

    # Display final summary
    print("\n" + "="*80)
    print("🎉 BENCHMARKING COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"  - Total samples processed: {len(results_df)}")
    print(f"  - Results saved to: {output_path}")
    print(f"  - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "="*80)
    print("SAMPLE RESULTS (First Entry)")
    print("="*80)
    if len(results_df) > 0:
        first = results_df.iloc[0]
        print(f"\nQuestion: {first['question'][:200]}...")
        print(f"\nGenerated Answer: {first['generated_answer'][:200]}...")
        print(f"\nGround Truth: {first['ground_truth'][:200]}...")
    print("="*80)


if __name__ == "__main__":
    main()
