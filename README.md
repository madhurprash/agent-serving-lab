# Agent Serving Lab

This repository contains code for deploying and testing deep research agents using vLLM and LangGraph.

## Table of Contents
- [Setup](#setup)
- [Loading BrowseComp Dataset](#loading-browsecomp-dataset)
- [Testing the Agent](#testing-the-agent)
- [Configuration](#configuration)

## Setup

### Prerequisites
- Python 3.9+
- Docker (for vLLM backend)
- NVIDIA GPU (L40/L40S or similar)
- Environment variables configured

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```bash
# OpenAI API settings (for vLLM compatibility)
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=sk-local-demo

# Tavily API key for web search
TAVILY_API_KEY=your_tavily_api_key_here
```

## Loading BrowseComp Dataset

The BrowseCompLongContext dataset is a benchmark for testing long-context capabilities of LLMs on research tasks.

### Step 1: Load the Dataset

Run the dataset loading script:

```bash
cd benchmarking_dataset
python load_browsecomp_dataset.py
```

This will:
- Download the BrowseCompLongContext dataset from HuggingFace
- Decrypt the encrypted content using the provided canary keys
- Save the dataset to `benchmarking_dataset/data/` in both Parquet and CSV formats
- Display sample data from the first row

**Output files:**
- `benchmarking_dataset/data/browsecomp_longcontext.parquet` - Main dataset (efficient format)
- `benchmarking_dataset/data/browsecomp_longcontext.csv` - Human-readable format

**Dataset Schema:**
- `problem`: The research question to answer
- `answer`: The expected answer
- `urls`: List of URLs to use for answering the question

### Step 2: Inspect the Dataset

You can inspect the dataset using Python:

```python
import pandas as pd

# Load the dataset
df = pd.read_parquet('benchmarking_dataset/data/browsecomp_longcontext.parquet')

# View dataset info
print(f"Number of samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# View a sample
print(df.iloc[0])
```

## Testing the Agent

### Step 1: Start the vLLM Server

The agent requires a running vLLM server with the model. Start the server using:

```bash
# Make sure your config.yaml is properly configured
# Then run the deployment script
python agent_deployment/deploy_vllm.py
```

This will:
- Pull the vLLM Docker image if not present
- Load the `openai/gpt-oss-20b` model
- Start the OpenAI-compatible API server on port 8000
- Run a test request to verify the server is ready

**Wait for the server to be ready** (this may take several minutes for model loading).

### Step 2: Test with BrowseComp Dataset

Create a test script to evaluate the agent on the BrowseComp dataset:

```python
# test_browsecomp.py
import pandas as pd
from agent_deployment.create_deep_agent import agent

# Load the dataset
df = pd.read_parquet('benchmarking_dataset/data/browsecomp_longcontext.parquet')

# Test on first sample
sample = df.iloc[0]
problem = sample['problem']
expected_answer = sample['answer']
urls = sample['urls']

print(f"Question: {problem}")
print(f"Expected Answer: {expected_answer}")
print(f"\nURLs to use:\n{urls}")
print("\n" + "="*80)
print("Running agent...")
print("="*80)

# Run the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": problem}]
})

# Extract the final answer
final_message = result["messages"][-1].content
print("\n" + "="*80)
print("Agent Response:")
print("="*80)
print(final_message)

# Check if format is correct
if "Final Answer:" in final_message:
    print("\n Response follows the required format!")
else:
    print("\n  Response does not follow the required format")
```

Run the test:
```bash
python test_browsecomp.py
```

### Step 3: Run Full Evaluation

To evaluate the agent on multiple samples from the dataset:

```python
# evaluate_agent.py
import pandas as pd
from agent_deployment.create_deep_agent import agent
import json

# Load dataset
df = pd.read_parquet('benchmarking_dataset/data/browsecomp_longcontext.parquet')

# Evaluate on first N samples
num_samples = 5
results = []

for idx in range(num_samples):
    sample = df.iloc[idx]
    problem = sample['problem']
    expected_answer = sample['answer']

    print(f"\n{'='*80}")
    print(f"Sample {idx+1}/{num_samples}")
    print(f"{'='*80}")
    print(f"Question: {problem[:200]}...")

    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": problem}]
        })

        response = result["messages"][-1].content

        # Check if format is correct
        follows_format = "Final Answer:" in response

        results.append({
            "sample_id": idx,
            "question": problem,
            "expected_answer": expected_answer,
            "agent_response": response,
            "follows_format": follows_format
        })

        print(f" Completed sample {idx+1}")

    except Exception as e:
        print(f"L Error on sample {idx+1}: {str(e)}")
        results.append({
            "sample_id": idx,
            "question": problem,
            "expected_answer": expected_answer,
            "agent_response": f"ERROR: {str(e)}",
            "follows_format": False
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('benchmarking_dataset/evaluation_results.csv', index=False)

# Print summary
format_compliance = sum(r['follows_format'] for r in results) / len(results) * 100
print(f"\n{'='*80}")
print("Evaluation Summary")
print(f"{'='*80}")
print(f"Total samples: {len(results)}")
print(f"Format compliance: {format_compliance:.1f}%")
print(f"Results saved to: benchmarking_dataset/evaluation_results.csv")
```

Run the evaluation:
```bash
python evaluate_agent.py
```

## Configuration

### vLLM Configuration (`config.yaml`)

Key parameters to adjust based on your setup:

```yaml
profiles:
  dev_profile:
    model_id: "openai/gpt-oss-20b"
    port: 8000
    engine:
      # Context window size - adjust based on dataset requirements
      max_model_len: 131072

      # GPU settings
      tensor_parallel_size: 1  # Number of GPUs
      gpu_memory_utilization: 0.92  # Fraction of GPU memory to use

      # Batching for throughput
      max_num_seqs: 256  # Max concurrent sequences
      max_num_batched_tokens: 8192  # Max tokens per batch

      # Precision settings
      dtype: auto  # bf16/fp16
      # kv_cache_dtype: "fp8"  # Uncomment for FP8 KV cache
```

### Agent Configuration

The agent is configured in `agent_deployment/create_deep_agent.py`:

- **Tools**: `internet_search` using Tavily API
- **Sub-agents**: Research agent and critique agent
- **Model**: OpenAI-compatible API (vLLM server)
- **Prompt**: Formatted for BrowseComp evaluation with strict "Final Answer:" format

## Expected Output Format

The agent is configured to provide answers in the following format:

```
Final Answer: [concise answer here]
```

This format is critical for the BrowseComp evaluation pipeline.

## Troubleshooting

### vLLM server not starting
- Check GPU availability: `nvidia-smi`
- Verify Docker is running: `docker ps`
- Check logs for memory issues (reduce `gpu_memory_utilization` or `max_model_len`)

### Agent not following format
- Ensure the prompt in `create_deep_agent.py` is correctly configured
- Check if the model supports instruction following
- Verify the model is loaded correctly (check vLLM logs)

### Dataset loading issues
- Ensure you have internet connection for HuggingFace downloads
- Install required packages: `pip install datasets pandas pyarrow`
- Check if the dataset has been updated (may need to clear cache)

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [BrowseCompLongContext Dataset](https://huggingface.co/datasets/openai/BrowseCompLongContext)
- [DeepAgents Framework](https://github.com/anthropics/deepagents)
