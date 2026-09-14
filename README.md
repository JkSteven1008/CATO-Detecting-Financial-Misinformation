### *CATO: Context-Aware Tool Orchestration for Detecting Financial Misinformation via Evidence-Propagation DAGs*

**CATO** (Context-Aware Tool Orchestrator) is an agentic framework for detecting financial misinformation. It dynamically composes analytical pathways with a metacognitive planner, executes a task-specific **Directed Acyclic Graph (DAG)** of specialized verification tools, and fuses multi-source evidence under uncertainty to produce an auditable verdict.

-----

## Key Features

  * **Metacognitive Planning**: Instead of a fixed linear pipeline, CATO uses an LLM-based planner to generate a task-specific **Directed Acyclic Graph (DAG)** of tools based on the claim's context.
  * **Dynamic DAG Scheduling**: A robust scheduler that handles tool dependencies, parallel layer-wise execution, and a "circuit breaker" mechanism (e.g., skipping factual verification when the source is deemed fundamentally untrustworthy).
  * **Evidence Propagation (Blackboard Pattern)**: Tools share a global "blackboard" state, allowing downstream tools to reason based on the findings of upstream tools.
  * **Specialized Financial Toolset**: 9 purpose-built tools covering contextual grounding, source credibility, semantics and pragmatics, fact and time verification, and higher-order reasoning.
  * **Context-Injected Fusion**: A dual-stage synthesis engine that compresses raw tool evidence and applies a dataset-specific golden prompt to produce a final Real/Fake verdict with an auditable explanation path.

-----

## Architecture

CATO is built on a modular, four-stage pipeline:

1.  **Retrieval Phase**: `HybridRetriever` combines BM25 keyword search with FAISS vector search (precomputed 1024-dimensional embeddings) to provide the necessary market context and policy background.
2.  **Planning Phase**: `MetaCognitivePlanner` analyzes the claim and generates a dependency-aware execution plan in JSON.
3.  **Execution Phase**: `DynamicDAGScheduler` executes the tools layer by layer:
      * **L1 (Grounding)**: CGT, SCP.
      * **L2 (Semantic & Pragmatic)**: SCA, PID.
      * **L3 (Fact & Time)**: FCV, TLV, RMD.
      * **L4 (Higher-order Reasoning)**: EVA, CPE.
4.  **Fusion Phase**: `DualStageFusionEngine` compresses the blackboard evidence and synthesizes the final verdict.

-----

## Repository Structure

| File                   | Description                                                  |
| :--------------------- | :----------------------------------------------------------- |
| `main.py`              | **Core entry point**. Manages dataset loading, multi-threaded inference, and metric calculation. |
| `Retrieval_Planner.py` | Contains the `HybridRetriever` and the LLM-based `MetaCognitivePlanner` for DAG generation. |
| `Router_1.py`          | The `DynamicDAGScheduler`. Implements topological layer execution and the circuit-breaker mechanism. |
| `Toolset.py`           | Implementation of the 9 analytical tools.                    |
| `Fusion_Engine.py`     | The `DualStageFusionEngine`. Merges multi-source evidence for the final verdict. |

-----

## The CATO Toolset (9 Specialized Tools)

  * **CGT**: Contextual Grounding Tool — identifies missing context or background information.
  * **SCP**: Source Credibility Propagator — analyzes the credibility of the information source.
  * **SCA**: Semantic Coherence Analyzer — detects internal logical contradictions within the text.
  * **PID**: Pragmatic Intent Decoder — identifies emotional manipulation or rhetorical strategies.
  * **FCV**: Factual Consistency Verifier — verifies factual claims against the retrieved context.
  * **TLV**: Temporal Logic Validator — checks for chronological inconsistencies (e.g., event timing vs. report date).
  * **RMD**: Rhetorical Manipulation Detector — detects financial hype keywords or "trap" phrases.
  * **EVA**: Expectation Violation Analyzer — detects claims that violate common market knowledge or common sense.
  * **CPE**: Contradiction Propagation Engine — aggregates evidence and propagates contradictions (usually the final step).

-----

## Getting Started

### Prerequisites

  * Python 3.9+
  * DashScope (Alibaba Cloud) API keys with access to `qwen-max` and `text-embedding-v4`

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

In `main.py`, fill in your API keys in the `API_KEYS` list. CATO rotates keys automatically for high-throughput batch processing. `BASE_URL` points to the DashScope OpenAI-compatible endpoint.

```python
API_KEYS = ["sk-your-api-key-1", "sk-your-api-key-2"]
```

### Data Layout

The default paths in `main.py` expect the following layout:

```
data/
├── DAG/Final_Combined_Data.jsonl     # knowledge base; each line needs "content" and a precomputed 1024-d "embedding"
└── Split_data/val/
    ├── finfact_val.json              # FinFact validation set
    ├── Finance_FAKE_val.csv          # FinGuard fake samples
    └── Finance_TRUE_val.csv          # FinGuard true samples
```

The retriever loads corpus embeddings directly from the JSONL `embedding` field; only queries (new news items) call the embedding API.

### Running the Evaluation

```bash
python main.py
```

Results are saved to `results/evaluation_results.json`. FinFact NEI samples are filtered out, and all predictions are mapped to the binary Real/Fake setting. `TEST_PARAMS` and `MAX_CONCURRENT_ITEMS` control evaluation size and concurrency.

-----

## Citation

If you find this work helpful in your research, please cite:

```bibtex
@article{cato2026,
  title={CATO: Context-Aware Tool Orchestration for Detecting Financial Misinformation via Evidence-Propagation DAGs},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
```
