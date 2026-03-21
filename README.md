# CATO: Context-Aware Tool Orchestrator

### *Adaptive Financial Misinformation Detection via Evidence Propagation DAG*

[](https://www.google.com/search?q=https://www.python.org/downloads/)
[](https://www.google.com/search?q=https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=%23)

**CATO** (Context-Aware Tool Orchestrator) is a sophisticated agentic framework designed to tackle the unique challenges of financial misinformation—such as semantic subtlety, knowledge dynamism, and rhetorical manipulation. By dynamically orchestrating a specialized suite of 10 tools through a **Metacognitive Planner**, CATO moves beyond static RAG pipelines to provide verifiable, multi-granular evidence for every verdict.

-----

## 🚀 Key Features

  * **Metacognitive Planning**: Instead of a fixed linear pipeline, CATO uses an LLM-based planner to generate a task-specific **Directed Acyclic Graph (DAG)** of tools based on the claim's context.
  * **Dynamic DAG Scheduling**: A robust scheduler that handles tool dependencies, parallel execution, and "Circuit Breaker" logic (e.g., skipping factual verification if a source is deemed fundamentally untrustworthy).
  * **Evidence Propagation (Blackboard Pattern)**: Tools share a global "Blackboard" state, allowing downstream tools to reason based on findings from upstream tools (e.g., using retrieved market data to check a specific revenue claim).
  * **Specialized Financial Toolset**: 10 purpose-built tools covering source credibility, sentiment-fact decoupling, timeline consistency, and global common-sense reasoning.
  * **Context-Injected Fusion**: A dual-stage synthesis engine that integrates raw evidence, tool outputs, and semantic insights to generate a calibrated veracity score.

-----

## 🏗️ Architecture

CATO is built on a modular, four-stage pipeline:

1.  **Retrieval Phase**: Hybrid retrieval (BM25 + Vector Search) provides the necessary market context and policy background.
2.  **Planning Phase**: The `MetaCognitivePlanner` analyzes the claim and generates a dependency-aware execution plan.
3.  **Execution Phase**: The `DynamicDAGScheduler` executes tools across four layers:
      * **L1 (Foundation)**: Claim Generation & Source Profiling.
      * **L2 (Semantic)**: Deception Pattern Detection & Sentiment Analysis.
      * **L3 (Verification)**: Factual Cross-checking & Timeline Validation.
      * **L4 (Reasoning)**: Global Evidence Synthesis.
4.  **Fusion Phase**: The `DualStageFusionEngine` synthesizes the "Blackboard" findings into a final report.

-----

## 📁 Repository Structure

| File | Description |
| :--- | :--- |
| `main.py` | **Core Entry Point**. Manages dataset loading, multi-threaded inference, and performance metrics calculation. |
| `Retrieval_Planner.py` | Contains the `HybridRetriever` and the LLM-based `MetaCognitivePlanner` for DAG generation. |
| `Router_1.py` | The `DynamicDAGScheduler`. Implements topological sorting and the **Circuit Breaker** mechanism. |
| `Toolset.py` | Implementation of the **10 analytical tools**. Each tool is optimized for a specific dimension of financial analysis. |
| `Fusion_Engine.py` | The `DualStageFusionEngine`. Merges multi-source evidence and performs high-level reasoning for the final verdict. |

-----

## 🛠️ The CATO Toolset (10 Specialized Tools)

CATO utilizes a hierarchical toolset to dismantle complex financial claims:

  * **CGT**: Core Grounding Tool (Retrieval analysis).
  * **SCP**: Source Credibility Profiler.
  * **SCA**: Semantic Conflict Analyzer.
  * **PID**: Persuasion Intent Detector.
  * **FCV**: Factual Cross-Verifier.
  * **TLV**: Temporal Logic Validator.
  * **RMD**: Raw Market Data checker.
  * **EVA**: Global Evidence/Common-Sense Evaluator.
  * **COM**: Contextual Orthogonality Monitor.
  * **CPE**: Contradiction Propagation Engine (Final tool-level synthesis).

-----

## 🚦 Getting Started

### Prerequisites

  * Python 3.9+
  * LLM API Keys (e.g., Qwen-Max, GPT-4)

### Installation

```bash
git clone https://github.com/your-username/CATO.git
cd CATO
pip install -r requirements.txt
```

### Configuration

In `main.py`, configure your API keys in the `API_KEYS` list. CATO supports rotating keys for high-throughput batch processing.

```python
API_KEYS = ["your-key-1", "your-key-2"]
```

### Running the Evaluation

To execute CATO on the FinFact or FinGuard benchmarks:

```bash
python main.py
```

-----

## 📈 Performance

CATO achieves state-of-the-art results on financial truth-discovery benchmarks:

  * **FinFact**: +11.9% F1-score improvement over standard RAG baselines.
  * **Interpretability**: Provides a full "Trace Report" showing which tools were triggered and why.

-----

## 📝 Citation

If you find this work helpful in your research, please cite:

```bibtex
@article{cato2024,
  title={CATO: Context-Aware Tool Orchestration for Financial Misinformation Detection via Evidence Propagation DAG},
  author={Your Name/Team},
  journal={arXiv preprint},
  year={2024}
}
```

-----

**Would you like me to create a `requirements.txt` file based on your imports, or help you write the specific docstrings for any of these modules?**
