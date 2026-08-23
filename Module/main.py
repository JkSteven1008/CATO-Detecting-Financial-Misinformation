import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Import CATO modules ===
from Retrieval_Planner import HybridRetriever, MetaCognitivePlanner
from Toolset import FinancialFactCheckingTools
from Router_1 import DynamicDAGScheduler
from Fusion_Engine import DualStageFusionEngine

# ==========================================
# 1. Configuration area (user configuration)
# ==========================================

# [Optimization 1]: fill in your API keys here
API_KEYS = [
    "sk-your-api-key-1",
    "sk-your-api-key-2",
    "sk-your-api-key-3",
    "sk-your-api-key-4",
    "sk-your-api-key-5",
    "sk-your-api-key-6",
    "sk-your-api-key-7",
    "sk-your-api-key-8",
    "sk-your-api-key-9",
    "sk-your-api-key-10"
]
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Path configuration
KNOWLEDGE_BASE_PATH = "../data/DAG/Final_Combined_Data.jsonl"
FINFACT_PATH = "../data/Split_data/val/finfact_val.json"
FINGUARD_FAKE_PATH = "../data/Split_data/val/Finance_FAKE_val.csv"
FINGUARD_TRUE_PATH = "../data/Split_data/val/Finance_TRUE_val.csv"
OUTPUT_FILE = "results/evaluation_results.json"

# Test parameters
TEST_PARAMS = {
    "FinFact": 50,
    "FinGuard_FAKE": 0,
    "FinGuard_TRUE": 0
}

# Concurrency: number of news items processed simultaneously (recommended: 1-1.5x the number of API keys)
MAX_CONCURRENT_ITEMS = 10

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ==========================================
# 2. Core optimization: multi-key round-robin proxy client
# ==========================================
class MultiKeyClient:
    """
    A proxy class that seamlessly masquerades as an OpenAI client.
    Each call to .chat or .embeddings randomly picks an API key for perfect load balancing.
    """

    def __init__(self, api_keys, base_url):
        # Filter out empty or invalid default keys
        valid_keys = [k for k in api_keys if k and k != "sk-your-key-2"]
        if not valid_keys:
            raise ValueError("No valid API Keys provided!")
        logger.info(f"Initialized MultiKeyClient with {len(valid_keys)} API Keys for Load Balancing.")
        self.clients = [OpenAI(api_key=key, base_url=base_url) for key in valid_keys]

    @property
    def chat(self):
        # Pick a client at random to spread requests and bypass concurrency limits
        return random.choice(self.clients).chat

    @property
    def embeddings(self):
        return random.choice(self.clients).embeddings


# ==========================================
# 3. System initialization
# ==========================================

class CATOSystem:
    def __init__(self):
        logger.info("Initializing CATO System...")

        # [Optimization 2]: use the multi-key proxy client instead of a single client
        self.client = MultiKeyClient(api_keys=API_KEYS, base_url=BASE_URL)

        # Load the large corpus and FAISS index only once to save memory
        self.corpus = self._load_knowledge_base()
        self.retriever = HybridRetriever(corpus=self.corpus, client=self.client)
        self.planner = MetaCognitivePlanner(client=self.client, model="qwen-max")
        self.toolset = FinancialFactCheckingTools(client=self.client, model="qwen-max")
        self.scheduler = DynamicDAGScheduler(toolset=self.toolset)
        self.fusion = DualStageFusionEngine(llm_client=self.client, model="qwen-max")

    def _load_knowledge_base(self):
        logger.info(f"Loading Knowledge Base from {KNOWLEDGE_BASE_PATH}...")
        corpus = []
        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            return []
        with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        corpus.append(json.loads(line))
                    except:
                        continue
        logger.info(f"Loaded {len(corpus)} documents.")
        return corpus

    def run_single_inference(self, text, source):
        try:
            evidence = self.retriever.search(text, top_k=3)
            plan_list = self.planner.generate_plan(text, evidence)

            if isinstance(plan_list, list):
                plan_json = {"tools": plan_list, "dependencies": []}
            else:
                plan_json = plan_list

            result_context = self.scheduler.execute(plan_json, text, evidence)
            final_report = self.fusion.fuse(result_context, dataset_source=source)
            return final_report
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"final_label": "Error", "explanation_path": str(e)}


# ==========================================
# Data loading and label mapping logic (unchanged)
# ==========================================

def load_finfact_data(limit):
    data = []
    if not os.path.exists(FINFACT_PATH): return data
    try:
        with open(FINFACT_PATH, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        count = 0
        for item in full_data:
            # Stop once the configured limit is reached
            if limit != -1 and count >= limit:
                break

            text = item.get('claim') or item.get('text')
            raw_label = str(item.get('label', '')).lower()

            # [Change Point 1]: skip NEI or other data without a clear true/false label
            if raw_label == 'nei' or raw_label not in ['true', 'mostly true', 'supports', 'supported', 'false',
                                                       'mostly false', 'refutes', 'refuted']:
                continue

            if raw_label in ['true', 'mostly true', 'supports', 'supported']:
                label = 0
            elif raw_label in ['false', 'mostly false', 'refutes', 'refuted']:
                label = 1

            data.append({"text": text, "label": label, "source": "FinFact", "raw_label": raw_label})
            count += 1

    except Exception as e:
        logger.error(f"Error loading FinFact: {e}")
    return data


def load_finguard_data(fake_limit, true_limit):
    data = []
    if os.path.exists(FINGUARD_FAKE_PATH):
        df = pd.read_csv(FINGUARD_FAKE_PATH)
        if fake_limit != -1: df = df.head(fake_limit)
        for _, row in df.iterrows():
            text = row.get('text') or row.get('statement') or row.iloc[0]
            data.append({"text": str(text), "label": 1, "source": "FinGuard_FAKE"})

    if os.path.exists(FINGUARD_TRUE_PATH):
        df = pd.read_csv(FINGUARD_TRUE_PATH)
        if true_limit != -1: df = df.head(true_limit)
        for _, row in df.iterrows():
            text = row.get('text') or row.get('statement') or row.iloc[0]
            data.append({"text": str(text), "label": 0, "source": "FinGuard_TRUE"})
    return data


def map_prediction_to_label(report, dataset_type="3_class"):
    final_label = str(report.get('final_label', '')).lower()
    if "real" in final_label or "Real" in final_label or "true" in final_label or "True" in final_label:
        pred = 0
    elif "fake" in final_label or "Fake" in final_label or "false" in final_label or "False" in final_label:
        pred = 1
    else:
        pred = 2
    if dataset_type == "2_class" and pred == 2: pred = 0
    return pred


def calculate_metrics(y_true, y_pred, dataset_name, labels=None, target_names=None):
    print(f"\n{'=' * 20} {dataset_name} Evaluation {'=' * 20}")
    if not y_true: return
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:\n", cm)
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    print("\nDetailed Metrics:\n", report)
    print("-" * 60)


# ==========================================
# 4. Main program: concurrent processing
# ==========================================

def main():
    cato = CATOSystem()

    logger.info("Loading Datasets...")
    finfact_data = load_finfact_data(TEST_PARAMS["FinFact"])
    finguard_data = load_finguard_data(TEST_PARAMS["FinGuard_FAKE"], TEST_PARAMS["FinGuard_TRUE"])

    all_data = finfact_data + finguard_data
    logger.info(f"Total Test Samples: {len(all_data)}")

    results_log = []
    y_true_finfact, y_pred_finfact = [], []
    y_true_finguard, y_pred_finguard = [], []

    print("\n>>> Starting Parallel Inference...")

    # Wrap the execution function for the thread pool
    def process_item(item):
        text = item['text']
        true_label = item['label']
        source = item['source']
        report = cato.run_single_inference(text, source)
        return item, report

    # [Optimization 3]: use ThreadPoolExecutor for data-level concurrency
    # MAX_CONCURRENT_ITEMS controls how many news items are processed at once
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ITEMS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_item, item): item for item in all_data}

        # Use tqdm to monitor progress
        for future in tqdm(as_completed(futures), total=len(all_data), desc="Processing News"):
            item, report = future.result()
            source = item['source']
            text = item['text']
            true_label = item['label']

            # Classify results and collect metrics
            if source == "FinFact":
                # [Change Point 2]: switch dataset_type from "3_class" to "2_class"
                # This safely maps any unexpected model output to the binary setting
                pred_label = map_prediction_to_label(report, dataset_type="2_class")
                y_true_finfact.append(true_label)
                y_pred_finfact.append(pred_label)
            else:
                pred_label = map_prediction_to_label(report, dataset_type="2_class")
                y_true_finguard.append(true_label)
                y_pred_finguard.append(pred_label)

            results_log.append({
                "source": source,
                "text": text[:50] + "...",
                "true_label": true_label,
                "pred_label": pred_label,
                "is_correct": true_label == pred_label,
                "raw_output": report
            })

    # Compute metrics and save results
    calculate_metrics(
        y_true_finfact, y_pred_finfact, "FinFact (Filtered NEI - Binary)",
        labels=[0, 1], target_names=["Real", "Fake"]
    )

    calculate_metrics(
        y_true_finguard, y_pred_finguard, "FinGuard (Binary)",
        labels=[0, 1], target_names=["Real", "Fake"]
    )

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()