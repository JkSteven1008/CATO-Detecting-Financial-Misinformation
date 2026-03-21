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

# === 引入 CATO 模块 ===
from Retrieval_Planner import HybridRetriever, MetaCognitivePlanner
from Toolset import FinancialFactCheckingTools
from Router_1 import DynamicDAGScheduler
from Fusion_Engine import DualStageFusionEngine

# ==========================================
# 1. 配置区域 (User Configuration)
# ==========================================

# 【优化 1】：在这里填入你的 8 个 API Keys
API_KEYS = [
    "sk-50faa0a25abf4340b3398af9ffea6168",
    "sk-1cb274266c2341d3be77f5b8fb0cea25",
    "sk-3e329ef6c7d24104ac8d1470418a970e",
    "sk-09964cbd8e0446879cac5bac49a87aad",
    "sk-a5866e555f2448b3a166ba2c7b252703",
    "sk-5449638cc6614016b9964f8d961294ee",
    "sk-81632092914e4b189ce3c20729072a83",
    "sk-f3d2b0beefc4450db3e58f7c2b1c2237",
    "sk-7de7cf625ac84827ba850c266b09edf4",
    "sk-6234f2144f4946fa81cbfaf6e382c3a0"
]
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 路径配置
KNOWLEDGE_BASE_PATH = "../data/DAG/Final_Combined_Data.jsonl"
FINFACT_PATH = "../data/Split_data/val/finfact_val.json"
FINGUARD_FAKE_PATH = "../data/Split_data/val/Finance_FAKE_val.csv"
FINGUARD_TRUE_PATH = "../data/Split_data/val/Finance_TRUE_val.csv"
OUTPUT_FILE = "results/evaluation_results.json"

# 测试参数
TEST_PARAMS = {
    "FinFact": 50,
    "FinGuard_FAKE": 0,
    "FinGuard_TRUE": 0
}

# 并发配置：外层同时处理的新闻条数（建议设置为 API Key 数量的 1~1.5 倍）
MAX_CONCURRENT_ITEMS = 10

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ==========================================
# 2. 核心优化：多 Key 轮询代理客户端
# ==========================================
class MultiKeyClient:
    """
    一个无缝伪装成 OpenAI Client 的代理类。
    每次调用 .chat 或 .embeddings 时，随机分配一个 API Key，完美负载均衡。
    """

    def __init__(self, api_keys, base_url):
        # 过滤掉空的或者默认的无效 key
        valid_keys = [k for k in api_keys if k and k != "sk-your-key-2"]
        if not valid_keys:
            raise ValueError("No valid API Keys provided!")
        logger.info(f"Initialized MultiKeyClient with {len(valid_keys)} API Keys for Load Balancing.")
        self.clients = [OpenAI(api_key=key, base_url=base_url) for key in valid_keys]

    @property
    def chat(self):
        # 随机挑选一个客户端，打散请求，绕过并发限制
        return random.choice(self.clients).chat

    @property
    def embeddings(self):
        return random.choice(self.clients).embeddings


# ==========================================
# 3. 系统初始化 (System Init)
# ==========================================

class CATOSystem:
    def __init__(self):
        logger.info("Initializing CATO System...")

        # 【优化 2】：使用多 Key 代理客户端代替单一客户端
        self.client = MultiKeyClient(api_keys=API_KEYS, base_url=BASE_URL)

        # 只加载一次庞大的语料库和 FAISS 索引，节省内存
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
# 数据加载与映射逻辑 (保持不变)
# ==========================================

def load_finfact_data(limit):
    data = []
    if not os.path.exists(FINFACT_PATH): return data
    try:
        with open(FINFACT_PATH, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        count = 0
        for item in full_data:
            # 满足设定的 limit 数量后退出
            if limit != -1 and count >= limit:
                break

            text = item.get('claim') or item.get('text')
            raw_label = str(item.get('label', '')).lower()

            # 【修改点 1】：跳过 NEI 或其他非明确真假的数据
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
# 4. 主程序：引入并发处理
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

    # 包装执行函数，便于扔进线程池
    def process_item(item):
        text = item['text']
        true_label = item['label']
        source = item['source']
        report = cato.run_single_inference(text, source)
        return item, report

    # 【优化 3】：使用 ThreadPoolExecutor 进行数据级并发
    # MAX_CONCURRENT_ITEMS 控制同时处理多少条新闻
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ITEMS) as executor:
        # 提交所有任务
        futures = {executor.submit(process_item, item): item for item in all_data}

        # 使用 tqdm 监控并发进度
        for future in tqdm(as_completed(futures), total=len(all_data), desc="Processing News"):
            item, report = future.result()
            source = item['source']
            text = item['text']
            true_label = item['label']

            # 结果分类与指标收集
            if source == "FinFact":
                # 【修改点 2】：将 dataset_type 从 "3_class" 改为 "2_class"
                # 这样即使模型输出了奇怪的内容，也会被安全映射为二分类
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

    # 计算指标并保存
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