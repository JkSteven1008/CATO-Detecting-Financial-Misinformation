import json
import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from collections import deque, defaultdict


# ==========================================
# 辅助函数：鲁棒的JSON解析
# ==========================================
def robust_json_parse(llm_output):
    """
    (保持不变，用于解析 LLM 输出)
    """
    if not isinstance(llm_output, str):
        return llm_output
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        pattern = r"\{.*\}"
        match = re.search(pattern, llm_output, re.DOTALL)
        json_str = match.group(0) if match else llm_output
    try:
        return json.loads(json_str)
    except:
        try:
            import ast
            return ast.literal_eval(json_str)
        except:
            return None


# ==========================================
# 模块 3.1: 上下文感知检索 (修正版：加载预计算向量)
# ==========================================
class HybridRetriever:
    def __init__(self, corpus, client):
        self.corpus = corpus
        self.client = client
        self.emb_model = "text-embedding-v4"

        # 1. 初始化 BM25 (基于文本)
        print("Initializing BM25...")
        self.tokenized_corpus = [doc['content'].lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 2. 初始化 FAISS (直接加载预计算向量)
        # text-embedding-v4 输出维度为 1024
        self.dimension = 1024
        self.index = faiss.IndexFlatIP(self.dimension)

        # 定义关键词 (保持不变)
        self.event_keywords = {
            "RISK": ["loss", "default", "investigation", "plunge", "delisting", "fraud", "scandal"],
            "GROWTH": ["growth", "profit", "acquisition", "record high", "revenue", "merger"],
            "POLICY": ["rate cut", "regulation", "law", "fine", "sanction", "ban"]
        }

        self._load_precomputed_index()

    def _get_query_embedding(self, text):
        """
        只为查询(新新闻)调用 API 获取向量
        """
        text = text.replace("\n", " ")
        if not text.strip():
            return np.zeros(self.dimension)

        try:
            # 调用 DashScope API
            response = self.client.embeddings.create(
                model=self.emb_model,
                input=text,
                dimensions=1024
            )
            # 注意：DashScope 返回结构通常是 response.data[0].embedding
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Query Embedding Error: {e}")
            return np.zeros(self.dimension)

    def _load_precomputed_index(self):
        """
        [核心修正] 直接从 corpus 中读取 'embedding' 字段构建索引
        """
        print("Loading pre-computed embeddings from knowledge base...")
        embeddings = []
        valid_count = 0

        for doc in self.corpus:
            # 检查每条数据是否有 embedding 字段
            if 'embedding' in doc and doc['embedding']:
                emb = doc['embedding']
                # 确保是列表格式
                if isinstance(emb, list):
                    if len(emb) == self.dimension:
                        embeddings.append(emb)
                        valid_count += 1
                    else:
                        print(f"Warning: Dimension mismatch. Expected {self.dimension}, got {len(emb)}")
                        embeddings.append(np.zeros(self.dimension))  # 补零防止错位
                else:
                    embeddings.append(np.zeros(self.dimension))
            else:
                # 如果某条数据没有 embedding，补零占位，保持索引对齐
                embeddings.append(np.zeros(self.dimension))

        if valid_count == 0:
            print("ERROR: No valid embeddings found in corpus! Please check your jsonl file.")
            return

        # 转换为 float32 的 numpy 数组
        emb_matrix = np.array(embeddings).astype('float32')

        # 归一化 (如果使用的是 Cosine Similarity，建议归一化；IndexFlatIP 计算的是内积)
        # 通常 text-embedding-v4 输出已经归一化，但为了保险可以再做一次
        faiss.normalize_L2(emb_matrix)

        self.index.add(emb_matrix)
        print(f"Index built successfully with {self.index.ntotal} documents. (Valid embeddings: {valid_count})")

    def _extract_structure_features(self, text):
        """提取简单的实体-事件特征 (保持不变)"""
        text_lower = text.lower()
        entity_pattern = r"([A-Z][a-zA-Z0-9]*\s+(?:Inc\.?|Corp\.?|Ltd\.?|Group|Bank))"
        entities = set(re.findall(entity_pattern, text))
        events = set()
        for category, keywords in self.event_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    events.add(kw)
        return entities, events

    def _calculate_structure_score(self, query_triplet, doc_content):
        """计算结构化匹配分数 (保持不变)"""
        q_entities, q_events = query_triplet
        doc_entities, doc_events = self._extract_structure_features(doc_content)
        score = 0.0
        if q_entities:
            for q_ent in q_entities:
                for d_ent in doc_entities:
                    if q_ent in d_ent or d_ent in q_ent:
                        score += 2.0;
                        break
        if q_events:
            score += len(q_events.intersection(doc_events)) * 1.0
        return score

    def search(self, query, top_k=3, alpha=0.4, beta=0.3):
        """
        混合检索：API生成Query向量 -> FAISS匹配 -> BM25 -> 结构化 -> 融合
        """
        # 1. 向量检索 (Vector)
        query_emb = self._get_query_embedding(query).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)  # 确保查询向量也归一化

        vec_scores, vec_indices = self.index.search(query_emb, len(self.corpus))
        vec_scores = vec_scores[0]
        # 归一化到 0-1
        vec_scores = np.clip(vec_scores, 0, 1)

        # 2. 关键词检索 (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        if np.max(bm25_scores) > 0:
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

        # 3. 结构化检索 (Structure)
        q_triplet = self._extract_structure_features(query)
        # 这里为了性能，只对 top-N 的候选做结构化重排，或者对全量做（全量慢）
        # MVP 阶段建议全量或取 vec_indices 前100个
        struct_scores = np.zeros(len(self.corpus))
        # 简化：仅计算 Vector Top 50 的结构化分数以加速
        candidate_indices = vec_indices[0][:50]
        for idx in candidate_indices:
            struct_scores[idx] = self._calculate_structure_score(q_triplet, self.corpus[idx]['content'])

        if np.max(struct_scores) > 0:
            struct_scores = struct_scores / np.max(struct_scores)

        # 4. 融合分数 (Hybrid Fusion)
        # 还原向量分数顺序到全局
        full_vec_scores = np.zeros(len(self.corpus))
        for score, idx in zip(vec_scores, vec_indices[0]):
            if idx < len(full_vec_scores):
                full_vec_scores[idx] = score

        final_scores = (alpha * bm25_scores) + \
                       (beta * struct_scores) + \
                       ((1 - alpha - beta) * full_vec_scores)

        # 5. 返回 Top-K
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            res = self.corpus[idx].copy()
            res['score'] = final_scores[idx]
            # 移除 embedding 字段，避免打印时刷屏
            if 'embedding' in res:
                del res['embedding']
            results.append(res)

        return results


# ==========================================
# 模块 3.2: 元认知规划器 (Meta-Cognitive Planner)
# ==========================================
class MetaCognitivePlanner:
    def __init__(self, client, model="qwen-max"): # 修改默认值为 qwen-max
        self.client = client
        self.model = model
        # 工具集合保持不变，因为这是系统内部标识符
        self.valid_tools = {
            "CGT", "SCP", "SCA", "PID", "FCV", "TLV", "RMD", "EVA", "CPE"
        }

    def _validate_dag(self, plan_json):
        """
        [升级] 校验层：确保生成的计划是有向无环图 (DAG) 且工具名合法。
        增加：自动将工具全名映射为缩写。
        """
        if not plan_json or 'tools' not in plan_json:
            return False, "Missing 'tools' field", plan_json

        tools = plan_json.get('tools', [])
        dependencies = plan_json.get('dependencies', [])

        # === 新增：工具名称清洗映射表 ===
        tool_mapping = {
            "Contextual Grounding Tool": "CGT",
            "Source Credibility Propagator": "SCP",
            "Semantic Coherence Analyzer": "SCA",
            "Pragmatic Intent Decoder": "PID",
            "Factual Consistency Verifier": "FCV",
            "Temporal Logic Validator": "TLV",
            "Rhetorical Manipulation Detector": "RMD",
            "Expectation Violation Analyzer": "EVA",
            "Contradiction Propagation Engine": "CPE"
        }

        cleaned_tools = []
        for t in tools:
            # 1. 如果是标准缩写，直接保留
            if t in self.valid_tools:
                cleaned_tools.append(t)
            # 2. 如果是全名，尝试映射
            elif t in tool_mapping:
                # print(f"Auto-correcting tool name: '{t}' -> '{tool_mapping[t]}'")
                cleaned_tools.append(tool_mapping[t])
            # 3. 如果是部分匹配（比如 LLM 输出 'SCP Tool'）
            else:
                found = False
                for full_name, abbr in tool_mapping.items():
                    if full_name in t:  # 模糊匹配全名
                        cleaned_tools.append(abbr)
                        found = True
                        break
                if not found:
                    print(f"Warning: Unknown tool '{t}' generated. Removing from plan.")

        # 更新工具列表
        tools = list(set(cleaned_tools))  # 去重

        # 同样清洗依赖关系中的名字
        cleaned_deps = []
        for upstream, downstream in dependencies:
            u = tool_mapping.get(upstream, upstream)
            v = tool_mapping.get(downstream, downstream)
            if u in tools and v in tools:
                cleaned_deps.append([u, v])

        # 2. 构建图与入度表 (保持不变)
        graph = defaultdict(list)
        in_degree = {t: 0 for t in tools}

        for upstream, downstream in cleaned_deps:
            graph[upstream].append(downstream)
            in_degree[downstream] += 1

        # 3. 拓扑排序 (保持不变)
        queue = deque([t for t in tools if in_degree[t] == 0])
        sorted_plan = []

        while queue:
            node = queue.popleft()
            sorted_plan.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 4. 环检测
        if len(sorted_plan) != len(tools):
            # 如果有环，尝试简单的降级策略：只返回工具列表，放弃依赖顺序
            print("Cycle detected. Fallback to unordered execution.")
            return True, tools, plan_json

        return True, sorted_plan, plan_json

    def generate_plan(self, text, context_results):
        """
        生成工具调用计划 (English Prompts)
        """
        context_str = "\n".join(
            [f"- [{doc.get('source', 'Unknown')}] {doc['content']} (Score: {doc['score']:.2f})" for doc in
             context_results])

        # 这里全部改为英文 Prompt
        prompt = f"""
        TASK: You are a Meta-Cognitive Planner for a financial misinformation detection system. 
        Your goal is to formulate a dynamic verification plan for the target text based on the retrieved context.

        TARGET TEXT (T):
        "{text}"

        RETRIEVED CONTEXT (C):
        {context_str}

        AVAILABLE TOOLSET:
        1. Contextual Grounding Tool (CGT): Identifies missing context or background information.
        2. Source Credibility Propagator (SCP): Analyzes the credibility of the information source.
        3. Semantic Coherence Analyzer (SCA): Detects internal logical contradictions within the text.
        4. Pragmatic Intent Decoder (PID): Identifies emotional manipulation or rhetorical strategies.
        5. Factual Consistency Verifier (FCV): Verifies factual claims against the retrieved context.
        6. Temporal Logic Validator (TLV): Checks for chronological inconsistencies (e.g., event timing vs. report date).
        7. Rhetorical Manipulation Detector (RMD): Detects specific financial hype keywords or "trap" phrases.
        8. Expectation Violation Analyzer (EVA): Detects claims that violate common market knowledge or common sense.
        9. Contradiction Propagation Engine (CPE): Aggregates evidence and propagates contradictions (usually the final step).

        INSTRUCTIONS:
        1. Analyze the text for potential misinformation risks (e.g., exaggerated numbers, timing conflicts, suspicious sources).
        2. Select the most appropriate SUBSET of tools to verify these specific risks.
        3. Define the DEPENDENCIES between tools. (e.g., verify source credibility (SCP) before checking facts (FCV)).
        4. OUTPUT FORMAT: strictly JSON. Do not include markdown formatting or code blocks.

        JSON OUTPUT STRUCTURE:
        {{
            "reasoning": "Brief analysis of why these tools are selected...",
            "tools": ["Tool_A", "Tool_B", "Tool_C"],
            "dependencies": [["Tool_A", "Tool_B"], ["Tool_B", "Tool_C"]] 
        }}
        Note: dependencies [["A", "B"]] means A executes BEFORE B.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a rigorous financial risk assessment expert. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        raw_content = response.choices[0].message.content

        # 1. 鲁棒解析
        plan_json = robust_json_parse(raw_content)

        if not plan_json:
            print("Failed to parse LLM output. Returning fallback plan.")
            return ["CGT", "SCP", "FCV"]

            # 2. DAG 校验
        is_valid, result, _ = self._validate_dag(plan_json)

        if is_valid:
            print(f"Plan generated successfully: {result}")
            return result
        else:
            print(f"Plan validation failed ({result}). Executing fallback linear plan.")
            # 降级策略：如果 DAG 错误，按列表顺序尝试执行
            return list(plan_json.get('tools', []))