import json
import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from collections import deque, defaultdict


# ==========================================
# Helper function: robust JSON parsing
# ==========================================
def robust_json_parse(llm_output):
    """
    (unchanged; used to parse LLM output)
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
# Module 3.1: Context-Aware Retrieval (revised: loads pre-computed embeddings)
# ==========================================
class HybridRetriever:
    def __init__(self, corpus, client):
        self.corpus = corpus
        self.client = client
        self.emb_model = "text-embedding-v4"

        # 1. Initialize BM25 (text-based)
        print("Initializing BM25...")
        self.tokenized_corpus = [doc['content'].lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 2. Initialize FAISS (directly load pre-computed embeddings)
        # text-embedding-v4 outputs 1024-dimensional embeddings
        self.dimension = 1024
        self.index = faiss.IndexFlatIP(self.dimension)

        # Define keywords (unchanged)
        self.event_keywords = {
            "RISK": ["loss", "default", "investigation", "plunge", "delisting", "fraud", "scandal"],
            "GROWTH": ["growth", "profit", "acquisition", "record high", "revenue", "merger"],
            "POLICY": ["rate cut", "regulation", "law", "fine", "sanction", "ban"]
        }

        self._load_precomputed_index()

    def _get_query_embedding(self, text):
        """
        Call the API to obtain embeddings only for the query (new news)
        """
        text = text.replace("\n", " ")
        if not text.strip():
            return np.zeros(self.dimension)

        try:
            # Call the DashScope API
            response = self.client.embeddings.create(
                model=self.emb_model,
                input=text,
                dimensions=1024
            )
            # Note: DashScope typically returns response.data[0].embedding
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Query Embedding Error: {e}")
            return np.zeros(self.dimension)

    def _load_precomputed_index(self):
        """
        [Core Fix] Build the index directly from the 'embedding' field in the corpus
        """
        print("Loading pre-computed embeddings from knowledge base...")
        embeddings = []
        valid_count = 0

        for doc in self.corpus:
            # Check whether each entry has an embedding field
            if 'embedding' in doc and doc['embedding']:
                emb = doc['embedding']
                # Ensure it is in list format
                if isinstance(emb, list):
                    if len(emb) == self.dimension:
                        embeddings.append(emb)
                        valid_count += 1
                    else:
                        print(f"Warning: Dimension mismatch. Expected {self.dimension}, got {len(emb)}")
                        embeddings.append(np.zeros(self.dimension))  # pad with zeros to avoid misalignment
                else:
                    embeddings.append(np.zeros(self.dimension))
            else:
                # If an entry has no embedding, pad with zeros to keep the index aligned
                embeddings.append(np.zeros(self.dimension))

        if valid_count == 0:
            print("ERROR: No valid embeddings found in corpus! Please check your jsonl file.")
            return

        # Convert to a float32 numpy array
        emb_matrix = np.array(embeddings).astype('float32')

        # Normalize (recommended if using Cosine Similarity; IndexFlatIP computes inner products)
        # text-embedding-v4 outputs are usually normalized already, but normalize again to be safe
        faiss.normalize_L2(emb_matrix)

        self.index.add(emb_matrix)
        print(f"Index built successfully with {self.index.ntotal} documents. (Valid embeddings: {valid_count})")

    def _extract_structure_features(self, text):
        """Extract simple entity-event features (unchanged)"""
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
        """Compute the structured matching score (unchanged)"""
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
        Hybrid retrieval: API-generated query embedding -> FAISS matching -> BM25 -> structure -> fusion
        """
        # 1. Vector retrieval
        query_emb = self._get_query_embedding(query).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)  # ensure the query embedding is also normalized

        vec_scores, vec_indices = self.index.search(query_emb, len(self.corpus))
        vec_scores = vec_scores[0]
        # Normalize to 0-1
        vec_scores = np.clip(vec_scores, 0, 1)

        # 2. Keyword retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        if np.max(bm25_scores) > 0:
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

        # 3. Structured retrieval
        q_triplet = self._extract_structure_features(query)
        # For performance, re-rank only the top-N candidates, or all items (all items is slow)
        # At the MVP stage, use all items or take the first 100 vec_indices
        struct_scores = np.zeros(len(self.corpus))
        # Simplified: compute structure scores only for the Vector Top 50 to speed up
        candidate_indices = vec_indices[0][:50]
        for idx in candidate_indices:
            struct_scores[idx] = self._calculate_structure_score(q_triplet, self.corpus[idx]['content'])

        if np.max(struct_scores) > 0:
            struct_scores = struct_scores / np.max(struct_scores)

        # 4. Hybrid fusion of scores
        # Restore vector scores to the global ordering
        full_vec_scores = np.zeros(len(self.corpus))
        for score, idx in zip(vec_scores, vec_indices[0]):
            if idx < len(full_vec_scores):
                full_vec_scores[idx] = score

        final_scores = (alpha * bm25_scores) + \
                       (beta * struct_scores) + \
                       ((1 - alpha - beta) * full_vec_scores)

        # 5. Return Top-K
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            res = self.corpus[idx].copy()
            res['score'] = final_scores[idx]
            # Remove the embedding field to avoid flooding the output
            if 'embedding' in res:
                del res['embedding']
            results.append(res)

        return results


# ==========================================
# Module 3.2: Meta-Cognitive Planner
# ==========================================
class MetaCognitivePlanner:
    def __init__(self, client, model="qwen-max"): # changed the default model to qwen-max
        self.client = client
        self.model = model
        # The tool set stays unchanged because these are internal system identifiers
        self.valid_tools = {
            "CGT", "SCP", "SCA", "PID", "FCV", "TLV", "RMD", "EVA", "CPE"
        }

    def _validate_dag(self, plan_json):
        """
        [Upgrade] Validation layer: ensure the generated plan is a Directed Acyclic Graph (DAG) with valid tool names.
        Adds automatic mapping of full tool names to abbreviations.
        """
        if not plan_json or 'tools' not in plan_json:
            return False, "Missing 'tools' field", plan_json

        tools = plan_json.get('tools', [])
        dependencies = plan_json.get('dependencies', [])

        # === New: tool-name cleaning mapping table ===
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
            # 1. If it is a standard abbreviation, keep it as is
            if t in self.valid_tools:
                cleaned_tools.append(t)
            # 2. If it is a full name, try to map it
            elif t in tool_mapping:
                # print(f"Auto-correcting tool name: '{t}' -> '{tool_mapping[t]}'")
                cleaned_tools.append(tool_mapping[t])
            # 3. If it is a partial match (e.g., the LLM outputs 'SCP Tool')
            else:
                found = False
                for full_name, abbr in tool_mapping.items():
                    if full_name in t:  # fuzzy-match the full name
                        cleaned_tools.append(abbr)
                        found = True
                        break
                if not found:
                    print(f"Warning: Unknown tool '{t}' generated. Removing from plan.")

        # Update the tool list
        tools = list(set(cleaned_tools))  # deduplicate

        # Clean dependency names as well
        cleaned_deps = []
        for upstream, downstream in dependencies:
            u = tool_mapping.get(upstream, upstream)
            v = tool_mapping.get(downstream, downstream)
            if u in tools and v in tools:
                cleaned_deps.append([u, v])

        # 2. Build the graph and in-degree table (unchanged)
        graph = defaultdict(list)
        in_degree = {t: 0 for t in tools}

        for upstream, downstream in cleaned_deps:
            graph[upstream].append(downstream)
            in_degree[downstream] += 1

        # 3. Topological sort (unchanged)
        queue = deque([t for t in tools if in_degree[t] == 0])
        sorted_plan = []

        while queue:
            node = queue.popleft()
            sorted_plan.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 4. Cycle detection
        if len(sorted_plan) != len(tools):
            # If a cycle exists, use a simple fallback: return only the tool list and drop the dependency order
            print("Cycle detected. Fallback to unordered execution.")
            return True, tools, plan_json

        return True, sorted_plan, plan_json

    def generate_plan(self, text, context_results):
        """
        Generate a tool-calling plan (English prompts)
        """
        context_str = "\n".join(
            [f"- [{doc.get('source', 'Unknown')}] {doc['content']} (Score: {doc['score']:.2f})" for doc in
             context_results])

        # All prompts here are written entirely in English
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

        # 1. Robust parsing
        plan_json = robust_json_parse(raw_content)

        if not plan_json:
            print("Failed to parse LLM output. Returning fallback plan.")
            return ["CGT", "SCP", "FCV"]

            # 2. DAG validation
        is_valid, result, _ = self._validate_dag(plan_json)

        if is_valid:
            print(f"Plan generated successfully: {result}")
            return result
        else:
            print(f"Plan validation failed ({result}). Executing fallback linear plan.")
            # Fallback: if the DAG is invalid, execute the tools in list order
            return list(plan_json.get('tools', []))