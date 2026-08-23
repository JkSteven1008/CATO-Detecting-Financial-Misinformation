import json
import re


class FinancialFactCheckingTools:
    def __init__(self, client, model="qwen-max"): # adjust the model name here as needed
        self.client = client
        self.model = model

    # ==========================================
    # Helper utility: robust JSON parsing
    # ==========================================
    def _parse_json(self, llm_output):
        """
        Robustly parse JSON from LLM output, handling markdown blocks and errors.
        """
        if not isinstance(llm_output, str):
            return llm_output

        # 1. Try regex for code blocks
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, llm_output, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # 2. Try regex for outer braces
            pattern = r"\{.*\}"
            match = re.search(pattern, llm_output, re.DOTALL)
            json_str = match.group(0) if match else llm_output

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                import ast
                return ast.literal_eval(json_str)
            except:
                print(f"[Error] Failed to parse JSON: {json_str[:50]}...")
                return {"error": "JSON parsing failed", "raw_output": llm_output}

    def _call_llm(self, prompt, sys_msg="You are a financial risk analyst."):
        """
        Centralized LLM call method.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{sys_msg} Output strictly in JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Low temperature for factual consistency
            )
            return self._parse_json(response.choices[0].message.content)
        except Exception as e:
            print(f"LLM Call Error: {e}")
            return {"error": str(e)}

    # ==========================================
    # L1: Basic grounding
    # ==========================================
    def CGT(self, context_data):
        """
        Contextual Grounding Tool: Identifies missing context.
        """
        text = context_data.get('initial_text')
        evidence = context_data.get('retrieved_evidence', [])
        evidence_text = "\n".join([f"- {doc['content']}" for doc in evidence])

        prompt = f"""
        TASK: Compare the Target Text with the Retrieved Evidence to identify MISSING critical context.

        Target Text: "{text}"
        Retrieved Evidence:
        {evidence_text}

        Analyze:
        1. Does the text mention specific figures/policies without background (e.g., "Subsidy approved" but missing "Audit required")?
        2. Are there conditions in the evidence that are omitted in the text?

        Output JSON:
        {{
            "missing_context": ["List of omitted key facts..."],
            "alignment_score": 0.0 to 1.0 (1.0 = Fully Aligned, 0.0 = Totally Misleading due to omission),
            "reason": "Explanation..."
        }}
        """
        return self._call_llm(prompt)

    def SCP(self, context_data):
        """
        Source Credibility Propagator: Analyzes source authority.
        """
        text = context_data.get('initial_text')
        # If the retrieved results contain a source field, extract it
        evidence = context_data.get('retrieved_evidence', [])
        sources = list(set([doc.get('source', 'Unknown') for doc in evidence]))

        prompt = f"""
        TASK: Analyze the credibility of the information sources associated with the text context.

        Target Text: "{text}"
        Detected Sources in Context: {sources}

        Analyze:
        1. Is the source a Tier-1 financial institution/regulator (High Credibility)?
        2. Is it an anonymous social media account or unverified blog (Low Credibility)?
        3. Does the text cite "insiders" or "rumors" without verification?

        Output JSON:
        {{
            "credibility_score": 0.0 to 1.0 (1.0 = Official/Verified, 0.2 = Rumor/Anonymous),
            "source_type": "Official/Media/Social/Anonymous",
            "risk_flag": true/false (true if source is suspicious)
        }}
        """
        return self._call_llm(prompt)

    # ==========================================
    # L2: Semantics and pragmatics
    # ==========================================
    def SCA(self, context_data):
        """
        Semantic Coherence Analyzer: Detects internal logic contradictions.
        """
        text = context_data.get('initial_text')

        prompt = f"""
        TASK: Detect INTERNAL logical contradictions within the text itself.

        Target Text: "{text}"

        Check for:
        1. Conflicting sentiment (e.g., "Revenue plummeted" AND "Best performance ever").
        2. Numerical mismatch (e.g., "grew by 10%" vs "dropped to 50%").

        Output JSON:
        {{
            "is_coherent": true/false,
            "contradictions": ["Quote part A vs Quote part B"],
            "coherence_score": 0.0 to 1.0
        }}
        """
        return self._call_llm(prompt)

    def PID(self, context_data):
        """
        Pragmatic Intent Decoder: Detects emotional manipulation/FOMO.
        """
        text = context_data.get('initial_text')

        prompt = f"""
        TASK: Analyze the pragmatic intent and rhetorical style. 
        Detect attempts to manipulate investor emotion (FOMO - Fear Of Missing Out, or FUD - Fear Uncertainty Doubt).

        Target Text: "{text}"

        Keywords to watch: "Guaranteed", "Once in a lifetime", "Skyrocket", "Panic", "Inside info".

        Output JSON:
        {{
            "intent": "Informative/Persuasive/Manipulative/Speculative",
            "emotional_intensity": 0.0 to 1.0,
            "manipulation_tactics": ["List tactics found e.g., Urgency, Authority Bias"]
        }}
        """
        return self._call_llm(prompt)

    # ==========================================
    # L3: Facts and time
    # ==========================================
    def FCV(self, context_data):
        """
        Factual Consistency Verifier: Claims vs Evidence.
        """
        text = context_data.get('initial_text')
        evidence = context_data.get('retrieved_evidence', [])
        evidence_text = "\n".join([f"- {doc['content']}" for doc in evidence])

        prompt = f"""
        TASK: Verify the factual claims in the text against the retrieved evidence.

        Target Text: "{text}"
        Trusted Evidence:
        {evidence_text}

        Instructions:
        1. Extract key claims (Entities, Numbers, Events) from the text.
        2. Check if each claim is SUPPORTED, CONTRADICTED, or UNSUBSTANTIATED by the evidence.

        Output JSON:
        {{
            "verified_facts": ["List confirmed facts"],
            "contradictions": [
                {{"claim": "Text claim", "evidence": "Evidence quote", "type": "Direct Contradiction/Number Mismatch"}}
            ],
            "consistency_score": 0.0 to 1.0
        }}
        """
        return self._call_llm(prompt)

    def TLV(self, context_data):
        """
        Temporal Logic Validator: Timeline analysis.
        """
        text = context_data.get('initial_text')
        evidence = context_data.get('retrieved_evidence', [])
        evidence_text = "\n".join(
            [f"- {doc['content']} (Source Date: {doc.get('publish_time', 'N/A')})" for doc in evidence])

        prompt = f"""
        TASK: Analyze the Temporal Logic (Timeline) of the text vs evidence.

        Target Text: "{text}"
        Evidence with Dates:
        {evidence_text}

        Check for:
        1. Anachronisms (e.g., Reporting an event BEFORE it happened).
        2. Expired News (e.g., Presenting 3-year-old data as "Breaking News").
        3. Logical Sequence violations (e.g., "Profits released" before "Quarter ended").

        Output JSON:
        {{
            "is_chronologically_valid": true/false,
            "time_errors": ["Description of error"],
            "temporal_status": "Current/Outdated/Future Prediction"
        }}
        """
        return self._call_llm(prompt)

    def RMD(self, context_data):
        """
        Rhetorical Manipulation Detector: Specific financial hype detection.
        """
        text = context_data.get('initial_text')

        prompt = f"""
        TASK: Detect specific "Financial Hype" or "Fraud" keywords/patterns.

        Target Text: "{text}"

        Look for patterns common in "Pump and Dump" schemes:
        - "Next Bitcoin"
        - "100x returns"
        - "Exclusive access"

        Output JSON:
        {{
            "hype_score": 0.0 to 1.0 (High score = High Hype),
            "trap_keywords": ["List found keywords"],
            "risk_level": "Low/Medium/High/Critical"
        }}
        """
        return self._call_llm(prompt)

    # ==========================================
    # L4: Higher-order reasoning
    # ==========================================
    def EVA(self, context_data):
        """
        Expectation Violation Analyzer: Common sense check.
        """
        text = context_data.get('initial_text')

        prompt = f"""
        TASK: Check if the text violates Financial Common Sense or Market Norms.

        Target Text: "{text}"

        Examples of Violations:
        - "Risk-free 50% monthly return" (Violates risk-return tradeoff).
        - "Company A acquired Apple" (Violates market capitalization reality).

        Output JSON:
        {{
            "violation_detected": true/false,
            "violation_type": "Impossible Return/Market Cap Logic/Regulatory Norm",
            "reasoning": "Explanation"
        }}
        """
        return self._call_llm(prompt)

    def CPE(self, context_data):
        """
        Contradiction Propagation Engine: The Final Aggregator.
        [Core Improvement]: This tool reads the outputs of all previous tools.
        """
        text = context_data.get('initial_text')
        # Retrieve the execution results of the previous tools
        previous_outputs = context_data.get('tool_outputs', {})

        # Serialize the previous tool results into a string for the LLM to analyze
        prev_tools_summary = json.dumps(previous_outputs, indent=2)

        prompt = f"""
        TASK: Synthesize all evidence from previous tools to make a Final Verdict on whether the text is Misinformation.

        Target Text: "{text}"

        Findings from Sub-Tools:
        {prev_tools_summary}

        Logic for Verdict:
        1. FAKE: If FCV found direct factual contradictions or TLV found timeline errors.
        2. FAKE (Rumor): If SCP found source is anonymous AND PID found high manipulation.
        3. REAL: If the retrieved evidence strictly aligns with and supports the core claim.
        

        Output JSON:
        {{
            "final_verdict": "Real/Fake/Misleading/Unverified",
            "fake_probability": 0.0 to 1.0,
            "primary_evidence": "The strongest proof found by tools (e.g. 'FCV found revenue data mismatch')",
            "explanation": "Comprehensive summary for the user."
        }}
        """
        return self._call_llm(prompt, sys_msg="You are the Chief Judge of a misinformation detection system.")