import logging
import json

logger = logging.getLogger(__name__)


class DualStageFusionEngine:
    def __init__(self, llm_client, model="qwen-max",
                 finfact_prompt_name="cross_check_simulator",
                 finguard_prompt_name="confidence_tiered_decision"):
        """
        Dual-Stage LLM Fusion Engine Initialization

        :param llm_client: LLM client instance
        :param model: model name to call (e.g., qwen-max)
        :param finfact_prompt_name: name of the FinFact prompt template to use
        :param finguard_prompt_name: name of the FinGuard prompt template to use
        """
        self.llm_client = llm_client
        self.model = model
        self.finfact_prompt_name = finfact_prompt_name
        self.finguard_prompt_name = finguard_prompt_name

        # ==========================================
        # Predefined dictionary of Golden Prompts
        # ==========================================
        self.FINFACT_PROMPTS = {
            "dual_track_verifier": {
                "system": """You are a dual-track verification system that performs both quantitative scoring and qualitative cross-referencing in parallel, then synthesizes results for final judgment.
Track A: Quantitative Scoring (0-30 scale)
Track B: Qualitative Cross-Reference (Pass/Fail checks)""",
                "user": """Execute dual-track verification on the following claim.

[CLAIM]
{claim}

[REFERENCE MATERIALS]
Summary: {justification}
Evidence: {evidence}

══════════════════════════════════════════
[TRACK A: QUANTITATIVE SCORING]
A1. Factual Accuracy (0-10): ___
A2. Evidence Alignment (0-10): ___
A3. Logical Consistency (0-10): ___
Track A Total: ___/30

══════════════════════════════════════════
[TRACK B: QUALITATIVE CROSS-REFERENCE]
B1. Core claim exists in evidence? □ PASS □ FAIL
B2. No contradictions detected? □ PASS □ FAIL
B3. Context preserved correctly? □ PASS □ FAIL
Track B Result: ___ of 3 PASS

══════════════════════════════════════════
[SYNTHESIS DECISION MATRIX]
| Track A Score | Track B Passes | Final Decision |
|---------------|----------------|----------------|
| >= 22         | >= 2           | True           |
| >= 25         | >= 1           | True           |
| < 22          | 3              | True           |
| Otherwise     | -              | False          |

Final Output: Prediction: True or Prediction: False"""
            },
            "weighted_evidence_scorer": {
                "system": """You are an evidence-weight-based scoring system. You will score the credibility factors of the claim (0-10 points); a total score below 20 (out of 30) will be marked as False.""",
                "user": """Please score the following claim based on evidence.

[CLAIM]
{claim}

[EVIDENCE MATERIALS]
{justification}
{evidence}

[SCORING ITEMS]
A. Evidence Coverage (0-10): 0 = no direct evidence, 10 = evidence fully covers every detail of the claim
B. Consistency (0-10): 0 = the claim contradicts the evidence, 10 = the claim is highly consistent with the evidence
C. Context Accuracy (0-10): 0 = severely taken out of context, 10 = fully faithful to the original meaning

Please calculate the total score.
Decision Rule: Total Score >= 20 -> True; Total Score < 20 -> False.
Output format: Prediction: True or Prediction: False"""
            },
            "cross_check_simulator": {
                "system": """You are a research assistant simulating a cross-referencing process.
You use the provided "Justification" and "Evidence" as your ground truth knowledge base to verify the "Claim".""",
                "user": """Verify the claim by cross-referencing it against the provided ground truth.

[Target Claim]
{claim}

[Ground Truth Knowledge Base]
{justification}
{evidence}

[Simulation]
- Initial Check: Does the claim exist in the Knowledge Base?
- Detail Verification: Do specific numbers, dates, and entities match exactly?
- Conflict Detection: Is there any statement in the Knowledge Base that directly contradicts the claim?

Verdict:
Prediction: True or Prediction: False"""
            }
        }

        self.FINGUARD_PROMPTS = {
            "enhanced_cot_6step": {
                "system": """You are a news verification AI that uses an enhanced chain-of-thought approach.
You will analyze the news through six refined steps to ensure the comprehensiveness of your judgment.""",
                "user": """Please analyze the following news using the six-step chain-of-thought method.

[NEWS CONTENT]
{text}

[SIX-STEP DETAILED ANALYSIS]
Step 1 - Information Extraction: Are the 5W1H elements of the news (who, what, when, where, why, how) complete?

Step 2 - Source Review: Who is the source? Is it authoritative and credible? Can it be traced and verified?

Step 3 - Language Analysis: Does it use neutral and objective language? Are there sensational words such as "shocking" or "astonishing"?

Step 4 - Data Verification: Are the numbers, dates, amounts, etc., specific and reasonable?

Step 5 - Logic Check: Is the reasoning rigorous? Are there causal fallacies or logical leaps?

Step 6 - Motivation Assessment: What is the likely purpose of publishing this news? Is there a clear interest-driven motive?

[OVERALL JUDGMENT]
Based on the six-step analysis, output: Prediction: True or Prediction: False"""
            },
            "confidence_tiered_decision": {
                "system": """You are a confidence-based news verification system.
You assess the news across multiple dimensions and calculate a confidence score.
Different confidence levels lead to different decision thresholds.""",
                "user": """Analyze the following news with confidence-based decision making.

[NEWS CONTENT]
{text}

[DIMENSION ANALYSIS]
D1. Source Quality
    - Assessment: (Strong/Moderate/Weak/None)
    - Confidence: (High/Medium/Low)

D2. Factual Precision
    - Assessment: (Detailed/Partial/Vague)
    - Confidence: (High/Medium/Low)

D3. Language Objectivity
    - Assessment: (Neutral/Slight bias/Heavy bias)
    - Confidence: (High/Medium/Low)

D4. Logical Coherence
    - Assessment: (Sound/Minor issues/Major flaws)
    - Confidence: (High/Medium/Low)

[CONFIDENCE SCORING]
High confidence = 3 points, Medium = 2 points, Low = 1 point
Total Confidence Score: ___/12

[TIERED DECISION]
- If Confidence >= 9 AND majority positive assessments → True
- If Confidence >= 9 AND majority negative assessments → False
- If Confidence >= 6 → Lean toward positive assessments
- If Confidence < 6 → Default to False (insufficient evidence)

Final Output: Prediction: True or Prediction: False"""
            },
            "cot_with_scoring": {
                "system": """You are a news verification system that combines chain-of-thought reasoning with quantitative scoring.
You will analyze step by step, score each step from 0 to 10, and make the final decision based on the total score.""",
                "user": """Please analyze the following news step by step and score each dimension.

[NEWS CONTENT]
{text}

[STEP-BY-STEP ANALYSIS AND SCORING]
Step 1 - Core Content (0-10 points): What is the core claim of the news? Is the information clear and specific?
Score: ___

Step 2 - Language Style (0-10 points): Is the language objective and neutral (10 points) or sensational and exaggerated (0 points)?
Score: ___

Step 3 - Evidence Quality (0-10 points): Does it cite verifiable sources and specific data?
Score: ___

Step 4 - Logical Rigor (0-10 points): Is the reasoning sound? Are there logical gaps?
Score: ___

[DECISION RULE] Total Score >= 28 -> True; Total Score < 28 -> False
Output format: Prediction: True or Prediction: False"""
            },
            "staged_tribunal": {
                "system": """You are a three-stage news adjudication system:
Stage 1: Initial Screening (quickly identify obvious issues)
Stage 2: Deep Analysis (detailed multi-dimensional review)
Stage 3: Final Adjudication (comprehensive scoring decision)""",
                "user": """Begin the three-stage adjudication of the following news.

[NEWS CONTENT]
{text}

═══════════════════════════════════════
[STAGE 1: INITIAL SCREENING]
□ Is there a clear information source?
□ Does it contain verifiable, specific information?
□ Does it contain obvious sensational language?
Screening Result: Pass / Questionable / Clearly False

═══════════════════════════════════════
[STAGE 2: DEEP ANALYSIS] (if the initial screening is not "Clearly False")
- Source credibility analysis: ...
- Content logic analysis: ...
- Professional accuracy analysis: ...
- Language objectivity analysis: ...

═══════════════════════════════════════
[STAGE 3: FINAL ADJUDICATION]
Overall Score (0-30):
- Credibility: ___/10
- Accuracy: ___/10
- Objectivity: ___/10
Total Score: ___/30

[ADJUDICATION RULE] Initial screening "Clearly False" -> False; Total Score >= 20 -> True; otherwise -> False
Output format: Prediction: True or Prediction: False"""
            }
        }

    def fuse(self, blackboard, dataset_source):
        """
        Execute dual-stage fusion (dynamically receives dataset_source)
        """
        logger.info(f"Starting Dual-Stage Fusion for [{dataset_source}] data...")

        # Stage 1: High-density evidence compression
        compressed_insights = self._compress_evidence(blackboard["tool_outputs"])
        logger.info(f"Stage 1 Completed. Compressed Insights:\n{compressed_insights}")

        # Stage 2: Inject the dataset-specific Golden Prompt for the final prediction
        final_response = self._apply_golden_prompt(blackboard, compressed_insights, dataset_source)

        # Parse the final label
        final_label = "Uncertain"
        if "Prediction: True" in final_response or "Prediction: true" in final_response:
            final_label = "Real"
        elif "Prediction: False" in final_response or "Prediction: false" in final_response:
            final_label = "Fake"

        logger.info(f"Stage 2 Completed. Final Label: {final_label}")

        return {
            "final_label": final_label,
            "explanation_path": final_response,
            "compressed_evidence": compressed_insights
        }

    def _compress_evidence(self, tool_outputs):
        if not tool_outputs:
            return "No agent tool clues available."

        raw_outputs_str = json.dumps(tool_outputs, ensure_ascii=False, indent=2)[:2000]

        system_prompt = "You are a ruthless summarization machine. Your goal is to condense verbose system logs into 0 or 1"
        user_prompt = f"""Please analyze the inspection outputs of the following tool modules for the same news item:
{raw_outputs_str}

Requirements:
Completely ignore formatting and confidence scores; strictly summarize as 0/1.
Format example:
1. 0.
Please output your summary directly:"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in Evidence Compression: {e}")
            return "Failed to extract tool clues."

    def _apply_golden_prompt(self, blackboard, compressed_insights, dataset_source):
        initial_text = blackboard.get("initial_text", "")
        retrieved_evidence = blackboard.get("retrieved_evidence", "No external retrieved evidence.")

        # Dynamically load the corresponding prompt template based on the dataset source
        if "FinFact" in dataset_source:
            template = self.FINFACT_PROMPTS.get(self.finfact_prompt_name, self.FINFACT_PROMPTS["cross_check_simulator"])
            system_p = template["system"]
            justification_injected = f"[Agent Multi-Dimensional Investigation Digest]\n{compressed_insights}"
            user_p = template["user"].format(
                claim=initial_text,
                justification=justification_injected,
                evidence=retrieved_evidence
            )

        elif "FinGuard" in dataset_source:
            template = self.FINGUARD_PROMPTS.get(self.finguard_prompt_name, self.FINGUARD_PROMPTS["enhanced_cot_6step"])
            system_p = template["system"]
            injected_text = f"{initial_text}\n\n[Additional Reference: Core Clues Extracted by the AI Investigation Matrix]\n{compressed_insights}"
            user_p = template["user"].format(
                text=injected_text
            )
        else:
            logger.warning(f"Unknown dataset_source: {dataset_source}. Defaulting to FinFact template.")
            template = self.FINFACT_PROMPTS["cross_check_simulator"]
            system_p = template["system"]
            user_p = template["user"].format(claim=initial_text, justification=compressed_insights,
                                             evidence=retrieved_evidence)

        # Final adjudication with the golden prompt
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_p},
                    {"role": "user", "content": user_p}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in Golden Prompt Predictor: {e}")
            return "Prediction: False"