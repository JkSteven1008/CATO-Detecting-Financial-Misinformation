import logging
import json

logger = logging.getLogger(__name__)


class DualStageFusionEngine:
    def __init__(self, llm_client, model="qwen-max",
                 finfact_prompt_name="cross_check_simulator",
                 finguard_prompt_name="confidence_tiered_decision"):
        """
        双阶段 LLM 融合器初始化

        :param llm_client: LLM 客户端实例
        :param model: 调用的模型名称 (如 qwen-max)
        :param finfact_prompt_name: 选择使用的 FinFact 提示词模板名称
        :param finguard_prompt_name: 选择使用的 FinGuard 提示词模板名称
        """
        self.llm_client = llm_client
        self.model = model
        self.finfact_prompt_name = finfact_prompt_name
        self.finguard_prompt_name = finguard_prompt_name

        # ==========================================
        # 预设的 Golden Prompts 字典
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
                "system": """你是一个基于证据权重的评分系统。你会对声明的可信度要素进行打分（0-10分），总分低于20分（满分30）将被标记为False。""",
                "user": """请对以下声明进行基于证据的打分评估。

【声明】
{claim}

【证据材料】
{justification}
{evidence}

【评分项】
A. 证据覆盖度 (0-10): 0=无直接证据，10=证据完全覆盖声明的所有细节
B. 一致性 (0-10): 0=声明与证据矛盾，10=声明与证据高度一致
C. 语境准确性 (0-10): 0=严重断章取义，10=完全忠实于原意

请计算总分。
Decision Rule: Total Score >= 20 -> True; Total Score < 20 -> False.
输出格式：Prediction: True 或 Prediction: False"""
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
                "system": """你是一个采用增强版链式思维的新闻核查AI。
你会通过6个精细化步骤进行分析，确保判断的全面性。""",
                "user": """请使用6步链式思维方法分析以下新闻。

【新闻内容】
{text}

【6步精细分析】
Step 1 - 信息提取：新闻的5W1H（何人、何事、何时、何地、为何、如何）是否完整？

Step 2 - 来源审查：消息来源是谁？是否为权威可信来源？是否可追溯验证？

Step 3 - 语言分析：是否使用中立客观的语言？有无"惊爆""震惊"等煽动性词汇？

Step 4 - 数据核实：涉及的数字、日期、金额等是否具体且合理？

Step 5 - 逻辑检验：论证过程是否严密？是否存在因果谬误或逻辑跳跃？

Step 6 - 动机评估：发布此新闻可能的目的是什么？是否有明显利益驱动？

【综合判断】
基于6步分析，输出：Prediction: True 或 Prediction: False"""
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
                "system": """你是一个结合链式思维和量化评分的新闻核查系统。
你会按步骤分析，并对每个步骤进行0-10分评估，最终根据总分判断。""",
                "user": """请逐步分析以下新闻，并对每个维度打分。

【新闻内容】
{text}

【分步分析与评分】
Step 1 - 内容核心 (0-10分): 新闻的核心主张是什么？信息是否清晰明确？
评分: ___

Step 2 - 语言风格 (0-10分): 语言是客观中立(10分)还是煽动夸张(0分)？
评分: ___

Step 3 - 证据质量 (0-10分): 是否引用了可验证的来源和具体数据？
评分: ___

Step 4 - 逻辑严密性 (0-10分): 论证过程是否合理？有无逻辑漏洞？
评分: ___

【决策规则】总分 >= 28 → True; 总分 < 28 → False
输出格式：Prediction: True 或 Prediction: False"""
            },
            "staged_tribunal": {
                "system": """你是一个三阶段新闻审判系统：
阶段一：初步筛查（快速识别明显问题）
阶段二：深度分析（多维度详细审查）
阶段三：最终裁决（综合评分决策）""",
                "user": """开始对以下新闻进行三阶段审判。

【新闻内容】
{text}

═══════════════════════════════════════
【阶段一：初步筛查】
□ 是否有明确的信息来源？
□ 是否包含可验证的具体信息？
□ 是否存在明显的煽动性语言？
初筛结果: 通过 / 存疑 / 明显虚假

═══════════════════════════════════════
【阶段二：深度分析】(如初筛非"明显虚假")
- 来源可信度分析：...
- 内容逻辑性分析：...
- 专业准确性分析：...
- 语言客观性分析：...

═══════════════════════════════════════
【阶段三：最终裁决】
综合评分 (0-30):
- 可信度: ___/10
- 准确性: ___/10
- 客观性: ___/10
总分: ___/30

【裁决规则】初筛"明显虚假"→False; 总分>=20→True; 否则→False
输出格式：Prediction: True 或 Prediction: False"""
            }
        }

    def fuse(self, blackboard, dataset_source):
        """
        执行双阶段融合 (动态接收 dataset_source)
        """
        logger.info(f"Starting Dual-Stage Fusion for [{dataset_source}] data...")

        # 阶段 1：高密度证据压缩
        compressed_insights = self._compress_evidence(blackboard["tool_outputs"])
        logger.info(f"Stage 1 Completed. Compressed Insights:\n{compressed_insights}")

        # 阶段 2：注入对应数据集的 Golden Prompt 进行最终预测
        final_response = self._apply_golden_prompt(blackboard, compressed_insights, dataset_source)

        # 解析最终标签
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
            return "无特工工具线索。"

        raw_outputs_str = json.dumps(tool_outputs, ensure_ascii=False, indent=2)[:2000]

        system_prompt = "你是一个无情的总结机器，目标是把冗长的系统日志，总结为0或1"
        user_prompt = f"""请分析以下各个工具模块针对同一条新闻的检查输出：
{raw_outputs_str}

要求：
完全忽略格式和置信度分数，用英文，强制总结为0/1。
格式示例：
1. 0.
请直接输出你的总结："""

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
            return "提取工具线索失败。"

    def _apply_golden_prompt(self, blackboard, compressed_insights, dataset_source):
        initial_text = blackboard.get("initial_text", "")
        retrieved_evidence = blackboard.get("retrieved_evidence", "无外部检索证据。")

        # 根据来源动态加载对应的提示词模板
        if "FinFact" in dataset_source:
            template = self.FINFACT_PROMPTS.get(self.finfact_prompt_name, self.FINFACT_PROMPTS["cross_check_simulator"])
            system_p = template["system"]
            justification_injected = f"【Agent特工多维调查提炼】\n{compressed_insights}"
            user_p = template["user"].format(
                claim=initial_text,
                justification=justification_injected,
                evidence=retrieved_evidence
            )

        elif "FinGuard" in dataset_source:
            template = self.FINGUARD_PROMPTS.get(self.finguard_prompt_name, self.FINGUARD_PROMPTS["enhanced_cot_6step"])
            system_p = template["system"]
            injected_text = f"{initial_text}\n\n[附加参考：AI调查矩阵提取的核心线索]\n{compressed_insights}"
            user_p = template["user"].format(
                text=injected_text
            )
        else:
            logger.warning(f"Unknown dataset_source: {dataset_source}. Defaulting to FinFact template.")
            template = self.FINFACT_PROMPTS["cross_check_simulator"]
            system_p = template["system"]
            user_p = template["user"].format(claim=initial_text, justification=compressed_insights,
                                             evidence=retrieved_evidence)

        # 黄金 Prompt 最终裁决
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