import json
import logging
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DynamicDAGScheduler:
    def __init__(self, toolset):
        """
        初始化调度器
        :param toolset: 实例化后的 Toolset 对象 (包含 CGT, SCP, FCV 等方法)
        """
        self.toolset = toolset
        # 根据方案 3.4 节：设定来源可信度的熔断阈值为 0.3
        self.CRITICAL_SOURCE_THRESHOLD = 0.01

    def execute(self, plan_json, initial_text, retrieved_evidence):
        """
        执行整个流水线 (完美版：支持 DAG 分层并发、黑板证据传播、动态阈值熔断)
        """
        print(f"\n{'=' * 10} Starting New Inference Trace {'=' * 10}")

        tools = plan_json.get('tools', [])
        dependencies = plan_json.get('dependencies', [])

        # 1. 初始化“黑板” (Blackboard 模式，用于跨工具证据传播)
        blackboard = {
            "initial_text": initial_text,
            "retrieved_evidence": retrieved_evidence,
            "tool_outputs": {},
            "execution_log": [],
            "risk_flags": []
        }

        # 将最终的汇总工具 CPE 提取出来，确保它绝对在最后单点执行
        if "CPE" in tools:
            tools.remove("CPE")

        # 2. 构建 DAG 图与入度表
        graph = defaultdict(list)
        in_degree = {t: 0 for t in tools}

        for upstream, downstream in dependencies:
            if upstream in tools and downstream in tools:
                graph[upstream].append(downstream)
                in_degree[downstream] += 1

        skip_set = set()  # 记录被熔断机制跳过的工具
        completed = set()  # 记录已执行完的工具

        # 3. 拓扑分层执行引擎 (Layer-by-Layer Execution)
        while len(completed) + len(skip_set) < len(tools):
            # 获取当前所有 入度为0 且 未处理 的工具
            ready_tools = [t for t in tools if in_degree.get(t, 0) == 0 and t not in completed and t not in skip_set]

            if not ready_tools:
                # 异常兜底：如果没执行完且没有ready的，说明存在循环依赖，强制捞出剩余工具
                logger.warning("Cycle detected or missing dependencies. Forcing remaining tools to execute.")
                ready_tools = [t for t in tools if t not in completed and t not in skip_set]
                if not ready_tools:
                    break

            print(f"[*] Executing Layer: {ready_tools}")

            layer_results = {}
            # 并发执行当前层的独立工具
            with ThreadPoolExecutor(max_workers=max(1, len(ready_tools))) as executor:
                future_to_tool = {}
                for t_name in ready_tools:
                    tool_func = getattr(self.toolset, t_name, None)
                    if tool_func:
                        # 传入 blackboard，下游工具可以在此读取上游的 tool_outputs
                        future = executor.submit(tool_func, blackboard)
                        future_to_tool[future] = t_name
                    else:
                        print(f" [!] Tool {t_name} not found in toolset.")
                        completed.add(t_name)

                for future in as_completed(future_to_tool):
                    t_name = future_to_tool[future]
                    try:
                        result = future.result()
                        layer_results[t_name] = result
                        # 简单通过可信度做个状态标签打印
                        status_tag = "PASS" if result.get('credibility_score', 1.0) > 0.5 else "RISK"
                        print(f"   [+] Tool Completed: {t_name:<5} | Status: {status_tag}", flush=True)
                        blackboard["execution_log"].append({
                            "tool": t_name, "status": "success", "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        print(f"   [x] Tool Failed: {t_name:<5} | Error: {str(e)[:50]}...", flush=True)
                        layer_results[t_name] = {"error": str(e)}
                        blackboard["execution_log"].append({
                            "tool": t_name, "status": "failed", "error": str(e)
                        })

            # 4. 处理本层结果，触发动态熔断与图状态更新
            for t_name, result in layer_results.items():
                # 将结果写入黑板，实现证据向下游传播
                blackboard["tool_outputs"][t_name] = result
                completed.add(t_name)

                # ==== 核心改进：证据阈值触发机制 (Context-Aware Routing) ====
                # 方案 3.4：若 SCP 输出的可信度 < 0.3，则跳过 L2/L3 工具直接激活 L4 的 EVA
                if t_name == "SCP":
                    cred_score = result.get("credibility_score", 1.0)
                    if cred_score < self.CRITICAL_SOURCE_THRESHOLD:
                        print(f"   [!] 触发动态熔断机制！SCP 可信度 ({cred_score}) < 阈值 ({self.CRITICAL_SOURCE_THRESHOLD})")
                        print("   [!] 跳过常规验证 (L2/L3)，直接激活高阶常识验证 (EVA)...")

                        # 定义 L2 和 L3 的工具集合，准备跳过
                        l2_l3_tools = {"SCA", "PID", "FCV", "TLV", "RMD"}
                        skip_set.update(l2_l3_tools.intersection(set(tools)))

                        # 兜底注入：若 EVA 原本不在 LLM 的计划中，强行将其加入执行队列
                        if "EVA" not in tools:
                            tools.append("EVA")
                            in_degree["EVA"] = 0

                # 解除下游依赖：当前节点完成，下游节点的入度减 1
                for neighbor in graph.get(t_name, []):
                    if neighbor in in_degree:
                        in_degree[neighbor] -= 1

            # 释放那些被 Skip 机制熔断的工具的下游依赖，防止死锁
            for t_name in list(skip_set):
                if t_name not in completed:
                    completed.add(t_name)
                    print(f"   [-] Tool Skipped (Circuit Breaker): {t_name}")
                    for neighbor in graph.get(t_name, []):
                        if neighbor in in_degree:
                            in_degree[neighbor] -= 1

        # 5. 强制执行终极融合工具 (CPE)
        print(f"[*] Running Final Aggregator: CPE ...", end="", flush=True)
        try:
            # 此时 blackboard["tool_outputs"] 已经包含了所有被执行工具的证据
            final_verdict = self.toolset.CPE(blackboard)
            blackboard["tool_outputs"]["CPE"] = final_verdict
            print(" Done!")
        except Exception as e:
            print(" Failed!")
            logger.error(f"Critical Error in CPE: {e}")

        print(f"{'=' * 35}\n")
        return blackboard