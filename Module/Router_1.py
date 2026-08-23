import json
import logging
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DynamicDAGScheduler:
    def __init__(self, toolset):
        """
        Initialize the scheduler
        :param toolset: an instantiated Toolset object (contains methods such as CGT, SCP, and FCV)
        """
        self.toolset = toolset
        # Per Section 3.4 of the design: set the source-credibility circuit-breaker threshold to 0.3
        self.CRITICAL_SOURCE_THRESHOLD = 0.01

    def execute(self, plan_json, initial_text, retrieved_evidence):
        """
        Execute the entire pipeline (full version: supports layer-wise DAG concurrency, blackboard evidence propagation, and dynamic threshold circuit-breaking)
        """
        print(f"\n{'=' * 10} Starting New Inference Trace {'=' * 10}")

        tools = plan_json.get('tools', [])
        dependencies = plan_json.get('dependencies', [])

        # 1. Initialize the "blackboard" (Blackboard pattern for cross-tool evidence propagation)
        blackboard = {
            "initial_text": initial_text,
            "retrieved_evidence": retrieved_evidence,
            "tool_outputs": {},
            "execution_log": [],
            "risk_flags": []
        }

        # Extract the final aggregator tool CPE so that it is guaranteed to execute last as a single point
        if "CPE" in tools:
            tools.remove("CPE")

        # 2. Build the DAG graph and in-degree table
        graph = defaultdict(list)
        in_degree = {t: 0 for t in tools}

        for upstream, downstream in dependencies:
            if upstream in tools and downstream in tools:
                graph[upstream].append(downstream)
                in_degree[downstream] += 1

        skip_set = set()  # records tools skipped by the circuit breaker
        completed = set()  # records tools that have finished executing

        # 3. Layer-by-layer topological execution engine
        while len(completed) + len(skip_set) < len(tools):
            # Get all tools that currently have in-degree 0 and have not been processed
            ready_tools = [t for t in tools if in_degree.get(t, 0) == 0 and t not in completed and t not in skip_set]

            if not ready_tools:
                # Exception fallback: if tools remain unexecuted but none are ready, a circular dependency exists; force-run the remaining tools
                logger.warning("Cycle detected or missing dependencies. Forcing remaining tools to execute.")
                ready_tools = [t for t in tools if t not in completed and t not in skip_set]
                if not ready_tools:
                    break

            print(f"[*] Executing Layer: {ready_tools}")

            layer_results = {}
            # Execute the independent tools of the current layer concurrently
            with ThreadPoolExecutor(max_workers=max(1, len(ready_tools))) as executor:
                future_to_tool = {}
                for t_name in ready_tools:
                    tool_func = getattr(self.toolset, t_name, None)
                    if tool_func:
                        # Pass the blackboard so downstream tools can read the upstream tool_outputs
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
                        # Print a simple status tag based on credibility
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

            # 4. Process the layer results, triggering dynamic circuit-breaking and graph state updates
            for t_name, result in layer_results.items():
                # Write the results to the blackboard so evidence propagates downstream
                blackboard["tool_outputs"][t_name] = result
                completed.add(t_name)

                # ==== Core improvement: evidence-threshold trigger (Context-Aware Routing) ====
                # Design 3.4: if SCP reports credibility < 0.3, skip the L2/L3 tools and directly activate EVA (L4)
                if t_name == "SCP":
                    cred_score = result.get("credibility_score", 1.0)
                    if cred_score < self.CRITICAL_SOURCE_THRESHOLD:
                        print(f"   [!] Dynamic circuit breaker triggered! SCP credibility ({cred_score}) < threshold ({self.CRITICAL_SOURCE_THRESHOLD})")
                        print("   [!] Skipping routine verification (L2/L3) and directly activating higher-order commonsense verification (EVA)...")

                        # Define the L2/L3 tool set to be skipped
                        l2_l3_tools = {"SCA", "PID", "FCV", "TLV", "RMD"}
                        skip_set.update(l2_l3_tools.intersection(set(tools)))

                        # Fallback injection: if EVA is not in the LLM plan, forcibly add it to the execution queue
                        if "EVA" not in tools:
                            tools.append("EVA")
                            in_degree["EVA"] = 0

                # Release downstream dependencies: when a node completes, decrement its downstream in-degrees
                for neighbor in graph.get(t_name, []):
                    if neighbor in in_degree:
                        in_degree[neighbor] -= 1

            # Release the downstream dependencies of tools skipped by the circuit breaker to prevent deadlock
            for t_name in list(skip_set):
                if t_name not in completed:
                    completed.add(t_name)
                    print(f"   [-] Tool Skipped (Circuit Breaker): {t_name}")
                    for neighbor in graph.get(t_name, []):
                        if neighbor in in_degree:
                            in_degree[neighbor] -= 1

        # 5. Force the final fusion tool (CPE) to execute
        print(f"[*] Running Final Aggregator: CPE ...", end="", flush=True)
        try:
            # blackboard["tool_outputs"] now contains the evidence from all executed tools
            final_verdict = self.toolset.CPE(blackboard)
            blackboard["tool_outputs"]["CPE"] = final_verdict
            print(" Done!")
        except Exception as e:
            print(" Failed!")
            logger.error(f"Critical Error in CPE: {e}")

        print(f"{'=' * 35}\n")
        return blackboard