# panel_discussion.py
from typing import Dict, Any, List
import copy, json, time

class PanelDiscussion:
    """
    1. 批量检查窗口内 N(<=5) 帧的 triplets / analysis
    2. 不合规时回溯让 SurgicalAgent 重新生成（最多 max_retries 次）
    3. judge_client 负责“一致性 + 格式” 判定；openai_client 用于回溯
    """

    def __init__(self,
                 judge_client,                 # 便宜的小模型（只做判定）
                 openai_client,                # 用于回溯时重新生成
                 fsm_prompt: str,              # ★ 传入 FSM_PROMPT
                 max_retries: int = 3,
                 base_messages: List[Dict] | None = None):
        self.judge_client  = judge_client
        self.openai_client = openai_client
        self.fsm_prompt    = fsm_prompt
        self.max_retries   = max_retries
        self.base_messages = base_messages or []

    # ------------------------------------------------------------------
    # 公共入口：一次传入最近 N 帧结果，必要时批量回溯
    # ------------------------------------------------------------------
    def batch_evaluate_and_refine(
            self,
            batch_results:   List[Dict[str, Any]],
            batch_messages:  List[List[Dict[str, Any]]],
            memory_summary:  str,
            surg_agent       ) -> List[Dict[str, Any]]:

        refined_results = copy.deepcopy(batch_results)

        bad_idx = self._find_first_invalid(batch_results)
        if bad_idx is None:            # 全部通过
            for r in refined_results:
                r.setdefault("panel_decision", "Passed-window")
            return refined_results

        # 需要回溯：从 bad_idx 开始逐帧修正
        for i in range(bad_idx, len(batch_results)):
            parsed   = refined_results[i]
            messages = batch_messages[i]

            fixed = self._retry_until_valid(parsed,
                                           messages,
                                           memory_summary,
                                           surg_agent)
            refined_results[i] = fixed
        return refined_results

    # =========  判定 / 回溯  ========= #
    def _find_first_invalid(self, batch_results: List[Dict]) -> int | None:
        """让 judge_client 判定窗口内第一个不合理的帧索引；若全通过返回 None"""
        messages = self.base_messages + [{
            "role": "user",
            "content": self._build_batch_validation_prompt(batch_results)
        }]

        try:
            resp = self.judge_client.chat_completions(
                model="o3-mini",
                messages=messages,
                # temperature=0.0,
                max_tokens=10,
            )
            ans = resp.strip().upper()

            if ans == "OK"  or ans == "OKAY":
                return None
            for w in ans.split():
                if w.isdigit():
                    return int(w)
            return 0
        except Exception as e:
            print(f"[judge_client error] {e}")
            return 0

    def _retry_until_valid(self,
                           parsed_result,
                           local_messages,
                           memory_summary,
                           surg_agent):
        """最多 max_retries 次重新生成"""
        checked = self._check_and_fsm_validate(parsed_result)
        if checked["panel_decision"].startswith("Passed"):
            return checked

        for _ in range(self.max_retries):
            regen = self._retry_generation(local_messages,
                                           memory_summary,
                                           surg_agent)
            checked = self._check_and_fsm_validate(regen)
            if checked["panel_decision"].startswith("Passed"):
                return checked

        checked["panel_decision"] = "Rejected => window retries exhausted"
        return checked

    # =========  prompt 构造 ========= #
    def _build_batch_validation_prompt(self, batch_results) -> str:
        lines = [
            "以下是最近一个 key‑frame 窗口内连续帧的解析结果。",
            "窗口允许 **至多一次** 场景变化（通常发生在索引 2 处）。",
            "若符合规则输出 OK；否则输出第一个不合理帧索引及原因。"
        ]
        for idx, item in enumerate(batch_results):
            lines.append(f"\n--- Frame {idx} ---")
            lines.append(f"triplets: {json.dumps(item.get('triplets', []), ensure_ascii=False)}")
        return "\n".join(lines)

    # =========  格式 + FSM 校验 ========= #
    def _check_and_fsm_validate(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        tri  = parsed_result.get("triplets", [])
        if not isinstance(tri, list):
            return {"triplets": [], "analysis": "Format error", "panel_decision": "Rejected => Format"}

        # TODO: 这里可按需要加更细的 FSM 校验规则
        return {**parsed_result, "panel_decision": "Passed => basic check"}

    # =========  回溯生成 ========= #
    def _retry_generation(self,
                          original_messages: List[Dict],
                          memory_summary:   str,
                          surg_agent):
        """调用 SurgicalAgent 重新生成"""
        if surg_agent is None:
            return {"triplets": [], "analysis": "(no agent)", "panel_decision": "Rejected => no agent"}

        new_msgs = copy.deepcopy(original_messages)
        if memory_summary:
            new_msgs.append({"role": "system",
                             "content": f"<MEMORY>\n{memory_summary}\n</MEMORY>"})
        new_msgs.append({
            "role": "system",
            "content": (
                "<FSM>\n" + self.fsm_prompt + "\n</FSM>\n"
                "你之前的输出不符合一致性要求，请重新给出 **合法 JSON**。"
            )
        })

        raw = surg_agent.analyze_frame(new_msgs)
        try:
            parsed = json.loads(raw.strip())
        except Exception:
            parsed = {"triplets": [], "analysis": "(Bad JSON)"}
        return parsed
