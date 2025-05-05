# memory_agent.py
from __future__ import annotations
from typing import List, Dict, Tuple


class MemoryAgent:
    """
    1. 负责跨帧记忆 (history_buffer)
    2. 内部维护 “滑动窗口” (pending) ，每 window_size 帧自动调用 PanelDiscussion
    """

    def __init__(self,
                 panel,                # PanelDiscussion 实例
                 surg_agent,           # SurgicalAgent  (回溯时需要)
                 max_history: int = 5,
                 window_size: int = 5):
        self.panel = panel
        self.surg_agent = surg_agent

        self.max_history = max_history        # 存储最近多少帧的历史结果
        self.window_size = window_size        # 满多少帧触发一次 panel

        self.history_buffer: List[Tuple[int, Dict]] = []  # [(frame_id, refined)]
        self.pending_results: List[Tuple[int, Dict]] = [] # [(frame_id, parsed)]
        self.pending_messages: List[List[Dict]] = []      # messages 列表

    # ------------------------------------------------------------------
    # =============  提供给 process_video 的 API  =======================
    # ------------------------------------------------------------------
    def add_parsed(self,
                   frame_id: int,
                   parsed_result: Dict,
                   local_messages: List[Dict]) -> None:
        """把【本帧解析结果】缓存进 pending，等待批量 panel"""
        self.pending_results.append((frame_id, parsed_result))
        self.pending_messages.append(local_messages)

    def flush_if_needed(self,
                        last_frame: bool = False) -> List[Tuple[int, Dict]]:
        """
        满 window_size 或到最后一帧时调用。
        返回 [(frame_id, refined_result), ...]，否则返回 []。
        """
        if (len(self.pending_results) < self.window_size) and not last_frame:
            return []

        if not self.pending_results:          # 没东西
            return []

        # --- 批量 panel ---
        parsed_batch   = [x[1] for x in self.pending_results]
        message_batch  = self.pending_messages

        refined_batch = self.panel.batch_evaluate_and_refine(
            parsed_batch,
            message_batch,
            self.get_near_history_summary(),
            self.surg_agent
        )

        # 对应写入 history_buffer
        refined_with_id = []
        for (fid, _), refined in zip(self.pending_results, refined_batch):
            self.update_memory(fid, refined)          # 进入历史
            refined_with_id.append((fid, refined))

        # 清空窗口
        self.pending_results.clear()
        self.pending_messages.clear()

        return refined_with_id

    # ------------------------------------------------------------------
    # =======================  内部函数  ================================
    # ------------------------------------------------------------------
    def update_memory(self, frame_id: int, refined_result: Dict):
        """把 panel 校正后的结果写入历史"""
        self.history_buffer.append((frame_id, refined_result))
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)

    def get_near_history_summary(self) -> str:
        """返回最近若干帧的摘要文本，插入 prompt"""
        summary = []
        for fid, res in self.history_buffer:
            tri = res.get("triplets", [])
            analysis = res.get("analysis", "")
            summary.append(f"Frame={fid}, triplets={tri}, analysis={analysis}")
        return "\n".join(summary)
