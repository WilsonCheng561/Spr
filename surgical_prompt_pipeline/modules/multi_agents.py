from typing import List, Dict
from modules.utils import OpenAIClient 

class BaseAgent:
    """
    Agent的基类，内置一个coT_prompt方法或抽象方法
    """
    def __init__(self, name: str):
        self.name = name
    
    def generate_cot_reasoning(self, input_data: dict) -> str:
        """
        生成CoT推理过程
        """
        return f"<{self.name} reasoning>..."

    def get_result(self, input_data: dict) -> dict:
        # demo, 返回空
        return {"triplets": [], "analysis": f"{self.name} default analysis"}

class InstrumentAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="InstrumentAgent")

    def get_result(self, input_data: dict) -> dict:
        return {"triplets": ["Grasper", "Gallbladder", "Hold"], "analysis": "Instrument recognized as grasper."}

class ActionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ActionAgent")

    def get_result(self, input_data: dict) -> dict:
        # ...
        return {"triplets": ["Scissors", "Cystic Duct", "Cut"], "analysis": "Likely cutting the cystic duct."}

class StageAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="StageAgent")

    def get_result(self, input_data: dict) -> dict:
        # ...
        return {"triplets": ["Hook", "Blood Vessel", "Coagulate"], "analysis": "Stage: Hemostasis."}

    
class SurgicalAgent(OpenAIClient):

    def __init__(self, openai_client: OpenAIClient = None):
        super().__init__()  # 用默认的 api_key 和 model

    def analyze_frame(self, local_messages: List[Dict[str, any]]) -> str:
        # 对图像帧执行分析任务，调用 GPT 模型生成 triplet 输出
        # 后面可以加Chain-of-Thought Prompt, <SUMMARY><CAPTION><REASONING><CONCLUSION>等

        response_text = self.chat_completions(local_messages)
        return response_text



class AgentManager:
    """
    同时管理多个Agent，集中调度并汇总结果。
    """
    def __init__(self):
        self.instrument_agent = InstrumentAgent()
        self.action_agent = ActionAgent()
        self.stage_agent = StageAgent()
    
    def process(self, input_data: dict) -> dict:
        """
        依次调用各Agent, 也可并行
        """
        results = {}
        inst_res = self.instrument_agent.get_result(input_data)
        action_res = self.action_agent.get_result(input_data)
        stage_res = self.stage_agent.get_result(input_data)
        # ...
        results["instrument"] = inst_res
        results["action"] = action_res
        results["stage"] = stage_res
        return results
