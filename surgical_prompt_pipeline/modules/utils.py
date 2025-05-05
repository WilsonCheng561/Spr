import os
import json
from openai import OpenAI
import openai

OPENAI_API_KEY = "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"
DEFAULT_MODEL = "gpt-4o"


class OpenAIClient:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or OPENAI_API_KEY
        self.default_model = model
        self.client = OpenAI(api_key=self.api_key)

    def chat_completions(
        self,
        messages,
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 800
    ):
        model_to_use = model or self.default_model

        try:
            if model_to_use.startswith("o3") or model_to_use.endswith("preview"):
                # o3 series models use max_completion_tokens
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_completion_tokens=max_tokens   
                )
            else:
                # gpt-3.5, gpt-4, gpt-4o use max_tokens
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"❌ ChatCompletion 调用失败（模型: {model_to_use}）: {e}")
            raise

    def list_valid_models(self):
        models = self.client.models.list()
        for m in models.data:
            print("Here are the available models:")
            print(m.id)



def strip_JSON(response):
    """清理 GPT 生成的 JSON 响应，确保格式正确"""
    result = response.strip("```").strip("json").strip()
    
    if not result:  # 处理空响应
        print("⚠️ GPT 返回空响应，跳过此帧")
        return {"triplets": [], "analysis": "No valid response"}

    try:
        return json.loads(result)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}, 响应内容: {result}")
        return {"triplets": [], "analysis": "Invalid JSON format"}

def extract_frame_info(image_url):
    """从 URL 提取 `frame_file` 和 `frame_id`"""
    frame_file = os.path.basename(image_url)  # 提取文件名
    frame_id = int(os.path.splitext(frame_file)[0]) // 25  # 计算 `frame_id`
    return frame_id, frame_file