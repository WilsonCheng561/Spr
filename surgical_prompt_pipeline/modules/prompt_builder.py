# prompt_utils.py
from typing import List, Dict

def build_pdf_prompt() -> str:

    pdf_prompt = """
    #阅读这些论文，思考提到的外科手术动作三元组检测（triplet detection），学习胆囊切除手术的几个关键的步骤阶段，涉及到的手术工具，涉及到的解剖部位（目标），和相应的行动
    # https://wilsoncheng561.github.io/surgical-images/Encoding%20Surgical%20.pdf
    # https://arxiv.org/abs/2302.06294?utm_source=chatgpt.com
    # https://surg.dxy.cn/article/514242
    # https://zhuanlan.zhihu.com/p/694429681
    
    """
    return pdf_prompt

def build_text_prompt1() -> str:

    text_prompt = """
    你是腹腔镜胆囊切除手术的专家, 很擅长准确地识别手术阶段三元组，基于这些手术视频帧请执行以下任务：
    **识别手术器械**（仅限于以下选项）：  
    - "Grasper"
    - "Bipolar"（形如这样https://wilsoncheng561.github.io/surgical-images/video16/0055600.jpg）  
    - "Hook"（电钩）  
    - "Scissors" （形如这样https://wilsoncheng561.github.io/surgical-images/video17/0016975.jpg 和 https://wilsoncheng561.github.io/surgical-images/video02/0000975.jpg）  
    - "Clipper" （形如这样https://wilsoncheng561.github.io/surgical-images/video26/0018750.jpg 和 https://wilsoncheng561.github.io/surgical-images/video36/0028050.jpg）   
    - "Irrigator" （形如这样https://wilsoncheng561.github.io/surgical-images/video29/0054450.jpg）    

    **识别目标**（仅限于以下选项）：  
    - "Gallbladder"（胆囊）  
    - "Cystic Duct"（胆囊管）  
    - "Blood Vessel"（血管）  
    - "Specimen Bag"（样本袋）
    - "Liver"（肝脏）  
    - "Fat"（脂肪）  

    **识别手术操作**  

    **生成手术三元组"triplets"和"analysis" 字段，用一句话描述手术场景或当前步骤，并返回严格的 JSON 格式的结果**：  
    - **格式示例**：
    ```json
    {
        "triplets": ["instrument","target","action"],
        "analysis": "....text...."
    }
    ```
    - **按照confidence从大到小排序triplet并最多存在三个结果，允许只有两个甚至一个结果**。
    """
    return text_prompt

def build_text_prompt2() -> str:

    text_prompt = """
    你是腹腔镜胆囊切除手术的专家。现在将看到一张只包含单个物体的图像，请从下列 8 个选项中识别该物体的类别，并**仅返回对应英文字符串**：
    **手术器械**（仅限于以下选项）：  
    - "Grasper"
    - "Bipolar"（形如这样https://wilsoncheng561.github.io/surgical-images/video16/0055600.jpg）  
    - "Hook"（电钩）  
    - "Scissors" （形如这样https://wilsoncheng561.github.io/surgical-images/video17/0016975.jpg 和 https://wilsoncheng561.github.io/surgical-images/video02/0000975.jpg）  
    - "Clipper" （形如这样https://wilsoncheng561.github.io/surgical-images/video26/0018750.jpg 和 https://wilsoncheng561.github.io/surgical-images/video36/0028050.jpg）   
    - "Irrigator" （形如这样https://wilsoncheng561.github.io/surgical-images/video29/0054450.jpg）    

    **目标**：  
    - "Gallbladder"（胆囊）    
    - "Specimen Bag"（样本袋）

    不要输出任何额外文字、标点或换行。
    """
    return text_prompt


def build_panel_prompt() -> str:

    FSM_PROMPT = """
    This is a high-level description of laparoscopic cholecystectomy (LC), intended to assist in reasoning about surgical scenes across different frames.

    The general surgical phases, common instruments, surgical targets, and possible actions should be inferred based on the information available from the following references:

    - https://surg.dxy.cn/article/514242
    - https://zhuanlan.zhihu.com/p/694429681

    Please note:
    - Instruments listed in any phase are **potential** tools and may not appear consistently.
    - Some tools like `Grasper` are used across many phases, while others like `Clipper` or `Scissors` may be used only briefly.
    - Surgical workflow is not strictly linear; transitions between phases may vary depending on visibility, bleeding, or anatomy.

    The purpose of this prompt is to provide background knowledge for identifying and interpreting instrument usage, surgical targets, and intended actions in laparoscopic cholecystectomy videos.
    """
    return FSM_PROMPT

def build_panel_prompt2() -> str:
    """
    新 Panel Prompt：强调“关键帧 = 手术场景转折点”，
    并给出窗口一致性/差异容忍度的判定规则。
    """
    FSM_PROMPT = """
    你是腹腔镜胆囊切除（LC）的术中质控专家。
    This is a high-level description of laparoscopic cholecystectomy (LC), intended to assist in reasoning about surgical scenes across different frames.

    The general surgical phases, common instruments, surgical targets, and possible actions should be inferred based on the information available from the following references:

    - https://surg.dxy.cn/article/514242
    - https://zhuanlan.zhihu.com/p/694429681

    ★ 关键概念
    1. **关键帧 (Key‑frame)**  
    - 由人工预先标注；  
    - 通常是手术场景发生“明显阶段/工具/目标切换”的第一帧。  
    2. **窗口 (Window)**  
    - 由关键帧向前后各取 2 帧，共 3‑5 张连续帧；  
    - 在一个窗口内，“场景转折”**最多出现一次**。  
        也就是说：  
        · 若关键帧前 1‑2 帧 & 关键帧内容基本一致 → 正常；  
        · 若关键帧后 1‑2 帧内容稳定在新场景 → 正常；  
        · 若窗口内出现多次跳变 → 可能是识别错误，需要返工。

    ★ 你的任务
    对每个窗口给出的 **解析结果 triplets / analysis**：
    1. **格式检查**：JSON 正确且字段齐全；  
    2. **一致性检查**：  
    - 统计窗口内每帧的 *instrument / target / action*；  
    - 允许“前半部分一致 + 后半部分一致 + 中间关键帧切换”这一种模式；  
    - 其它任何模式 (多次切换、随机抖动) 视为“不一致”。  
    3. **工具‑目标‑动作 合规**：结合胆囊切除流程与下表  
    （省略…保持原有 clipper / scissors 等规则）。

    输出要求：
    - 若全部通过 → 返回 `OK`。  
    - 若不通过 → 返回 **第一个不合理帧的索引 (0‑based)**，并简要说明原因。

    """
    return FSM_PROMPT


