import os
import json
import math
from pathlib import Path
from google import genai
from google.genai import types

# 定位家谱：确保程序知道自己在哪个文件夹，数据在哪
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "outputs", "agent1_raw_data.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "outputs", "pinn_domain_config.json")

def calculate_vertices_precise(h, b_top, b_bottom_anno, angle_up, angle_down):
    """
    【数学核心】：放弃 AI 的感性，使用初中几何公式进行精准计算
    """
    # 角度转弧度（Python 数学库的要求）
    rad_up = math.radians(angle_up)
    rad_down = math.radians(angle_down)

    # 根据三角函数计算左右两边的"坡脚位移" (Offset)
    x_offset_up = h / math.tan(rad_up)
    x_offset_down = h / math.tan(rad_down)

    # 【精密推演】：根据高度、顶宽和坡角，计算出"理论上"底宽应该是多少
    b_bottom_theoretical = x_offset_up + b_top + x_offset_down

    # 【生成坐标系】：以左下角为 (0,0) 点
    v1 = [0.0, 0.0]
    v2 = [b_bottom_theoretical, 0.0] # 这里使用的是精密计算的底宽，而非 AI 提取的数字
    v3 = [x_offset_up, h]
    v4 = [x_offset_up + b_top, h]

    return [v1, v2, v3, v4], b_bottom_theoretical

def run_physics_validation(api_key: str, image_name: str, image_path: str = None):
    """
    Agent 2 的主执行函数：进行视觉复核与物理逻辑审计

    Args:
        api_key: API key for the vision model.
        image_name: Name of the image file.
        image_path: Full path to the image file. If None, uses default data/inputs/{image_name}.
    """
    print("🚀 [Agent 2] 启动：正在进行视觉复核与物理逻辑审计...")

    # 检查 Agent 1 的作业写完没
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ 找不到 {INPUT_FILE}，请先运行 Agent 1！")

    if image_path is None:
        image_path = os.path.join(PROJECT_ROOT, "data", "inputs", image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 找不到图片文件 {image_path}！")

    # 读取 Agent 1 提取的原始数据
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        agent1_raw_str = f.read()

    # 读取图片为 bytes
    image_bytes = Path(image_path).read_bytes()
    ext = Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    client = genai.Client(api_key=api_key)

    # --- 步骤 1：Agent 2 升级为视觉审计官，进行"交叉复核" ---
    agent2_prompt = f"""
    你是一个高级水利工程审计智能体 (Agent 2)。

    【你的任务】：
    1. 视觉复核：请仔细查阅提供的原始草图，并核对 Agent 1 提取的初步数据。
       Agent 1 的初步数据如下：
       {agent1_raw_str}

       请检查 Agent 1 提取的数值是否与图片标注完全一致。如果 Agent 1 漏提、提错，请你纠正它。

    2. 格式规范化：将你最终确认正确的数据，整理为以下严格的 JSON 格式输出，确保所有值为纯数字（Float）：
       {{"h": 高度, "b_top": 顶宽, "b_bottom": 底宽, "angle_up": 上游角, "angle_down": 下游角, "h_up": 上游水头, "h_down": 下游水头}}

    请只输出 JSON 对象，不要包含任何额外的解释文本。
    """

    # 使用 gemini-3.1-flash-lite-preview，传入图片和 Prompt
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=[
            image_part,
            agent2_prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    # 解析复核并整理后的数据
    data = json.loads(response.text)
    print("🔍 [Agent 2] 视觉复核与数据清洗完成！最终采信数据：", data)

    # --- 步骤 2：Python 亲自下场，进行精密数学计算（保持原样） ---
    vertices, b_theo = calculate_vertices_precise(
        data['h'], data['b_top'], data['b_bottom'], data['angle_up'], data['angle_down']
    )

    # 计算误差：理论底宽 vs 最终确认的图片标注底宽
    error_rel = abs(data['b_bottom'] - b_theo) / b_theo * 100

    # --- 步骤 3：判定并输出最终 JSON ---
    final_result = {
        "status": "pending",
        "precision_data": None,
        "check_report": ""
    }

    if error_rel < 0.5: # 误差小于 0.5% 视为合格
        print(f"✅ 几何自洽性通过！误差仅为 {error_rel:.4f}%")
        final_result["status"] = "success"
        final_result["check_report"] = "视觉复核无误且几何逻辑完美，坐标已生成。"
        final_result["pinn_domain"] = {
            "vertices": [[round(c, 4) for c in v] for v in vertices],
            "upstream_head": data['h_up'],
            "downstream_head": data['h_down']
        }
    else:
        print(f"❌ 几何冲突！理论底宽应为 {b_theo:.2f}，但图片标注为 {data['b_bottom']}")
        final_result["status"] = "error"
        final_result["check_report"] = f"数据冲突：图片标注底宽与坡角推算结果不符，误差 {error_rel:.2f}%"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    print(f"📁 精密配置已保存至：{OUTPUT_FILE}")
    return final_result

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    MY_KEY = os.environ.get("GEMINI_API_KEY", "")
    TEST_IMAGE = "dam_sketch.jpg"
    run_physics_validation(MY_KEY, TEST_IMAGE)
