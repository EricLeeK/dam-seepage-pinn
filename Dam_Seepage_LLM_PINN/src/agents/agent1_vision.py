import os
import json
from pathlib import Path
from google import genai
from google.genai import types

# 动态定位项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 输出文件路径保持固定
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "outputs", "agent1_raw_data.json")

def run_vision_extraction(api_key: str, image_name: str, image_path: str = None):
    """
    Agent 1 的主执行函数：调用多模态大模型提取草图参数

    Args:
        api_key: API key for the vision model.
        image_name: Name of the image file.
        image_path: Full path to the image file. If None, uses default data/inputs/{image_name}.
    """
    # 动态拼装图片路径
    if image_path is None:
        image_path = os.path.join(PROJECT_ROOT, "data", "inputs", image_name)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 找不到图片文件 {image_path}，请确认 data/inputs 目录下有该文件！")

    print(f"🚀 [Agent 1] 正在识别图片: {image_name}")

    client = genai.Client(api_key=api_key)

    agent1_prompt = """
你是一个专业的水利工程图像识别智能体 (Agent 1)。你的唯一职责是从草图中精确提取标注的几何参数和总水头，不要进行任何几何计算或逻辑推演。

【大坝横截面的几何约定】：
- 大坝是一个梯形（上窄下宽），底部贴着水平地面。
- 所有角度一律从【水平地面线】起算，向坝体斜面方向旋转。
- 上游坡角 (upstream_slope_angle)：从左侧水平地面线，逆时针旋转到上游斜面的角度。
- 下游坡角 (downstream_slope_angle)：从右侧水平地面线，顺时针旋转到下游斜面的角度。
- 举例：如果上游斜面与水平地面成 60°，则 upstream_slope_angle = 60（不是 120）。
- 两个坡角的范围都是 0°~90°，且斜面一定从底边向内收缩到坝顶（即梯形，不是平行四边形）。

【必须输出 JSON】：
1. 几何形状 (geometry_shape，字符串)
2. 标注的大坝高度 (dam_height)
3. 标注的坝顶宽度 (top_width)
4. 标注的坝底宽度 (bottom_width)
5. 标注的上游坡角 (upstream_slope_angle)
6. 标注的下游坡角 (downstream_slope_angle)
7. 标注的上游总水头 (upstream_total_head)
8. 标注的下游总水头 (downstream_total_head)

【严格要求】：
1. 除了 geometry_shape，所有数值必须是"纯数字"（Float 类型），绝不包含单位（如"m", "米", "度"）。
2. 如果图片中某个数据缺失或模糊，请返回 null。
3. 如果图片中标注的角度是从竖直方向量的（即与铅垂线的夹角），请自动转换为与水平地面的夹角（即 90° 减去标注值）。
"""

    # 读取图片为 bytes
    image_bytes = Path(image_path).read_bytes()
    # 根据扩展名推断 mime_type
    ext = Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=[
            image_part,
            "提取大坝标注参数，剥离单位，只保留纯数字。"
        ],
        config=types.GenerateContentConfig(
            system_instruction=agent1_prompt,
            response_mime_type="application/json",
        ),
    )

    result = response.text
    print("✅ [Agent 1] 提取完成！")
    print(result)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 将结果保存到本地文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"📁 原始数据已成功保存至：{OUTPUT_FILE}，准备交给 Agent 2 检验物理逻辑。")

    return json.loads(result)

# 如果单独运行此脚本用于测试
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    TEST_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    TEST_IMAGE = "dam_sketch.jpg"
    try:
        run_vision_extraction(TEST_API_KEY, TEST_IMAGE)
    except Exception as e:
        print(f"Agent 1 运行失败: {e}")
