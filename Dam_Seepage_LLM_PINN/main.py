import os
import sys

# 确保 Python 解释器能找到 src 目录下的包
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.agents.agent1_vision import run_vision_extraction
from src.agents.agent2_physics import run_physics_validation
from src.pinn.train import train_iterative_pinn

# ================= 配置区 =================
API_KEY = os.environ.get("GEMINI_API_KEY", "")
IMAGE_NAME = "dam_sketch4.jpg"
# ==========================================

def main():
    print("="*60)
    print("🌊 基于大模型与多智能体的 PINN 渗流自动求解系统启动 🌊")
    print("="*60)

    # ---------------------------------------------------------
    # 阶段 1：Agent 1 视觉提取
    # ---------------------------------------------------------
    print("\n>>> [阶段 1/3] 启动视觉感知智能体...")
    try:
        raw_data = run_vision_extraction(API_KEY, IMAGE_NAME)
    except Exception as e:
        print(f"❌ 阶段 1 失败: {e}")
        return

    # ---------------------------------------------------------
    # 阶段 2：Agent 2 物理校验与参数重构
    # ---------------------------------------------------------
    print("\n>>> [阶段 2/3] 启动物理逻辑自洽智能体...")
    try:
        # 修改点在这里：把 IMAGE_NAME 传进去
        physics_data = run_physics_validation(API_KEY, IMAGE_NAME) 
        if physics_data["status"] != "success":
            print("🛑 智能体判定几何参数存在严重物理冲突，已拦截！系统安全退出。")
            return
    except Exception as e:
        print(f"❌ 阶段 2 失败: {e}")
        return

    # ---------------------------------------------------------
    # 阶段 3：PINN 求解与可视化
    # ---------------------------------------------------------
    print("\n>>> [阶段 3/3] 启动 PINN 物理信息神经网络求解器...")
    try:
        train_iterative_pinn()
    except Exception as e:
        print(f"❌ 阶段 3 失败: {e}")
        return

    print("\n" + "="*60)
    print("✅ 全系统自动化流水线执行完毕，感谢使用！")
    print("="*60)

if __name__ == "__main__":
    main()