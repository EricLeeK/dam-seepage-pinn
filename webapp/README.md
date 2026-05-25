# 🌊 大坝渗流 PINN Web 应用

基于 React + FastAPI 的大坝渗流场分析系统。支持**手绘草图智能识别**和**交互式梯形编辑器**两种输入模式，后端调用物理信息神经网络（PINN）自动求解拉普拉斯方程。

---

## 项目结构

```
webapp/
├── backend/          # FastAPI 后端
│   ├── main.py       # API 路由 + Agent/PINN 调用
│   ├── requirements.txt
│   └── results/      # 训练结果存储（运行时生成）
└── frontend/         # React + Vite 前端
    ├── src/
    │   ├── App.jsx           # 主应用
    │   ├── main.jsx          # 入口
    │   └── components/
    │       └── CanvasEditor.jsx   # HTML5 Canvas 梯形编辑器
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## 快速启动

### 1. 后端

```bash
cd webapp/backend

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 前端

```bash
cd webapp/frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端默认运行在 `http://localhost:5173`，会自动代理 `/api` 请求到后端。

---

## 使用流程

### 模式一：交互绘制
1. 在左侧输入几何参数（高度、顶宽、坡角等）
2. 或在中间 Canvas 上**直接拖拽红色顶点**调整梯形形状
3. 输入水力参数（上下游水头）
4. 点击「▶️ 启动 PINN 求解」
5. 等待 2-5 分钟，右侧显示等势线图和浸润线

### 模式二：草图识别
1. 切换到「🖼️ 草图识别」模式
2. 上传大坝手绘草图（标注了几何参数的）
3. 点击「🚀 运行智能体分析」
4. Agent 1（Qwen-VL）自动提取参数 → Agent 2 校验几何自洽性
5. 校验通过后自动填充参数，可手动修正
6. 运行 PINN 求解

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/agent/analyze` | 上传草图，运行 Agent 1+2 |
| POST | `/api/pinn/solve` | 启动 PINN 训练，返回 task_id |
| GET | `/api/pinn/status/{task_id}` | 查询训练进度 |
| GET | `/api/pinn/result/{task_id}/plot` | 获取结果 PNG |
| GET | `/api/pinn/result/{task_id}/npz` | 获取结果 NPZ 数据 |

---

## 高级设置

点击左侧面板「⚙️ 高级设置」可调整：
- **渗透系数 K** — 达西定律参数（默认 1.0）
- **域内采样点数** — PINN 内部点数量
- **Adam/L-BFGS 迭代次数** — 训练超参数

---

## 技术栈

- **前端**: React 18 + Vite + HTML5 Canvas
- **后端**: FastAPI + Uvicorn
- **AI 模型**: Qwen-VL-Max（多模态识别）
- **物理求解**: PyTorch PINN（Float64 精度）
- **可视化**: Matplotlib
