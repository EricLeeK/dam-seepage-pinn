# 启动指南

## 1. 启动后端

cd webapp/backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

## 2. 启动前端 (新终端)

cd webapp/frontend
npm install   # 首次运行
npm run dev

## 3. 打开浏览器

http://localhost:5173

## 4. API Key (草图识别模式)

默认使用硬编码 key。如失效，在前端界面上传草图时会提示输入，
或设置环境变量:

export DASHSCOPE_API_KEY=your_key_here
