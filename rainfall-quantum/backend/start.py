"""
短时降雨量预测 - FastAPI 启动脚本

用法:
  python backend/start.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
