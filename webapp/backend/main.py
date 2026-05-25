"""FastAPI backend for Dam Seepage PINN web application."""

import os
import sys
import uuid
import json
import base64
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── sys.path setup so backend can import from Dam_Seepage_LLM_PINN/src/ ──
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent.parent  # AI教学案例申报-PINN/
_PINN_SRC = _PROJECT_ROOT / "Dam_Seepage_LLM_PINN"
if str(_PINN_SRC) not in sys.path:
    sys.path.append(str(_PINN_SRC))

from src.agents.agent1_vision import run_vision_extraction
from src.agents.agent2_physics import run_physics_validation
from src.pinn.train import train_pinn, TrainingCancelled

# ── Ensure results directory exists ──
RESULTS_DIR = _BACKEND_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory task store (replace with Redis in production) ──
_task_store = {}
_task_lock = threading.Lock()

# ── FastAPI app ──
app = FastAPI(title="Dam Seepage PINN API", version="0.1.0")

# ── CORS: allow Vite dev server ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════

class GeometryConfig(BaseModel):
    shape: str = "trapezoid"
    vertices: list[list[float]]
    h: float
    b_top: float
    b_bottom: float
    angle_up: float
    angle_down: float


class HydraulicConfig(BaseModel):
    upstream_head: float
    downstream_head: float
    permeability_k: float = 1.0


class TrainingConfig(BaseModel):
    outer_iters: int = 30
    num_domain: int = 10000
    num_boundary: int = 2000
    adam_epochs: int = 1000
    lbfgs_max_iter: int = 5000


class SolveRequest(BaseModel):
    geometry: GeometryConfig
    hydraulic: HydraulicConfig
    training: TrainingConfig


class AgentAnalyzeResponse(BaseModel):
    status: str
    agent1_data: dict
    agent2_data: dict
    geometry: Optional[GeometryConfig] = None
    validation_report: str


# ═══════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════

def _build_config_dict(req: SolveRequest) -> dict:
    """Build unified config dict from frontend request."""
    return {
        "geometry": req.geometry.model_dump(),
        "hydraulic": req.hydraulic.model_dump(),
        "training": req.training.model_dump(),
    }


def _run_pinn_training(task_id: str, config_dict: dict):
    """Run PINN training in background thread."""
    output_dir = RESULTS_DIR / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable phase labels
    _PHASE_NAMES = {
        "outer_iter": "外部空间迭代",
        "adam": "Adam 预研",
        "lbfgs_inner": "L-BFGS 内层收敛",
        "fs_update": "浸润线更新",
        "lbfgs_final": "L-BFGS 终极压榨",
    }

    def progress_callback(phase: str, current: int, total: int, loss_val: Optional[float]):
        phase_cn = _PHASE_NAMES.get(phase, phase)
        loss_str = f"loss={loss_val:.4e}" if loss_val is not None else ""
        msg = f"[{phase_cn}] {current}/{total} {loss_str}"
        print(f"  📊 {msg}", flush=True)

        with _task_lock:
            _task_store[task_id]["progress"].append({
                "phase": phase,
                "current": current,
                "total": total,
                "loss": loss_val,
                "message": msg,
                "timestamp": datetime.now().isoformat(),
            })

    def check_cancelled():
        with _task_lock:
            return _task_store[task_id].get("cancelled", False)

    try:
        with _task_lock:
            _task_store[task_id]["status"] = "running"

        result = train_pinn(config_dict, str(output_dir),
                            progress_callback=progress_callback,
                            cancel_check=check_cancelled)

        with _task_lock:
            _task_store[task_id]["status"] = "completed"
            _task_store[task_id]["result"] = {
                "final_loss": result["final_loss"],
                "plot_url": f"/api/pinn/result/{task_id}/plot",
                "npz_url": f"/api/pinn/result/{task_id}/npz",
            }

    except TrainingCancelled:
        with _task_lock:
            _task_store[task_id]["status"] = "cancelled"
            _task_store[task_id]["error"] = "用户手动取消"

    except Exception as e:
        import traceback
        with _task_lock:
            _task_store[task_id]["status"] = "failed"
            _task_store[task_id]["error"] = str(e)
            _task_store[task_id]["traceback"] = traceback.format_exc()


# ═══════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════
# Agent Analysis (Sketch Recognition)
# ═══════════════════════════════════════════════════════════

@app.post("/api/agent/analyze")
async def agent_analyze(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None),
):
    """
    Upload a dam sketch image, run Agent 1 + Agent 2 pipeline.
    Returns extracted parameters and validation report.
    """
    # ── API key handling: request body → env var → 400 error ──
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "api_key is required: provide in request body or set GEMINI_API_KEY env var"}
        )

    # Save uploaded image temporarily
    upload_dir = _BACKEND_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / file.filename

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Run Agent 1: vision extraction (pass explicit image_path for uploaded file)
        agent1_data = run_vision_extraction(resolved_key, file.filename, image_path=str(temp_path))

        # Save raw data for Agent 2
        raw_path = _BACKEND_DIR / "data" / "outputs" / "agent1_raw_data.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(agent1_data, f, ensure_ascii=False, indent=2)

        # Run Agent 2: physics validation (pass explicit image_path for uploaded file)
        agent2_data = run_physics_validation(resolved_key, file.filename, image_path=str(temp_path))

        # Build response with extracted parameters and validation report
        response = {
            "status": agent2_data.get("status", "error"),
            "agent1_data": agent1_data,
            "agent2_data": agent2_data,
            "validation_report": agent2_data.get("check_report", ""),
        }

        # Include geometry if validation passed
        if agent2_data.get("status") == "success":
            domain = agent2_data.get("pinn_domain", {})
            vertices = domain.get("vertices", [])
            if vertices:
                h = agent1_data.get("dam_height", 0)
                b_top = agent1_data.get("top_width", 0)
                b_bottom = agent1_data.get("bottom_width", 0)
                response["geometry"] = {
                    "shape": agent1_data.get("geometry_shape", "trapezoid"),
                    "vertices": vertices,
                    "h": h,
                    "b_top": b_top,
                    "b_bottom": b_bottom,
                    "angle_up": agent1_data.get("upstream_slope_angle", 0),
                    "angle_down": agent1_data.get("downstream_slope_angle", 0),
                }
                response["hydraulic"] = {
                    "upstream_head": domain.get("upstream_head", 0),
                    "downstream_head": domain.get("downstream_head", 0),
                    "permeability_k": 1.0,
                }

        return response

    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        )
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


# ═══════════════════════════════════════════════════════════
# PINN Solver
# ═══════════════════════════════════════════════════════════

@app.post("/api/pinn/solve")
async def pinn_solve(request: SolveRequest, background_tasks: BackgroundTasks):
    """
    Start PINN training asynchronously.
    Returns task_id for polling status.
    """
    task_id = str(uuid.uuid4())
    config_dict = _build_config_dict(request)

    with _task_lock:
        _task_store[task_id] = {
            "status": "queued",
            "cancelled": False,
            "progress": [],
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
        }

    # Run training in background thread
    thread = threading.Thread(
        target=_run_pinn_training,
        args=(task_id, config_dict),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id, "status": "queued"}


@app.get("/api/pinn/status/{task_id}")
def pinn_status(task_id: str):
    """Get training status and progress."""
    with _task_lock:
        task = _task_store.get(task_id)

    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    latest = task["progress"][-1] if task["progress"] else None

    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "latest": latest,
        "progress_count": len(task["progress"]),
        "result": task.get("result"),
        "error": task.get("error"),
    }


@app.post("/api/pinn/cancel/{task_id}")
def pinn_cancel(task_id: str):
    """Cancel a running training task."""
    with _task_lock:
        task = _task_store.get(task_id)
        if not task:
            return JSONResponse(status_code=404, content={"error": "Task not found"})
        if task["status"] not in ("queued", "running"):
            return JSONResponse(status_code=400, content={"error": f"Task is {task['status']}, cannot cancel"})
        task["cancelled"] = True
    print(f"🛑 Cancel requested for task {task_id}", flush=True)
    return {"status": "cancelling"}


@app.get("/api/pinn/result/{task_id}")
def pinn_result(task_id: str):
    """Get training result URLs (image + NPZ data). Returns 404 if not ready."""
    with _task_lock:
        task = _task_store.get(task_id)

    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    if task["status"] != "completed":
        return JSONResponse(status_code=404, content={"error": "Result not ready", "status": task["status"]})

    result_dir = RESULTS_DIR / task_id
    plot_path = result_dir / "pinn_seepage_result.png"
    npz_path = result_dir / "seepage_plot_data.npz"

    if not plot_path.exists() or not npz_path.exists():
        return JSONResponse(status_code=404, content={"error": "Result files not found"})

    return {
        "task_id": task_id,
        "status": "completed",
        "plot_url": f"/api/pinn/result/{task_id}/plot",
        "npz_url": f"/api/pinn/result/{task_id}/npz",
        "final_loss": task.get("result", {}).get("final_loss"),
    }


@app.get("/api/pinn/result/{task_id}/plot")
def pinn_result_plot(task_id: str):
    """Get result PNG image."""
    plot_path = RESULTS_DIR / task_id / "pinn_seepage_result.png"
    if not plot_path.exists():
        return JSONResponse(status_code=404, content={"error": "Result not ready or not found"})
    return FileResponse(plot_path, media_type="image/png", filename="pinn_seepage_result.png")


@app.get("/api/pinn/result/{task_id}/npz")
def pinn_result_npz(task_id: str):
    """Get result NPZ data file."""
    npz_path = RESULTS_DIR / task_id / "seepage_plot_data.npz"
    if not npz_path.exists():
        return JSONResponse(status_code=404, content={"error": "Result not ready or not found"})
    return FileResponse(npz_path, media_type="application/octet-stream", filename="seepage_plot_data.npz")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
