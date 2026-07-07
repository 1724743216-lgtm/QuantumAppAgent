"""FastAPI backend for VQE H2 application."""

import json
import sys
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on path for vqe_solver import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vqe_solver import (
    run_vqe,
    exact_diagonalization,
    H2_HAMILTONIAN,
    hardware_efficient_ansatz,
)

app = FastAPI(title="VQE H2 App", version="1.0.0")

# Load pre-computed reports at module level (validator requirement)
REPORTS_DIR = PROJECT_ROOT

def _load_json(name):
    p = REPORTS_DIR / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


_baseline_report = _load_json("baseline_report.json")
_quantum_report = _load_json("quantum_report.json")


@app.get("/api/results", response_class=JSONResponse)
async def get_results():
    """Return VQE and baseline results."""
    return {
        "baseline": _baseline_report,
        "quantum": _quantum_report,
    }


@app.get("/api/run", response_class=JSONResponse)
async def run_new_vqe(layers: int = 2, max_iter: int = 200, seed: int = 42):
    """Run a new VQE optimization with given parameters."""
    try:
        vqe_result = run_vqe(
            hamiltonian=H2_HAMILTONIAN,
            n_qubits=2,
            layers=layers,
            max_iter=max_iter,
            seed=seed,
        )
        exact = exact_diagonalization(H2_HAMILTONIAN)
        return {
            "status": "ok",
            "vqe_energy": vqe_result["optimal_energy"],
            "exact_energy": exact,
            "error": abs(vqe_result["optimal_energy"] - exact),
            "iterations": vqe_result["iterations"],
            "convergence_trace": vqe_result["convergence_trace"],
            "param_count": vqe_result["param_count"],
            "layers": layers,
            "seed": seed,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/hamiltonian", response_class=JSONResponse)
async def get_hamiltonian():
    """Return H2 Hamiltonian terms."""
    terms = []
    for coeff, ops in H2_HAMILTONIAN:
        label = "I" if not ops else " ".join(f"{op}{q}" for q, op in ops)
        terms.append({"coefficient": coeff, "term": label})
    return {"hamiltonian": terms, "n_qubits": 2}


# Serve frontend
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend page."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>VQE H2 App</h1><p>Frontend not found.</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
