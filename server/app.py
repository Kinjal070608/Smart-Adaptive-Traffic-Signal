"""
server/app.py – OpenEnv-compliant server entry point.

openenv validate requires:
  - A [project.scripts] entry in pyproject.toml pointing here
  - A main() function that starts the ASGI server
"""
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from smart_traffic_signal.env import SmartAdaptiveTrafficSignalEnv
from smart_traffic_signal.schemas import TrafficAction

app = FastAPI(title="Smart Adaptive Traffic Signal", version="0.1.0")

_env_store: Dict[str, SmartAdaptiveTrafficSignalEnv] = {}


class ResetRequest(BaseModel):
    task_name: str = "easy"
    seed: int = 0


class StepRequest(BaseModel):
    action: TrafficAction
    task_name: str = "easy"


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "project": "Smart Adaptive Traffic Signal"}


@app.post("/reset")
async def reset(request: ResetRequest = None) -> Dict[str, object]:
    if request is None:
        request = ResetRequest()
    env = SmartAdaptiveTrafficSignalEnv(task_name=request.task_name, seed=request.seed)
    _env_store[request.task_name] = env
    observation = env.reset()
    return {
        "observation": observation.model_dump(),
        "reward": None,
        "done": False,
        "info": {"task": request.task_name},
    }


@app.get("/reset")
async def reset_get(task_name: str = "easy", seed: int = 0) -> Dict[str, object]:
    return await reset(ResetRequest(task_name=task_name, seed=seed))


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, object]:
    env = _env_store.get(request.task_name)
    if env is None:
        env = SmartAdaptiveTrafficSignalEnv(task_name=request.task_name, seed=0)
        _env_store[request.task_name] = env
        env.reset()
    observation, reward, done, info = env.step(request.action)
    info["metrics"] = env.get_metrics()
    info["score"] = env.evaluate()
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state(task_name: str = "easy") -> Dict[str, object]:
    env = _env_store.get(task_name)
    if env is None:
        env = SmartAdaptiveTrafficSignalEnv(task_name=task_name, seed=0)
        _env_store[task_name] = env
        env.reset()
    return {"state": env.state()}


def main() -> None:
    """Entry point invoked by the [project.scripts] console script."""
    import uvicorn

    # Mount Gradio UI at root if environment flag is set
    import os
    if os.getenv("ENABLE_WEB_INTERFACE", "false").lower() == "true":
        try:
            import gradio as gr
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
            from app import run_demo  # root-level app.py

            demo = run_demo()
            mounted_app = gr.mount_gradio_app(app, demo, path="/")
            uvicorn.run(mounted_app, host="0.0.0.0", port=7860)
        except Exception:
            uvicorn.run(app, host="0.0.0.0", port=7860)
    else:
        uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
