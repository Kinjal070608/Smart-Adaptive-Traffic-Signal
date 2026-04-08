from typing import Dict

from fastapi import FastAPI, Body
from pydantic import BaseModel

from smart_traffic_signal.env import SmartAdaptiveTrafficSignalEnv
from smart_traffic_signal.schemas import TrafficAction, TrafficObservation, TrafficReward

app = FastAPI(title="Smart Adaptive Traffic Signal", version="0.1.0")

_env_store: Dict[str, SmartAdaptiveTrafficSignalEnv] = {}


class ResetRequest(BaseModel):
    task_name: str = "easy"
    seed: int = 0


class StepRequest(BaseModel):
    action: TrafficAction
    task_name: str = "easy"


@app.get("/")
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "project": "Smart Adaptive Traffic Signal"}


@app.post("/reset")
async def reset(request: ResetRequest = Body(...)) -> Dict[str, object]:
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


import gradio as gr
from app import run_demo

# Mount the Gradio UI at the root so it's still accessible!
demo = run_demo()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
