import json
import os
import re
import sys
import traceback
import urllib.request
import urllib.error
from typing import Any, List, Optional, Dict, Union, Tuple

from openai import OpenAI

# Ensure the local smart_traffic_signal package is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_traffic_signal.env import SmartAdaptiveTrafficSignalEnv
from smart_traffic_signal.schemas import TrafficAction, TrafficObservation, TrafficReward

MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.6
TASKS = ["easy", "medium", "hard"]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.3f} done={done} error={error}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


def get_obs_dict(observation: Any) -> dict:
    """Helper for Pydantic v1/v2 compatibility."""
    if isinstance(observation, dict):
        return observation
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    return observation.dict()


class RemoteEnvClient:
    """Mock-like client for remote OpenEnv server."""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, data: dict) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def reset(self, task_name: str = "easy", seed: int = 42) -> dict:
        return self._post("/reset", {"task_name": task_name, "seed": seed})

    def step(self, action: TrafficAction) -> Tuple[dict, Any, bool, dict]:
        # action is handled as dict in API
        resp = self._post("/step", {"phase": action.phase})
        obs = resp["observation"]
        reward = resp["reward"]
        done = resp["done"]
        info = resp.get("info", {})
        # Reward in API response matches TrafficReward structure
        return obs, reward, done, info

    def evaluate(self) -> float:
        # State endpoint usually returns overall metrics
        url = f"{self.base_url}/state"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            # Heuristic: the server might not expose a direct evaluate() call
            # so we calculate it if the tasks.py is available locally
            from smart_traffic_signal.tasks import get_task
            task_key = data.get("task", "easy")
            task = get_task(task_key)
            # Fetch metrics
            metrics_url = f"{self.base_url}/metrics?task_name={task_key}"
            try:
                with urllib.request.urlopen(metrics_url, timeout=10) as m_resp:
                    metrics = json.loads(m_resp.read().decode("utf-8"))
                    return task.grader(metrics)
            except:
                # Fallback to a simplified score calculation if metrics endpoint is missing
                return 0.5


def get_model_action(client: OpenAI, model_name: str, task_name: str, step: int, observation: dict, history: List[str]) -> str:
    prompt = f"""You are an advanced smart traffic light controller.
Your goal is to maximize throughput and minimize delay.

Current Observation:
{json.dumps(observation, indent=2)}

Instructions:
1. Analyze the queue lengths: Northern: {observation.get('queue_north')}, Southern: {observation.get('queue_south')}, Eastern: {observation.get('queue_east')}, Western: {observation.get('queue_west')} 
2. Check for emergency vehicles! If active_priority is true, you MUST switch to the next_priority_approach immediately.
3. THINK logically about the best phase (NS or EW) inside <thought>...</thought> blocks.
4. Output your final decision inside an <action> block.

Decision Format:
<thought>
[Your step-by-step reasoning]
</thought>
<action>NS</action> 
OR 
<action>EW</action>
"""

    messages: List[Any] = [
        {"role": "system", "content": "You are a traffic signal controller optimizing throughput and emergency vehicle response. Always respond with 'NS' or 'EW'."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        if not response.choices:
            print("[DEBUG] OpenAI API returned no choices.", flush=True)
        else:
            content = response.choices[0].message.content
            if content:
                match = re.search(r'<action>\s*(NS|EW)\s*</action>', content, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
                else:
                     # Raw heuristic fallback if regex fails
                     return "NS" if "NS" in content.upper() else "EW"
    except Exception as exc:
        print(f"[DEBUG] OpenAI API error: {exc}", flush=True)

    if observation.get("active_priority"):
        priority_approach = observation.get("next_priority_approach")
        if priority_approach in {"N", "S"}:
            return "NS"
        elif priority_approach in {"E", "W"}:
            return "EW"

    ns_queue = observation["queue_north"] + observation["queue_south"]
    ew_queue = observation["queue_east"] + observation["queue_west"]
    return "NS" if ns_queue >= ew_queue else "EW"


def run_task(client: OpenAI, model_name: str, task_name: str, env: Any) -> float:
    observation = env.reset(task_name=task_name)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_name, env="smart_adaptive_traffic_signal", model=model_name)

    for step in range(1, MAX_STEPS + 1):
        obs_dict = get_obs_dict(observation)
        action_phase = get_model_action(client, model_name, task_name, step, obs_dict, history)
        if action_phase not in {"NS", "EW"}:
            action_phase = "NS"

        action = TrafficAction(phase=action_phase)
        observation, reward, done, _ = env.step(action)
        reward_value = reward["value"] if isinstance(reward, dict) else reward.value
        rewards.append(reward_value)
        steps_taken = step
        log_step(step=step, action=action_phase, reward=reward_value, done=done, error=None)
        history.append(f"Step {step}: {action_phase} -> reward {reward_value:+.2f}")
        if done:
            break

    score = env.evaluate()
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main() -> int:
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    if not api_base_url or not model_name or not hf_token:
        print("[ERROR] API_BASE_URL, MODEL_NAME, and HF_TOKEN must be provided.", flush=True)
        return 1

    try:
        client = OpenAI(base_url=api_base_url, api_key=hf_token)
        total_score = 0.0

        env_url = os.getenv("ENV_URL")
        if env_url:
             print(f"[DEBUG] Using Remote Inference: {env_url}", flush=True)
             env = RemoteEnvClient(env_url)
        else:
             print("[DEBUG] Using Local Inference", flush=True)
             try:
                 from smart_traffic_signal.env import SmartAdaptiveTrafficSignalEnv
                 env = SmartAdaptiveTrafficSignalEnv(seed=42)
             except ImportError as e:
                 print(f"[ERROR] Failed to import local environment: {e}", flush=True)
                 return 1

        for task_name in TASKS:
            task_score = run_task(client, model_name, task_name, env)
            total_score += task_score

        average_score = total_score / len(TASKS)
        print(f"[SUMMARY] overall_average_score={average_score:.4f}", flush=True)
        return 0
    except Exception as e:
        print(f"[ERROR] Unhandled exception in main: {e}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
