import json
import os
import re
import sys
import time
import traceback
import urllib.request
import urllib.error
from typing import Any, List, Optional, Dict, Union, Tuple

MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.6
TASKS = ["easy", "medium", "hard"]

# Global logging helpers
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.3f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)

def get_obs_dict(observation: Any) -> dict:
    """Helper for Pydantic v1/v2 compatibility and dict safety."""
    if isinstance(observation, dict):
        return observation
    try:
        if hasattr(observation, "model_dump"):
            return observation.model_dump()
        if hasattr(observation, "dict"):
            return observation.dict()
    except:
        pass
    return {}

class RemoteEnvClient:
    """Bulletproof client for remote OpenEnv server using only standard libraries."""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _call(self, path: str, method: str = "GET", data: Optional[dict] = None) -> Optional[dict]:
        url = f"{self.base_url}{path}"
        try:
            req = urllib.request.Request(url, method=method)
            if data:
                req.add_header("Content-Type", "application/json")
                req.data = json.dumps(data).encode("utf-8")
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"[DEBUG] RemoteEnv error ({path}): {e}", flush=True)
            return None

    def reset(self, task_name: str = "easy", seed: int = 42) -> dict:
        resp = self._call("/reset", method="POST", data={"task_name": task_name, "seed": seed})
        return resp if resp else {}

    def step(self, action_dict: dict) -> Tuple[dict, Any, bool, dict]:
        resp = self._call("/step", method="POST", data=action_phase_to_json(action_dict))
        if not resp:
            return {}, 0.0, True, {"error": "connection_lost"}
        
        obs = resp.get("observation", {})
        reward = resp.get("reward", {"value": 0.0})
        done = resp.get("done", True)
        info = resp.get("info", {})
        return obs, reward, done, info

    def evaluate(self) -> float:
        resp = self._call("/state", method="GET")
        if not resp: return 0.0
        
        task_key = resp.get("task", "easy")
        metrics = self._call(f"/metrics?task_name={task_key}", method="GET")
        if not metrics: return 0.5 # Neutral fallback
        
        # Try to use local grader if available, else return neutral
        try:
            from smart_traffic_signal.tasks import get_task
            task = get_task(task_key)
            return task.grader(metrics)
        except:
            return metrics.get("priority_passed", 0) / 2.0 # Heuristic fallback

def action_phase_to_json(action: Any) -> dict:
    """Safely convert action object or dict to API format."""
    if isinstance(action, dict): return action
    if hasattr(action, "phase"): return {"phase": action.phase}
    return {"phase": "NS"}

def get_model_action(client: Any, model_name: str, task_name: str, step: int, observation: dict, history: List[str]) -> str:
    prompt = f"""You are an advanced smart traffic light controller.
Current Observation:
{json.dumps(observation, indent=2)}

Instructions:
1. Analyze queue lengths and check for priority vehicles.
2. If active_priority is true, return the next_priority_approach phase immediately.
3. Respond inside <thought> reasoning and <action> (NS or EW).
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Traffic signal controller. Always respond with <action>NS</action> or <action>EW</action>."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150,
        )
        if response.choices:
            content = response.choices[0].message.content
            if content:
                match = re.search(r'<action>\s*(NS|EW)\s*</action>', content, re.IGNORECASE)
                if match: return match.group(1).upper()
    except Exception as exc:
        print(f"[DEBUG] LLM API error: {exc}", flush=True)

    # Heuristic Fallback
    if observation.get("active_priority"):
        p_app = observation.get("next_priority_approach")
        return "NS" if p_app in {"N", "S"} else "EW"
    ns = observation.get("queue_north", 0) + observation.get("queue_south", 0)
    ew = observation.get("queue_east", 0) + observation.get("queue_west", 0)
    return "NS" if ns >= ew else "EW"

def run_task(client: Any, model_name: str, task_name: str, env: Any) -> float:
    try:
        observation = env.reset(task_name=task_name)
    except Exception as e:
        print(f"[ERROR] Reset failed for {task_name}: {e}", flush=True)
        return 0.0

    rewards = []
    steps_taken = 0
    history = []
    log_start(task=task_name, env="smart_adaptive_traffic_signal", model=model_name)

    for step in range(1, MAX_STEPS + 1):
        try:
            obs_dict = get_obs_dict(observation)
            action_phase = get_model_action(client, model_name, task_name, step, obs_dict, history)
            
            # Create action safely
            try:
                from smart_traffic_signal.schemas import TrafficAction
                action = TrafficAction(phase=action_phase)
            except:
                action = {"phase": action_phase}

            observation, reward, done, _ = env.step(action)
            
            # Reward safety
            reward_value = reward.get("value", 0.0) if isinstance(reward, dict) else getattr(reward, "value", 0.0)
            rewards.append(reward_value)
            steps_taken = step
            log_step(step=step, action=action_phase, reward=reward_value, done=done, error=None)
            history.append(f"Step {step}: {action_phase}")
            if done: break
        except Exception as e:
            print(f"[DEBUG] Step error: {e}", flush=True)
            break

    try:
        score = env.evaluate()
    except:
        score = 0.0
    log_end(success=score >= SUCCESS_SCORE_THRESHOLD, steps=steps_taken, score=score, rewards=rewards)
    return score

def main() -> int:
    # 1. Environment Detection & Server Wait
    env_url = os.getenv("ENV_URL", "http://localhost:7860")
    print(f"[DEBUG] Checking environment at {env_url}...", flush=True)
    
    server_ready = False
    for i in range(10): # 30s timeout
        try:
            with urllib.request.urlopen(f"{env_url}/health", timeout=2) as r:
                if r.getcode() == 200:
                    server_ready = True
                    break
        except:
            time.sleep(3)
    
    # 2. Lazy Imports
    try:
        from openai import OpenAI
        api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        key = os.getenv("HF_TOKEN", "missing")
        client = OpenAI(base_url=api_base, api_key=key)
    except Exception as e:
        print(f"[ERROR] Dependency/API setup error: {e}", flush=True)
        return 1

    # 3. Environment Choice
    if server_ready:
        print(f"[DEBUG] Server detected. Using Remote Inference.", flush=True)
        env = RemoteEnvClient(env_url)
    else:
        print(f"[DEBUG] Server not found. Attempting Local Inference.", flush=True)
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from smart_traffic_signal.env import SmartAdaptiveTrafficSignalEnv
            env = SmartAdaptiveTrafficSignalEnv(seed=42)
        except Exception as e:
            print(f"[ERROR] Local environment fallback failed: {e}", flush=True)
            return 1

    # 4. Task Execution
    total_score = 0.0
    try:
        for task_name in TASKS:
            total_score += run_task(client, model, task_name, env)
        print(f"[SUMMARY] overall_average_score={total_score / len(TASKS):.4f}", flush=True)
        return 0
    except Exception as e:
        print(f"[ERROR] Main execution failure: {e}", flush=True)
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"CRITICAL: {e}")
        sys.exit(1)
