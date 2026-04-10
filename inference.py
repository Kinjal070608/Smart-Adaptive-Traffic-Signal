import json
import os
import re
from typing import Any, List

from openai import OpenAI

from smart_traffic_signal.env import SmartAdaptiveTrafficSignalEnv
from smart_traffic_signal.schemas import TrafficAction

MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.6
TASKS = ["easy", "medium", "hard"]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.3f} done={done} error={error}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


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


def run_task(client: OpenAI, model_name: str, task_name: str) -> float:
    env = SmartAdaptiveTrafficSignalEnv(task_name=task_name, seed=42)
    observation = env.reset()
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_name, env="smart_adaptive_traffic_signal", model=model_name)

    for step in range(1, MAX_STEPS + 1):
        action_phase = get_model_action(client, model_name, task_name, step, observation.model_dump(), history)
        if action_phase not in {"NS", "EW"}:
            action_phase = "NS"

        action = TrafficAction(phase=action_phase)
        observation, reward, done, _ = env.step(action)
        rewards.append(reward.value)
        steps_taken = step
        log_step(step=step, action=action_phase, reward=reward.value, done=done, error=None)
        history.append(f"Step {step}: {action_phase} -> reward {reward.value:+.2f}")
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

    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    total_score = 0.0

    for task_name in TASKS:
        task_score = run_task(client, model_name, task_name)
        total_score += task_score

    average_score = total_score / len(TASKS)
    print(f"[END] overall_score={average_score:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
