import os
from typing import Any, List

from openai import OpenAI

from smart_traffic_signal.env import SmartAdaptiveTrafficSignalEnv
from smart_traffic_signal.schemas import TrafficAction

MAX_STEPS = 40
SUCCESS_SCORE_THRESHOLD = 0.6
TASKS = ["easy", "medium", "hard"]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.3f} done={done} error={error}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


def get_model_action(client: OpenAI, model_name: str, task_name: str, step: int, observation: dict, history: List[str]) -> str:
    prompt = (
        "You are controlling a traffic signal at a four-way intersection. "
        "Choose the next phase as NS or EW only. "
        "You must minimize overall queue build-up and STRICTLY prioritize any emergency vehicle waiting. "
        "Current state:\n"
        f"- Task: {task_name}\n"
        f"- Step: {step}\n"
        f"- Phase: {observation['phase']}\n"
        f"- Queues: N={observation['queue_north']}, E={observation['queue_east']}, "
        f"S={observation['queue_south']}, W={observation['queue_west']}\n"
        f"- Active priority: {observation['active_priority']}\n"
        f"- Priority direction: {observation['next_priority_approach']}\n"
        "Return only NS or EW."
    )

    messages: List[Any] = [
        {"role": "system", "content": "You are a traffic signal controller optimizing throughput and emergency vehicle response. Always respond with 'NS' or 'EW'."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=20,
        )
        content = response.choices[0].message.content
        if content:
            content = content.strip().upper()
            if "NS" in content:
                return "NS"
            if "EW" in content:
                return "EW"
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
