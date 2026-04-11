from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class PriorityEvent:
    step: int
    approach: str
    deadline: int


@dataclass(frozen=True)
class TaskConfig:
    key: str
    difficulty: str
    description: str
    seed: int
    horizon: int
    arrival_rate: int
    initial_queues: Dict[str, int]
    priority_schedule: List[PriorityEvent]
    grader: Callable[[Dict[str, float]], float]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def grade_easy(metrics: Dict[str, float]) -> float:
    average_queue = metrics.get("average_queue", 0.0)
    score = 1.0 - ((average_queue - 2.0) / 6.0)
    return _clamp(score)


def grade_medium(metrics: Dict[str, float]) -> float:
    if metrics.get("priority_passed", 0) == 0:
        return 0.0
    delay = metrics.get("priority_delay_total", 0.0)
    score = 1.0 - min(delay / 8.0, 1.0)
    return _clamp(score)


def grade_hard(metrics: Dict[str, float]) -> float:
    throughput = metrics.get("total_departed", 0.0)
    average_queue = metrics.get("average_queue", 0.0)
    priority_passed = metrics.get("priority_passed", 0)
    priority_delay = metrics.get("priority_delay_total", 0.0)
    
    throughput_score = _clamp(throughput / 34.0)
    # Hard task always has 2 priority vehicles
    priority_count = 2
    priority_passage_ratio = _clamp(priority_passed / priority_count)
    priority_delay_score = _clamp(1.0 - min(priority_delay / 15.0, 1.0))
    priority_score = priority_passage_ratio * priority_delay_score
    
    balance_score = _clamp(1.0 - min(average_queue / 8.0, 1.0))
    return _clamp(0.4 * throughput_score + 0.4 * priority_score + 0.2 * balance_score)


TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        key="easy",
        difficulty="easy",
        description="Light intersection traffic with a single priority vehicle. Maintain low queue lengths.",
        seed=42,
        horizon=30,
        arrival_rate=1,
        initial_queues={"N": 2, "E": 2, "S": 2, "W": 2},
        priority_schedule=[PriorityEvent(step=8, approach="N", deadline=8)],
        grader=grade_easy,
    ),
    "medium": TaskConfig(
        key="medium",
        difficulty="medium",
        description="Moderate traffic and one ambulance arrival. Ensure the priority vehicle crosses before its deadline.",
        seed=64,
        horizon=36,
        arrival_rate=2,
        initial_queues={"N": 3, "E": 4, "S": 3, "W": 4},
        priority_schedule=[PriorityEvent(step=10, approach="E", deadline=9)],
        grader=grade_medium,
    ),
    "hard": TaskConfig(
        key="hard",
        difficulty="hard",
        description="Heavy traffic with two priority arrivals. Balance throughput and emergency responsiveness.",
        seed=123,
        horizon=42,
        arrival_rate=3,
        initial_queues={"N": 5, "E": 6, "S": 5, "W": 6},
        priority_schedule=[
            PriorityEvent(step=7, approach="S", deadline=8),
            PriorityEvent(step=20, approach="W", deadline=8),
        ],
        grader=grade_hard,
    ),
}


def get_task(key: str) -> TaskConfig:
    if key not in TASKS:
        raise ValueError(f"Unknown task '{key}'. Valid tasks: {list(TASKS.keys())}")
    return TASKS[key]
