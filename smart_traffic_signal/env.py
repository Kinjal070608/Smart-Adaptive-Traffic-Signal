import random
from typing import Dict, List, Tuple, Any, cast

from .schemas import TrafficAction, TrafficObservation, TrafficReward, Phase, PriorityApproach
from .tasks import get_task

PHASE_DIRECTIONS = {
    "NS": ["N", "S"],
    "EW": ["E", "W"],
}

MAX_PRIORITY_WAIT = 12


class SmartAdaptiveTrafficSignalEnv:
    def __init__(self, task_name: str = "easy", seed: int = 0):
        self.task_name = task_name
        self.seed = seed
        self.task = get_task(task_name)
        self.random = random.Random(seed)
        self.reset()

    def reset(self) -> TrafficObservation:
        self.step_count = 0
        self.phase: Phase = "NS"
        self.queues: Dict[str, int] = self.task.initial_queues.copy()
        self.priority_queue: List[Dict[str, int]] = []
        self.total_departed = 0
        self.total_priority_departed = 0
        self.priority_delay_total = 0
        self.priority_passed = 0
        self.switches = 0
        self.cumulative_queue = 0
        self._pending_events = list(self.task.priority_schedule)
        self._last_action = TrafficAction(phase=self.phase)
        self.active_priority = False
        self.active_priority_approach: PriorityApproach = "NONE"
        self.active_priority_deadline = 0
        self.priority_wait = 0
        self._update_observation()
        return self.observation

    def step(self, action: TrafficAction) -> Tuple[TrafficObservation, TrafficReward, bool, Dict[str, Any]]:
        if action.phase not in PHASE_DIRECTIONS:
            raise ValueError("Action phase must be 'NS' or 'EW'.")

        self.step_count += 1
        previous_phase = self.phase
        if action.phase != self.phase:
            self.phase = action.phase
            self.switches += 1
            switch_cost = 1
        else:
            switch_cost = 0

        self._spawn_arrivals()
        self._spawn_priority_vehicle()

        departed_normal, departed_priority = self._serve_current_phase(switch_cost > 0)
        self.priority_delay_total += self.priority_wait
        self._update_observation()

        reward, details = self._compute_reward(departed_normal, departed_priority, switch_cost)
        done = self.step_count >= self.task.horizon
        self.cumulative_queue += self._total_queue()

        info = {
            "task": self.task.key,
            "step": self.step_count,
            "departed_normal": departed_normal,
            "departed_priority": departed_priority,
            "pending_priority_events": len(self._pending_events),
        }

        return self.observation, TrafficReward(value=reward, details=details), done, info

    def state(self) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "phase": self.phase,
            "queues": self.queues.copy(),
            "active_priority": self.active_priority,
            "priority_wait": self.priority_wait,
            "total_departed": self.total_departed,
            "total_priority_departed": self.total_priority_departed,
            "priority_delay_total": self.priority_delay_total,
            "switches": self.switches,
            "task": self.task.key,
        }

    def get_metrics(self) -> Dict[str, float]:
        average_queue = self.cumulative_queue / max(1, self.step_count)
        return {
            "total_departed": float(self.total_departed),
            "total_priority_departed": float(self.total_priority_departed),
            "priority_delay_total": float(self.priority_delay_total),
            "priority_passed": float(self.priority_passed),
            "average_queue": float(average_queue),
            "switches": float(self.switches),
        }

    def evaluate(self) -> float:
        return self.task.grader(self.get_metrics())

    def _spawn_arrivals(self) -> None:
        for approach in self.queues:
            self.queues[approach] += self.random.randint(0, self.task.arrival_rate)

    def _spawn_priority_vehicle(self) -> None:
        arrived = [event for event in self._pending_events if event.step == self.step_count]
        for event in arrived:
            self.queues[event.approach] += 1
            self.active_priority = True
            self.active_priority_approach = cast(PriorityApproach, event.approach)
            self.active_priority_deadline = event.deadline
            self.priority_wait = 0
        self._pending_events = [event for event in self._pending_events if event.step != self.step_count]

    def _serve_current_phase(self, is_switching: bool) -> Tuple[int, int]:
        departed_normal = 0
        departed_priority = 0
        if is_switching:
            self.priority_wait += 1 if self.active_priority else 0
            return departed_normal, departed_priority

        directions = PHASE_DIRECTIONS[self.phase]
        total_queue = self._total_queue()
        # Adaptive capacity: reduce if less traffic
        base_capacity = 2 if total_queue >= 10 else 1

        for approach in directions:
            if self.active_priority and self.queues[approach] > 0:
                departed_priority += 1
                self.queues[approach] -= 1
                self.total_priority_departed += 1
                self.priority_passed += 1
                self.active_priority = False
                self.priority_wait = 0
                break

        for approach in directions:
            capacity = base_capacity
            while capacity > 0 and self.queues[approach] > 0:
                self.queues[approach] -= 1
                departed_normal += 1
                self.total_departed += 1
                capacity -= 1

        if self.active_priority:
            self.priority_wait += 1
        return departed_normal, departed_priority

    def _compute_reward(self, departed_normal: int, departed_priority: int, switch_cost: int) -> Tuple[float, Dict[str, float]]:
        total_queue = self._total_queue()
        reward = 0.35 * departed_normal + 1.0 * departed_priority
        reward -= 0.05 * total_queue
        reward -= 0.25 * switch_cost
        reward -= 0.35 * (self.priority_wait if self.active_priority else 0)
        deadline_exceeded = float(self.active_priority and self.priority_wait > self.active_priority_deadline)
        if deadline_exceeded:
            reward -= 1.0
        if self.active_priority and self.priority_wait > MAX_PRIORITY_WAIT:
            reward -= 1.0
        details = {
            "departed_normal": float(departed_normal),
            "departed_priority": float(departed_priority),
            "queue_penalty": float(0.05 * total_queue),
            "switch_penalty": float(0.25 * switch_cost),
            "priority_wait_penalty": float(0.35 * (self.priority_wait if self.active_priority else 0)),
            "deadline_exceeded_penalty": deadline_exceeded,
            "total_queue": float(total_queue),
        }
        return reward, details

    def _update_observation(self) -> None:
        next_priority_approach = cast(PriorityApproach, next((event.approach for event in self.task.priority_schedule if event.step <= self.step_count), "NONE"))
        self.observation = TrafficObservation(
            step=self.step_count,
            phase=self.phase,
            queue_north=self.queues["N"],
            queue_east=self.queues["E"],
            queue_south=self.queues["S"],
            queue_west=self.queues["W"],
            active_priority=self.active_priority,
            next_priority_approach=self.active_priority_approach if self.active_priority else cast(PriorityApproach, "NONE"),
            priority_wait=self.priority_wait,
        )

    def _total_queue(self) -> int:
        return sum(self.queues.values())
