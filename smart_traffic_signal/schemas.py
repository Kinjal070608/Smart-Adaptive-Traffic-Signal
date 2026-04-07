from pydantic import BaseModel
from typing import Literal, Dict

Phase = Literal["NS", "EW"]
PriorityApproach = Literal["N", "E", "S", "W", "NONE"]


class TrafficObservation(BaseModel):
    step: int
    phase: Phase
    queue_north: int
    queue_east: int
    queue_south: int
    queue_west: int
    active_priority: bool
    next_priority_approach: PriorityApproach
    priority_wait: int


class TrafficAction(BaseModel):
    phase: Phase


class TrafficReward(BaseModel):
    value: float
    details: Dict[str, float]
