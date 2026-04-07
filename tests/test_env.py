from smart_traffic_signal.env import SmartAdaptiveTrafficSignalEnv
from smart_traffic_signal.schemas import TrafficAction


def test_reset_returns_initial_observation():
    env = SmartAdaptiveTrafficSignalEnv(task_name="easy", seed=0)
    observation = env.reset()

    assert observation.step == 0
    assert observation.phase == "NS"
    assert observation.queue_north >= 0
    assert observation.queue_east >= 0
    assert observation.queue_south >= 0
    assert observation.queue_west >= 0
    assert observation.active_priority is False
    assert observation.next_priority_approach == "NONE"


def test_step_updates_state_and_reward():
    env = SmartAdaptiveTrafficSignalEnv(task_name="easy", seed=0)
    env.reset()
    observation, reward, done, info = env.step(TrafficAction(phase="EW"))

    assert observation.step == 1
    assert observation.phase == "EW"
    assert isinstance(reward.value, float)
    assert done is False
    assert info["task"] == "easy"
    assert "metrics" not in info or isinstance(info["task"], str)


def test_priority_vehicle_arrives_and_passes():
    env = SmartAdaptiveTrafficSignalEnv(task_name="easy", seed=42)
    env.reset()

    for _ in range(7):
        env.step(TrafficAction(phase="NS"))

    assert env.active_priority is False
    env.step(TrafficAction(phase="NS"))
    assert env.active_priority is False
    assert env.priority_passed == 1
    assert env.evaluate() >= 0.0


def test_evaluate_score_is_valid():
    env = SmartAdaptiveTrafficSignalEnv(task_name="medium", seed=42)
    env.reset()
    for _ in range(5):
        env.step(TrafficAction(phase="NS"))

    score = env.evaluate()
    assert 0.0 <= score <= 1.0
