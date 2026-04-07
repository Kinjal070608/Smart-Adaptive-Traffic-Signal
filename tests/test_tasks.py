from smart_traffic_signal.tasks import TASKS, get_task


def test_all_tasks_are_defined():
    assert set(TASKS.keys()) == {"easy", "medium", "hard"}


def test_get_task_returns_task():
    task = get_task("easy")
    assert task.key == "easy"
    assert task.difficulty == "easy"


def test_get_task_raises_for_unknown_task():
    try:
        get_task("unknown")
        assert False, "Expected ValueError for unknown task"
    except ValueError:
        pass


def test_grader_outputs_are_clamped():
    metrics = {
        "average_queue": 1.0,
        "priority_passed": 1,
        "priority_delay_total": 0.0,
        "total_departed": 10.0,
    }
    for task in TASKS.values():
        score = task.grader(metrics)
        assert 0.0 <= score <= 1.0
