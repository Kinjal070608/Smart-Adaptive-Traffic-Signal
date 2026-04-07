---
title: Smart Adaptive Traffic Signal 🚦
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Smart Adaptive Traffic Signal OpenEnv Environment

A real-world traffic management environment that simulates a four-way intersection with adaptive signal control and priority handling for emergency vehicles such as ambulances.

## Problem Motivation

Traffic signal control is a critical real-world problem. Cities waste billions annually on congestion, while emergency response times are often delayed by traffic. This environment lets agents learn to:
- Minimize vehicle queue buildup and wait times
- Respond dynamically to changing vehicle arrivals
- Prioritize emergency vehicles (ambulances) to reduce critical response delays
- Balance conflicting objectives under realistic constraints

Agents trained here could inform actual adaptive traffic control systems deployed in smart cities.

## Environment Overview

The environment models a realistic signal control problem with:
- four incoming approaches (north, east, south, west)
- two signal phases: `NS` green and `EW` green
- variable arrival rates for normal traffic
- deterministic emergency vehicle arrivals with strict priority
- phase switching costs and safety-aware penalties

Agents interact through a standard `step(action)`, `reset()`, and `state()` API and observe queue lengths, current phase, and priority vehicle status.

## Action Space

`TrafficAction`
- `phase`: `NS` or `EW`

The agent chooses which signal phase to apply at each time step.

## Observation Space

`TrafficObservation` includes:
- `step`: current step index
- `phase`: active signal phase
- `queue_north`, `queue_east`, `queue_south`, `queue_west`: queued vehicles on each approach
- `active_priority`: whether a priority vehicle is currently waiting
- `next_priority_approach`: direction of the next priority vehicle or `NONE`
- `priority_wait`: total waiting time for the current priority vehicle

## Reward Structure

The reward function is designed to provide dense feedback:
- **Departure reward**: +0.35 per normal vehicle, +1.0 per priority vehicle
- **Queue penalty**: -0.05 per queued vehicle
- **Switch penalty**: -0.25 per phase switch (to discourage thrashing)
- **Priority wait penalty**: -0.35 per step while priority vehicle waits
- **Deadline penalty**: -1.0 if priority deadline is exceeded
- **Critical timeout**: -1.0 if priority wait exceeds 12 steps

The step-by-step signal ensures agents learn to balance throughput and responsiveness across the episode.

## Task Grading

Each task has a deterministic grader that evaluates the final trajectory:

- **easy**: `score = 1 - (avg_queue - 2) / 6`, clamped to [0, 1]
- **medium**: `score = 1 - min(priority_delay / 8, 1)` if priority passed, else 0.0
- **hard**: weighted combination of throughput (40%), priority response (40%), and balance (20%)

Graders are deterministic and reproducible across runs.

## Tasks and Difficulty

Three deterministic tasks with increasing difficulty:

1. **easy**: light traffic and one emergency vehicle; objective is low average queue length.
2. **medium**: moderate traffic with a single priority arrival; objective is on-time priority passage.
3. **hard**: heavy traffic, two priority arrivals, and balanced throughput with critical deadline handling.

Each task has a grader that returns a score in `[0.0, 1.0]`.

## Running the Environment

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### OpenEnv API

The environment implements the standard OpenEnv interface:

**`reset() -> TrafficObservation`**
- Initializes the environment to a clean state
- Returns initial observation with step=0, phase="NS", and reset queue counts

**`step(action: TrafficAction) -> Tuple[TrafficObservation, TrafficReward, bool, Dict]`**
- Advances the simulation by one time step
- Action specifies the next signal phase: `NS` or `EW`
- Returns:
  - `observation`: next state including queues, phase, priority status
  - `reward`: numeric value and component breakdown
  - `done`: whether episode has terminated
  - `info`: metadata including metrics and task score

**`state() -> Dict`**
- Returns a snapshot of the current environment state
- Includes step count, current phase, all queue lengths, and metrics

### HTTP Server API

Run the FastAPI server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8080
```

The server exposes:
- `GET /health` — returns `{"status": "ok"}`
- `POST /reset` — request: `{"task_name": "easy", "seed": 0}` → response with initial observation
- `POST /step` — request: `{"task_name": "easy", "action": {"phase": "NS"}}` → response with step result and metrics
- `GET /state` — query: `?task_name=easy` → returns current environment state

## Testing

Run the unit test suite with:

```bash
pytest
```

This includes:
- environment-level tests
- task grading sanity checks
- API endpoint tests for the FastAPI server

A helper validation script is also available:

```bash
bash validate.sh
```

## Baseline Inference

The baseline script uses the OpenAI API client and requires these environment variables:

- `OPENAI_API_KEY` — your OpenAI or compatible LLM API key
- `API_BASE_URL` — the API endpoint (e.g., `https://api.openai.com/v1`)
- `MODEL_NAME` — the model identifier (e.g., `gpt-4`, `gpt-3.5-turbo`)
- `HF_TOKEN` — your Hugging Face token (required by hackathon infrastructure)

Run:

```bash
export OPENAI_API_KEY="your-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-3.5-turbo"
export HF_TOKEN="your-hf-token"
python inference.py
```

### Expected Log Format

The script emits structured logs in this format:

```
[START] task=easy env=smart_adaptive_traffic_signal model=gpt-3.5-turbo
[STEP] step=1 action=NS reward=0.350 done=False error=None
[STEP] step=2 action=EW reward=-0.125 done=False error=None
...
[END] success=True steps=30 score=0.7500 rewards=[0.35, -0.125, ...]
[START] task=medium env=smart_adaptive_traffic_signal model=gpt-3.5-turbo
...
[END] overall_score=0.6833
```

Each task runs to completion or `MAX_STEPS` (40). Final score is computed by the task's grader function based on metrics like throughput, priority passage, and queue length.

## Pre-Submission Checklist

Before submitting, verify:

- [ ] Docker builds cleanly: `docker build -t smart-traffic-signal .`
- [ ] OpenEnv spec passes: `openenv validate`
- [ ] HF Space deploys and responds to `/health`
- [ ] Baseline inference runs without errors and produces valid scores
- [ ] All 3 tasks execute and graders return scores in `[0.0, 1.0]`
- [ ] Repository includes `openenv.yaml`, `Dockerfile`, `requirements.txt`, `inference.py`, and `README.md`

## Docker

Build and run the container:

```bash
docker build -t smart-traffic-signal .
docker run --rm -p 8080:8080 smart-traffic-signal
```

The container starts the FastAPI server for the OpenEnv-compatible space.
