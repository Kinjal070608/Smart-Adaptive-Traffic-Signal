# Smart Adaptive Traffic Signal - Hackathon Submission Guide

## Project Summary

**Environment Name**: Smart Adaptive Traffic Signal  
**Author**: [Your Name]  
**Real-world Task**: Adaptive traffic signal control with priority vehicle (ambulance) handling

### Key Features
- ✅ Real-world utility: solves genuine traffic management problem with emergency response
- ✅ Full OpenEnv spec: typed models, `reset()`, `step()`, `state()` API
- ✅ 3 graded tasks: easy, medium, hard with deterministic graders
- ✅ Dense reward shaping: step-level feedback with partial progress signals
- ✅ Baseline inference: reproducible scores via OpenAI API client
- ✅ Docker support: containerized deployment ready
- ✅ Testing: comprehensive pytest suite (11 tests)
- ✅ Documentation: complete README with API details

---

## Pre-Submission Validation Checklist

### Phase 1: Local Development

- [ ] **Clone/download the repository**
  ```bash
  git clone <repo-url>
  cd smart_adaptive_traffic_signal
  ```

- [ ] **Install dependencies**
  ```bash
  python -m pip install -r requirements.txt
  ```

- [ ] **Run test suite**
  ```bash
  pytest -v
  ```
  Expected: 11 tests pass

- [ ] **Test environment locally**
  ```bash
  python -c "from smart_traffic_signal import SmartAdaptiveTrafficSignalEnv; env = SmartAdaptiveTrafficSignalEnv('easy'); obs = env.reset(); print('OK:', obs.phase)"
  ```

- [ ] **Verify FastAPI server**
  ```bash
  uvicorn server:app --host 0.0.0.0 --port 8080
  # In another terminal:
  curl http://localhost:8080/health
  ```
  Expected: `{"status":"ok"}`

- [ ] **Test Gradio demo (optional)**
  ```bash
  python app.py
  # Opens at http://localhost:7860
  ```

### Phase 2: Docker & OpenEnv Validation

- [ ] **Build Docker image**
  ```bash
  docker build -t smart-traffic-signal .
  ```
  Expected: build succeeds without errors

- [ ] **Run container**
  ```bash
  docker run --rm -p 8080:8080 smart-traffic-signal
  # In another terminal:
  curl http://localhost:8080/health
  ```

- [ ] **Install openenv-core**
  ```bash
  pip install openenv-core
  ```

- [ ] **Validate OpenEnv spec**
  ```bash
  openenv validate
  ```
  Expected: passes all validation checks

### Phase 3: Baseline Inference

- [ ] **Set environment variables**
  ```bash
  export OPENAI_API_KEY="your-api-key"
  export API_BASE_URL="https://api.openai.com/v1"
  export MODEL_NAME="gpt-3.5-turbo"
  export HF_TOKEN="your-hf-token"
  ```

- [ ] **Run baseline inference**
  ```bash
  python inference.py
  ```
  Expected output format:
  ```
  [START] task=easy env=smart_adaptive_traffic_signal model=gpt-3.5-turbo
  [STEP] step=1 action=NS reward=... done=False error=None
  [STEP] step=2 action=EW reward=... done=False error=None
  ...
  [END] success=... steps=... score=... rewards=[...]
  [START] task=medium ...
  [END] success=... steps=... score=... rewards=[...]
  [START] task=hard ...
  [END] success=... steps=... score=... rewards=[...]
  [END] overall_score=...
  ```

- [ ] **Verify baseline scores**
  - Task scores should be in range [0.0, 1.0]
  - Overall score should be average of task scores
  - Script runs within 20 minutes
  - No errors in log output

### Phase 4: Hugging Face Spaces Deployment

- [ ] **Create HF Space repo** (or use existing)
  - Visit: https://huggingface.co/new-space
  - Repo ID: `your-username/smart-adaptive-traffic-signal`
  - License: Apache 2.0
  - Space SDK: Docker

- [ ] **Push code to HF repo**
  ```bash
  git remote add hf https://huggingface.co/spaces/your-username/smart-adaptive-traffic-signal
  git push hf main
  # Or if using HF CLI:
  huggingface-cli repo create smart-adaptive-traffic-signal --type space --space-sdk docker
  git clone https://huggingface.co/spaces/your-username/smart-adaptive-traffic-signal
  cp -r <local-files> <hf-space-dir>
  ```

- [ ] **Configure environment secrets in HF Space**
  - Go to Space settings → "Repository secrets"
  - Add: `OPENAI_API_KEY`, `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`

- [ ] **Verify Space deploys**
  - Space should build and run within a few minutes
  - HTTP endpoint should respond to `/health` with 200 status
  - Check logs for any errors

- [ ] **Test Space endpoints**
  ```bash
  curl https://your-username-smart-adaptive-traffic-signal.hf.space/health
  # Expected: {"status": "ok"}
  ```

- [ ] **Test Space via UI (if Gradio app)**
  - Navigate to Space URL
  - Click "Reset Environment"
  - Select a phase and click "Step"
  - Verify observations and scores display

### Phase 5: Final Submission

- [ ] **Verify all required files are present**
  - `openenv.yaml` — environment metadata
  - `Dockerfile` — containerization
  - `requirements.txt` — dependencies
  - `inference.py` — baseline script
  - `README.md` — documentation
  - `server.py` — FastAPI server
  - `smart_traffic_signal/` — package with env, schemas, tasks
  - `tests/` — pytest suite

- [ ] **Confirm README includes**
  - [ ] Problem motivation
  - [ ] Action/observation space definitions
  - [ ] Task descriptions with difficulty
  - [ ] Setup and usage instructions
  - [ ] Baseline scores and expected performance
  - [ ] OpenEnv API (`reset()`, `step()`, `state()`)

- [ ] **Copy HF Space URL**
  - Format: `https://<username>-<space-name>.hf.space`

- [ ] **Submit to hackathon portal**
  - Paste HF Space URL
  - Confirm all validation gates pass
  - Submit before deadline

---

## Troubleshooting

### Docker Build Fails
- Ensure `Dockerfile` exists in repo root
- Check Python version compatibility (3.10+)
- Verify all dependencies in `requirements.txt` are available

### OpenEnv Validate Fails
- Run: `openenv validate --verbose`
- Ensure `openenv.yaml` is valid YAML
- Check that entrypoint class is correctly importable

### Inference Script Hangs
- Ensure LLM API is reachable at `API_BASE_URL`
- Check API key is valid
- Reduce `MAX_STEPS` temporarily for testing
- Set a timeout: `timeout 5m python inference.py`

### HF Space Doesn't Deploy
- Check Docker build logs in Space → "Logs"
- Ensure `Dockerfile` correctly starts the server
- Verify environment variables are set in Space secrets

---

## Local Testing Example

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run tests
pytest -q

# 3. Test environment directly
python << 'EOF'
from smart_traffic_signal import SmartAdaptiveTrafficSignalEnv
env = SmartAdaptiveTrafficSignalEnv("medium", seed=42)
obs = env.reset()
print(f"Task: medium, Phase: {obs.phase}")
for _ in range(5):
    from smart_traffic_signal.schemas import TrafficAction
    obs, reward, done, _ = env.step(TrafficAction(phase="NS"))
    print(f"Reward: {reward.value:.2f}, Done: {done}")
print(f"Final Score: {env.evaluate():.4f}")
EOF

# 4. Test server
uvicorn server:app --port 8080 &
sleep 2
curl http://localhost:8080/health
# Expected: {"status":"ok"}
kill %1

# 5. Test inference (with API keys)
export OPENAI_API_KEY="sk-..."
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-3.5-turbo"
export HF_TOKEN="hf_..."
timeout 5m python inference.py
```

---

## Submission Scoring Criteria

| Criterion | Weight | Assessment |
|-----------|--------|-----------|
| Real-world utility | 30% | Traffic signal control for cities + ambulance priority |
| Task & grader quality | 25% | 3 tasks, deterministic graders, clear difficulty progression |
| Environment design | 20% | Clean state mgmt, sensible action/obs, dense reward, proper boundaries |
| Code quality & spec | 15% | OpenEnv API, typed models, tested, Dockerfile works |
| Creativity & novelty | 10% | Interesting mechanics, deadline-aware grading, dual objectives |

---

## Contact & Support

For issues or questions, refer to:
- **README.md** — environment and API documentation
- **openenv.yaml** — metadata and configuration
- **smart_traffic_signal/tasks.py** — task definitions and graders
- **Hackathon Docs** — https://huggingface.co/OpenEnv/hackathon

Good luck! 🚦
