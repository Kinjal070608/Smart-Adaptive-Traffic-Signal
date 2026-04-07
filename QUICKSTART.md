# Quick Start Guide

## 1-Minute Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (verify environment works)
pytest -q

# Start web demo
python app.py
# Open http://localhost:7860 in browser
```

## 5-Minute API Test

```bash
# Start server
uvicorn server:app --port 8080 &

# Test health check
curl http://localhost:8080/health
# Response: {"status":"ok"}

# Reset environment (easy task)
curl -X POST http://localhost:8080/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"easy","seed":0}'

# Step (choose NS phase)
curl -X POST http://localhost:8080/step \
  -H "Content-Type: application/json" \
  -d '{"task_name":"easy","action":{"phase":"NS"}}'
```

## 10-Minute Baseline Test

```bash
# Set API credentials
export OPENAI_API_KEY="your-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-3.5-turbo"
export HF_TOKEN="your-hf-token"

# Run baseline
python inference.py
# Expect: [START], [STEP], [END] logs for all 3 tasks
```

## Docker Test

```bash
# Build
docker build -t smart-traffic-signal .

# Run
docker run --rm -p 8080:8080 smart-traffic-signal

# Test (in another terminal)
curl http://localhost:8080/health
```

## Full Validation

```bash
# Run all tests
pytest -v

# Validate OpenEnv spec
pip install openenv-core
openenv validate

# Check Docker build
docker build -t smart-traffic-signal .
```

## Next Steps for Submission

1. **Create GitHub/HF repo** and push this code
2. **Deploy to HF Spaces** using Docker SDK
3. **Set environment variables** in Space secrets
4. **Run baseline inference** to confirm scores
5. **Submit Space URL** to hackathon portal

See `SUBMISSION.md` for detailed checklist.
