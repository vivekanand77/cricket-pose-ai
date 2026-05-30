# cricket-pose-ai

Beginner-friendly cricket bowling analysis MVP using:
- Python + FastAPI backend
- MediaPipe pose landmarks for joint mechanics
- Google AI Studio (Gemini API) feedback generation

## MVP flow
1. Upload a bowling video (`<= 10 seconds`) to `POST /analyze-bowling`
2. Backend extracts right-side bowling joint angles (elbow, shoulder, knee)
3. API returns:
   - joint angle analysis
   - AI-generated coaching feedback
   - per-angle correction image (base64 encoded preview)

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Test
```bash
pytest -q tests/test_app.py
```
