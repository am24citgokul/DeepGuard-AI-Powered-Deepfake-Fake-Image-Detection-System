import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from audio_branch.inference.audio_verifier import predict_audio

print("=== DEEPGUARD X BACKEND AUDIO API TEST ===")
client = app.test_client()

# 1. Health Endpoint
res = client.get("/api/health")
print(f"\n1. GET /api/health -> Status: {res.status_code}")
health_data = res.get_json()
print("   Audio ready:", health_data.get("models", {}).get("audio"))

# 2. Audio Metrics Endpoint
res = client.get("/api/audio/metrics")
print(f"\n2. GET /api/audio/metrics -> Status: {res.status_code}")
metrics_data = res.get_json()
print(f"   Accuracy: {metrics_data.get('accuracy')*100:.2f}%")
print(f"   F1-Score: {metrics_data.get('f1'):.4f}")
print(f"   Fake Recall: {metrics_data.get('fake_recall')*100:.2f}%")
print(f"   Real Recall: {metrics_data.get('real_recall')*100:.2f}%")

# 3. Predict Endpoint on Real Audio
real_sample = Path(r"E:\ADL_PROJECT_DATASET\release_in_the_wild\real\291.wav")
if real_sample.is_file():
    with open(real_sample, "rb") as f:
        res = client.post("/api/audio/predict", data={"file": (f, "291.wav")}, content_type="multipart/form-data")
    print(f"\n3. POST /api/audio/predict (Known Real: 291.wav) -> Status: {res.status_code}")
    print("   Data:", res.data.decode('utf-8'))
    real_out = res.get_json() or {}
    print("   Model:", real_out.get("model"))
    print("   Prediction:", real_out.get("prediction"))
    print("   Real Probability:", real_out.get("real_probability"))
    print("   Fake Probability:", real_out.get("fake_probability"))
    print("   Confidence:", real_out.get("confidence"))
    print("   Status:", real_out.get("status"))

# 4. Predict Endpoint on Fake Audio
fake_sample = Path(r"E:\ADL_PROJECT_DATASET\release_in_the_wild\fake\6441.wav")
if fake_sample.is_file():
    with open(fake_sample, "rb") as f:
        res = client.post("/api/audio/predict", data={"file": (f, "6441.wav")}, content_type="multipart/form-data")
    print(f"\n4. POST /api/audio/predict (Known Fake: 6441.wav) -> Status: {res.status_code}")
    fake_out = res.get_json()
    print("   Model:", fake_out.get("model"))
    print("   Prediction:", fake_out.get("prediction"))
    print("   Real Probability:", fake_out.get("real_probability"))
    print("   Fake Probability:", fake_out.get("fake_probability"))
    print("   Confidence:", fake_out.get("confidence"))
    print("   Status:", fake_out.get("status"))

# 5. Predict Endpoint on AASIST Model
if fake_sample.is_file():
    with open(fake_sample, "rb") as f:
        res = client.post("/api/audio/predict", data={"file": (f, "6441.wav"), "model": "aasist"}, content_type="multipart/form-data")
    print(f"\n5. POST /api/audio/predict [AASIST] -> Status: {res.status_code}")
    aasist_out = res.get_json()
    print("   Model:", aasist_out.get("model"))
    print("   Prediction:", aasist_out.get("prediction"))
    print("   Fake Probability:", aasist_out.get("fake_probability"))

print("\n=== BACKEND AUDIO API TEST COMPLETE ===")
