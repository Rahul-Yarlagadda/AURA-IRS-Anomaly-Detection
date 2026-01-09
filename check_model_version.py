from joblib import load

try:
    detector = load("corrected_aura_detector_v2.pkl")
    print("✅ Model loaded successfully.")
    print("Model class type:", type(detector))
except Exception as e:
    print("❌ Error loading model:", e)
