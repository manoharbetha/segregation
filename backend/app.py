import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from tensorflow.keras.models import load_model


# --------------------
# Configuration
# --------------------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
DEFAULT_MODEL_PATH = BASE_DIR / "saved_model" / "waste_classifier.h5"
MODEL_PATH = Path(os.environ.get("WASTE_MODEL_PATH", DEFAULT_MODEL_PATH))

IMAGE_SIZE = (224, 224)
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Class names must match the order used when the model was trained.
WASTE_CLASSES = ["Biodegradable", "E-Waste", "Non-Biodegradable"]

model = None


# --------------------
# Model loader & utils
# --------------------
def load_ml_model():
    """Load the trained Keras model into memory once when the API starts."""
    global model
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
        model = load_model(MODEL_PATH, compile=False)
        print(f"Model loaded successfully from: {MODEL_PATH}")
    except Exception as exc:
        print(f"Could not load model: {exc}")
        model = None


def safe_filename(filename: str) -> str:
    """Return a simple safe filename without depending on Flask/Werkzeug."""
    cleaned = Path(filename or "uploaded_image").name
    return "".join(ch for ch in cleaned if ch.isalnum() or ch in ("-", "_", ".", " ")).strip() or "uploaded_image"


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes into a model-ready array."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ml_model()
    print(f"Frontend served from: {FRONTEND_DIR}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    yield


# --------------------
# FastAPI app setup
# --------------------
app = FastAPI(
    title="Smart Waste Segregation API",
    description="Classifies waste images as Biodegradable, E-Waste, or Non-Biodegradable.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------
# API: prediction
# --------------------
@app.post("/api/segregate")
async def segregate_waste(file: UploadFile = File(...)):
    """Receive an image, predict its class, and return all class probabilities."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server issue.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        filename = safe_filename(file.filename)
        file_path = UPLOAD_FOLDER / filename
        file_path.write_bytes(image_bytes)

        image_array = preprocess_image(image_bytes)
        raw_preds = model.predict(image_array)
        raw_vector = np.array(raw_preds[0], dtype=float)

        # Softmax guard: ensure probabilities sum to approximately 1.
        if not np.isclose(raw_vector.sum(), 1.0, atol=1e-3):
            probs = tf.nn.softmax(raw_vector).numpy()
        else:
            probs = raw_vector

        class_probs = []
        for class_name, probability in zip(WASTE_CLASSES, probs):
            class_probs.append(
                {
                    "class": class_name,
                    "probability": float(probability),
                    "percentage": round(float(probability) * 100.0, 2),
                }
            )

        predicted_index = int(np.argmax(probs))
        category = WASTE_CLASSES[predicted_index]
        confidence = float(probs[predicted_index])

        print(f"Prediction: {category} ({confidence:.4f}) - vector: {probs.tolist()}")

        return {
            "category": category,
            "confidence": confidence,
            "class_probs": class_probs,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Processing Error: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc


# --------------------
# Routes: frontend + static files
# --------------------
@app.get("/")
async def home():
    """Serve the frontend index page."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(index_file)


@app.get("/{asset_path:path}")
async def serve_frontend_asset(asset_path: str):
    """Serve frontend assets such as styles.css and script.js."""
    requested_file = (FRONTEND_DIR / asset_path).resolve()
    frontend_root = FRONTEND_DIR.resolve()

    if not str(requested_file).startswith(str(frontend_root)) or not requested_file.is_file():
        return JSONResponse(status_code=404, content={"detail": "File not found"})

    return FileResponse(requested_file)
