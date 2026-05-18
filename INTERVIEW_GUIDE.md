# Smart Waste Segregation - Complete Interview Guide

This file explains the full project, the code, the technology stack, deployment, and common interview questions with answers.

## 1. Project Summary

Smart Waste Segregation is a machine learning web application that classifies an uploaded waste image into one of three categories:

- Biodegradable
- Non-Biodegradable
- E-Waste

The user opens a web page, uploads an image, clicks **Classify**, and the application displays confidence percentages for all three waste classes. The backend uses a trained TensorFlow/Keras model to perform image classification.

## 2. Problem Statement

Waste segregation is important because different waste categories require different disposal methods:

- Biodegradable waste can decompose naturally.
- Non-biodegradable waste usually needs recycling or controlled disposal.
- E-waste contains electronic components and may include harmful materials.

Manual segregation can be slow, inconsistent, and error-prone. This project automates basic waste classification using image recognition.

## 3. Objective

The objective is to build an AI-based waste classification system that:

- accepts a waste image from a user,
- preprocesses the image,
- predicts the waste category using a trained deep learning model,
- returns confidence scores,
- displays the result in a simple web interface.

## 4. High-Level Architecture

```text
User
  |
  v
Frontend: HTML, CSS, JavaScript
  |
  | image upload using fetch + FormData
  v
FastAPI Backend: /api/segregate
  |
  | preprocess image
  v
TensorFlow/Keras Model
  |
  | prediction probabilities
  v
FastAPI JSON Response
  |
  v
Frontend Result Cards
```

## 5. Folder Structure

```text
Smart_Waste_Seggregation-main/
|-- backend/
|   |-- __init__.py
|   |-- app.py
|   |-- saved_model/
|   |   `-- waste_classifier.h5
|   `-- uploads/
|       `-- .gitkeep
|-- frontend/
|   |-- index.html
|   |-- script.js
|   `-- styles.css
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- INTERVIEW_GUIDE.md
|-- .python-version
`-- .gitignore
```

## 6. Full Technology Stack

### Python

Python is used for the backend and machine learning code. It is popular for AI/ML because it has strong libraries like TensorFlow, NumPy, Pillow, and scikit-learn.

### FastAPI

FastAPI is used to build the backend API. It receives the uploaded image, calls the ML model, and returns JSON results.

Why FastAPI is used:

- It is modern and fast.
- It is API-focused.
- It supports file uploads easily.
- It provides automatic Swagger documentation at `/docs`.
- It uses Python type hints, which make code cleaner and easier to understand.

### Uvicorn

Uvicorn is the ASGI server used to run the FastAPI application.

In local development:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

In deployment:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

### TensorFlow

TensorFlow is used for loading and running the trained deep learning model. The model predicts the waste category from the uploaded image.

### Keras

Keras is the high-level neural network API used with TensorFlow. It is used in:

- `train_model.py` to build and train the model,
- `backend/app.py` to load the saved model using `load_model`.

### MobileNetV2

MobileNetV2 is the pretrained CNN model used for transfer learning.

Why MobileNetV2 is used:

- It is lightweight.
- It is faster than many large CNN models.
- It works well for image classification.
- It is suitable for future mobile or embedded use.
- It already knows useful visual features from large-scale pretraining.

### NumPy

NumPy is used to convert images into numerical arrays and handle model prediction vectors.

Example:

```python
img_array = np.array(img).astype("float32") / 255.0
```

### Pillow

Pillow is used for image processing. It opens the uploaded image, converts it to RGB, and resizes it.

Example:

```python
img = Image.open(BytesIO(image_bytes)).convert("RGB")
img = img.resize(IMAGE_SIZE)
```

### scikit-learn

scikit-learn is used in the training script to compute class weights. Class weights help handle an imbalanced dataset.

### HTML

HTML creates the structure of the web page. It contains the upload input, preview area, classify button, and result cards.

### CSS

CSS styles the page. It controls layout, colors, buttons, cards, progress bars, and image preview appearance.

### JavaScript

JavaScript handles frontend behavior:

- opens the file picker,
- previews the selected image,
- sends the image to the API,
- reads the JSON response,
- updates the result percentages.

### Render

Render is used for deployment. It hosts the FastAPI web service and gives a public URL.

### GitHub

GitHub stores the code and connects with Render for deployment.

## 7. Backend Code Explanation

Main file:

```text
backend/app.py
```

### Imports

```python
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
```

These are standard Python modules:

- `os` reads environment variables.
- `BytesIO` lets Pillow read image bytes like a file.
- `Path` handles file paths cleanly.
- `asynccontextmanager` is used for FastAPI startup/shutdown lifespan.

```python
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from tensorflow.keras.models import load_model
```

These are project libraries:

- `numpy` handles arrays.
- `tensorflow` handles softmax and model inference.
- `FastAPI` creates the API app.
- `UploadFile` and `File` handle file upload.
- `HTTPException` returns proper error responses.
- `CORSMiddleware` allows frontend/backend communication.
- `FileResponse` serves HTML, CSS, and JS files.
- `PIL.Image` handles image preprocessing.
- `load_model` loads the trained `.h5` model.

### Configuration

```python
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
DEFAULT_MODEL_PATH = BASE_DIR / "saved_model" / "waste_classifier.h5"
MODEL_PATH = Path(os.environ.get("WASTE_MODEL_PATH", DEFAULT_MODEL_PATH))
```

This sets important paths:

- `BASE_DIR` points to the `backend/` folder.
- `FRONTEND_DIR` points to the `frontend/` folder.
- `DEFAULT_MODEL_PATH` points to the saved model.
- `WASTE_MODEL_PATH` lets you override the model path in deployment if needed.

```python
IMAGE_SIZE = (224, 224)
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
```

Every image is resized to `224 x 224`, which matches MobileNetV2 input size. The uploads folder is created if it does not exist.

```python
WASTE_CLASSES = ["Biodegradable", "E-Waste", "Non-Biodegradable"]
```

These are the output labels. The order must match the order used during model training.

### Model Loading

```python
model = None
```

The model is stored globally so it loads once and can be reused for all predictions.

```python
def load_ml_model():
    global model
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
        model = load_model(MODEL_PATH, compile=False)
        print(f"Model loaded successfully from: {MODEL_PATH}")
    except Exception as exc:
        print(f"Could not load model: {exc}")
        model = None
```

This function loads the `.h5` model. `compile=False` is used because the backend only needs inference, not training.

### File Name Cleaning

```python
def safe_filename(filename: str) -> str:
    cleaned = Path(filename or "uploaded_image").name
    return "".join(ch for ch in cleaned if ch.isalnum() or ch in ("-", "_", ".", " ")).strip() or "uploaded_image"
```

This prevents unsafe file paths from uploaded filenames. It keeps only simple filename characters.

### Image Preprocessing

```python
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)
```

Steps:

1. Read image bytes.
2. Convert image to RGB.
3. Resize to `224 x 224`.
4. Convert image to NumPy array.
5. Normalize pixel values from `0-255` to `0-1`.
6. Add batch dimension, making shape `(1, 224, 224, 3)`.

### FastAPI Lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ml_model()
    print(f"Frontend served from: {FRONTEND_DIR}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    yield
```

This runs when the FastAPI app starts. It loads the model and prints useful startup information.

### FastAPI App

```python
app = FastAPI(
    title="Smart Waste Segregation API",
    description="Classifies waste images as Biodegradable, E-Waste, or Non-Biodegradable.",
    version="1.0.0",
    lifespan=lifespan,
)
```

This creates the FastAPI application and also controls the automatic `/docs` page.

### CORS

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

CORS allows requests from different origins. It is useful if the frontend and backend are deployed separately. In this project, the same backend serves the frontend, but CORS still makes the API flexible.

### Prediction Endpoint

```python
@app.post("/api/segregate")
async def segregate_waste(file: UploadFile = File(...)):
```

This endpoint receives the uploaded image.

Important validations:

```python
if model is None:
    raise HTTPException(status_code=503, detail="Model not loaded. Server issue.")
```

This returns `503` if the model failed to load.

```python
if not file.filename:
    raise HTTPException(status_code=400, detail="No file selected")
```

This returns `400` if no file was uploaded.

Prediction flow:

```python
image_bytes = await file.read()
filename = safe_filename(file.filename)
file_path = UPLOAD_FOLDER / filename
file_path.write_bytes(image_bytes)
image_array = preprocess_image(image_bytes)
raw_preds = model.predict(image_array)
```

The backend reads the uploaded image, saves a copy, preprocesses it, and sends it to the model.

Probability handling:

```python
raw_vector = np.array(raw_preds[0], dtype=float)

if not np.isclose(raw_vector.sum(), 1.0, atol=1e-3):
    probs = tf.nn.softmax(raw_vector).numpy()
else:
    probs = raw_vector
```

If model outputs are already probabilities, they are used directly. If not, softmax converts them into probabilities.

Final response:

```python
return {
    "category": category,
    "confidence": confidence,
    "class_probs": class_probs,
}
```

The frontend expects this JSON response.

### Serving Frontend

```python
@app.get("/")
async def home():
    return FileResponse(index_file)
```

This serves `frontend/index.html`.

```python
@app.get("/{asset_path:path}")
async def serve_frontend_asset(asset_path: str):
```

This serves frontend files like `styles.css` and `script.js`.

## 8. Frontend Code Explanation

### `frontend/index.html`

This file defines the page structure:

- sidebar title,
- image upload section,
- hidden file input,
- image preview box,
- classify button,
- loading spinner,
- result cards for the three classes.

Important element IDs:

- `waste-image-input`: file input.
- `choose-btn`: opens file selector.
- `image-preview`: shows selected image.
- `classify-button`: sends image to backend.
- `results-section`: displays final result.

### `frontend/styles.css`

This file controls the UI design:

- green sidebar,
- white upload card,
- dashed image preview box,
- primary and ghost buttons,
- loading spinner,
- result cards,
- confidence progress bars.

The result cards use different colors:

- amber for Biodegradable,
- blue for Non-Biodegradable,
- purple for E-Waste.

### `frontend/script.js`

This file handles frontend logic.

API endpoint:

```javascript
const API = '/api/segregate';
```

Since FastAPI serves both frontend and backend, this relative URL works locally and on Render.

File selection:

```javascript
fileInput.addEventListener('change', (e) => {
  const file = e.target.files && e.target.files[0];
  selectedFile = file;
});
```

Image preview:

```javascript
const url = URL.createObjectURL(file);
showPreviewFromUrl(url);
```

API call:

```javascript
const fd = new FormData();
fd.append('file', file);
const r = await fetch(API, { method: 'POST', body: fd });
```

This sends the image as `multipart/form-data`.

Result update:

```javascript
updateCardSafe('Biodegradable', percentage);
updateCardSafe('Non-Biodegradable', percentage);
updateCardSafe('E-Waste', percentage);
```

The frontend updates percentages and progress bars.

## 9. Model Training Explanation

Main file:

```text
train_model.py
```

### Dataset Loading

The dataset must be arranged like this:

```text
waste_dataset/
|-- Biodegradable/
|-- E-waste/
`-- Non-Biodegradable/
```

Keras reads each folder name as a class label.

### ImageDataGenerator

The project uses `ImageDataGenerator` for:

- rescaling,
- validation split,
- rotation,
- width shift,
- height shift,
- shear,
- zoom,
- horizontal flip,
- brightness adjustment.

Why data augmentation is used:

- It improves generalization.
- It reduces overfitting.
- It creates variations from existing images.

### Transfer Learning

The model uses MobileNetV2 with pretrained ImageNet weights:

```python
base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
```

`include_top=False` removes the original ImageNet classifier so a custom classifier can be added for waste categories.

### Custom Classifier

```python
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
```

Explanation:

- `GlobalAveragePooling2D` reduces feature maps.
- `Dropout` reduces overfitting.
- `Dense(128, relu)` learns task-specific patterns.
- `Dense(NUM_CLASSES, softmax)` outputs class probabilities.

### Training Phase 1

MobileNetV2 base layers are frozen:

```python
for layer in base_model.layers:
    layer.trainable = False
```

Only the new classifier layers are trained first.

### Training Phase 2

The last 40 layers are unfrozen:

```python
for layer in base_model.layers[-40:]:
    layer.trainable = True
```

This fine-tunes deeper features for the waste dataset.

### Callbacks

The project uses:

- `EarlyStopping`: stops training when validation loss stops improving.
- `ModelCheckpoint`: saves the best model.
- `ReduceLROnPlateau`: reduces learning rate when training slows.

### Class Weights

Class weights are used because the dataset is imbalanced. If one class has fewer images, class weights make the model pay more attention to that class during training.

## 10. Deployment Explanation

The project is deployed on Render as a web service.

Important files:

- `.python-version`
- `requirements.txt`
- `backend/app.py`

### `.python-version`

```text
3.11.9
```

This is needed because TensorFlow may not support very new Python versions like Python 3.14.

### Build Command

```text
pip install -r requirements.txt
```

This installs all backend dependencies.

### Start Command

```text
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

Render provides `$PORT`, and the app must bind to `0.0.0.0` so Render can expose it publicly.

## 11. How To Run Locally

```bash
python -m venv .venv
```

Windows:

```bash
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## 12. Interview Project Explanation Script

You can say:

"My project is a Smart Waste Segregation web application. It classifies waste images into Biodegradable, Non-Biodegradable, and E-Waste categories. The frontend is built using HTML, CSS, and JavaScript. The backend is built with FastAPI. When a user uploads an image, JavaScript sends it to the FastAPI endpoint `/api/segregate` using FormData. The backend preprocesses the image by converting it to RGB, resizing it to 224 by 224 pixels, normalizing pixel values, and passing it to a TensorFlow/Keras model. The model is based on MobileNetV2 transfer learning. The backend returns confidence scores as JSON, and the frontend displays them using result cards and progress bars. I deployed the application on Render using Uvicorn as the ASGI server."

## 13. Strengths of This Project

- It solves a real-world environmental problem.
- It combines frontend, backend, machine learning, and deployment.
- It uses a real trained model, not only static logic.
- It has an API-first architecture.
- It uses transfer learning for efficient model training.
- It is deployed and publicly accessible.

## 14. Limitations

- Accuracy depends on dataset quality and balance.
- E-Waste may need more images if the dataset is imbalanced.
- Images with poor lighting or unclear objects can be misclassified.
- The system classifies one image at a time.
- It is a prototype and needs more real-world testing before production use.

## 15. Future Enhancements

- Add camera capture.
- Add drag-and-drop upload.
- Add prediction history.
- Add user authentication.
- Improve dataset balance.
- Add confusion matrix and accuracy charts.
- Add real-time object detection.
- Connect to IoT smart bin hardware.
- Use a lighter model format such as TensorFlow Lite for edge deployment.
- Add Docker deployment.

## 16. Common Interview Questions and Answers

### Q1. What is your project about?

It is an AI-based smart waste segregation system that classifies uploaded waste images into Biodegradable, Non-Biodegradable, and E-Waste categories.

### Q2. What problem does it solve?

It helps automate basic waste segregation, reducing manual effort and improving waste sorting accuracy.

### Q3. What are the main modules?

The main modules are the frontend, FastAPI backend, trained TensorFlow model, and training script.

### Q4. Which algorithm or model did you use?

I used MobileNetV2 with transfer learning for image classification.

### Q5. Why did you use MobileNetV2?

MobileNetV2 is lightweight, fast, and pretrained. It gives good performance without needing to train a CNN from scratch.

### Q6. What is transfer learning?

Transfer learning means using a model pretrained on a large dataset and adapting it to a new task. In this project, MobileNetV2 already knows general image features, and I train custom layers for waste categories.

### Q7. Why did you use FastAPI?

FastAPI is modern, fast, and API-focused. It handles file uploads cleanly, gives automatic API documentation, and works well for ML inference APIs.

### Q8. Why not Flask?

Flask is also valid, but FastAPI gives better API documentation, type validation, and cleaner request handling. Since this project is API-based, FastAPI is a strong choice.

### Q9. What does Uvicorn do?

Uvicorn is the ASGI server that runs the FastAPI app and handles HTTP requests.

### Q10. What is the endpoint for prediction?

The prediction endpoint is:

```text
POST /api/segregate
```

### Q11. What type of data does the endpoint accept?

It accepts an image file using `multipart/form-data`.

### Q12. What does the API return?

It returns the predicted category, confidence value, and probability percentages for all classes.

### Q13. How is the image preprocessed?

The image is converted to RGB, resized to `224 x 224`, converted to a NumPy array, normalized to `0-1`, and expanded into a batch shape.

### Q14. Why resize to 224 x 224?

MobileNetV2 commonly uses `224 x 224` input size, so all images must be resized to that shape before prediction.

### Q15. Why normalize pixel values?

Neural networks train and predict better when input values are scaled. Pixel values are divided by `255.0` to convert them from `0-255` to `0-1`.

### Q16. What is softmax?

Softmax converts raw model outputs into probability values whose sum is approximately 1. It is commonly used for multi-class classification.

### Q17. What are the output classes?

The output classes are Biodegradable, E-Waste, and Non-Biodegradable.

### Q18. Why is class order important?

The model outputs probabilities by index. The backend class list must match the training class order, otherwise predictions can be labeled incorrectly.

### Q19. What is the saved model file?

The saved model is:

```text
backend/saved_model/waste_classifier.h5
```

### Q20. What is an `.h5` file?

It is an HDF5 model file format used by Keras to store a trained model.

### Q21. What is `compile=False` in `load_model`?

It loads the model without compiling it for training. For prediction/inference, compilation is not required.

### Q22. How did you train the model?

I used Keras ImageDataGenerator to load images, applied data augmentation, used MobileNetV2 as the base model, trained custom classifier layers, then fine-tuned the top layers.

### Q23. What is data augmentation?

Data augmentation creates modified versions of training images, such as rotated or zoomed images, to improve generalization.

### Q24. Why use dropout?

Dropout reduces overfitting by randomly disabling some neurons during training.

### Q25. What is overfitting?

Overfitting happens when a model performs well on training data but poorly on new unseen images.

### Q26. What is validation split?

Validation split separates part of the dataset for testing model performance during training.

### Q27. What is EarlyStopping?

EarlyStopping stops training when validation loss stops improving, preventing unnecessary training and overfitting.

### Q28. What is ModelCheckpoint?

ModelCheckpoint saves the best model during training.

### Q29. What is ReduceLROnPlateau?

It reduces the learning rate when validation loss stops improving, helping the model continue learning more carefully.

### Q30. What are class weights?

Class weights give more importance to underrepresented classes during training.

### Q31. Why is the dataset ignored in Git?

The dataset is large and not necessary for running the deployed prediction app. The trained model is included, while the dataset can be kept locally for retraining.

### Q32. What is the role of `requirements.txt`?

It lists the Python packages needed to run the project.

### Q33. What is the role of `.python-version`?

It tells Render to use Python `3.11.9`, which is compatible with TensorFlow.

### Q34. Why did deployment fail first?

Render used Python 3.14 by default, but TensorFlow was not available for that Python version. Pinning Python to 3.11.9 fixed it.

### Q35. Why bind to `0.0.0.0` in deployment?

Cloud platforms need the app to listen on all network interfaces so they can route external traffic to it.

### Q36. Why use `$PORT` on Render?

Render dynamically assigns a port. The app must use Render's `$PORT` environment variable.

### Q37. What is CORS?

CORS controls whether browsers allow frontend code to call APIs from another domain. It is useful when frontend and backend are hosted separately.

### Q38. How does the frontend send the image?

It uses JavaScript `FormData` and `fetch` to send the image file to `/api/segregate`.

### Q39. How are results displayed?

The frontend receives JSON and updates the percentage text and progress bars for each class.

### Q40. What happens if no image is selected?

The frontend keeps the classify button disabled until a file is selected. The backend also validates the file.

### Q41. What happens if the model is missing?

The backend returns a `503` error saying the model is not loaded.

### Q42. What happens if the uploaded file is empty?

The backend returns a `400` error.

### Q43. Can this project run without internet?

After dependencies and model are installed locally, it can run without internet on localhost.

### Q44. Can this project be deployed on Vercel?

The static frontend can be deployed on Vercel, but the TensorFlow backend is better deployed on Render or another Python backend service because ML dependencies are heavy.

### Q45. Why deploy backend and frontend together on Render?

It is simpler. FastAPI serves both the frontend files and the prediction API from the same domain.

### Q46. What is the biggest challenge in this project?

The biggest challenges are dataset quality, class imbalance, model accuracy, and deploying TensorFlow with a compatible Python version.

### Q47. How can accuracy be improved?

Accuracy can be improved by adding more balanced data, cleaning mislabeled images, tuning hyperparameters, trying better architectures, and using evaluation metrics like confusion matrix.

### Q48. Is this a classification or detection project?

It is an image classification project. It predicts one class for the whole image. It does not locate objects inside the image.

### Q49. How would you make it real-time?

I would add camera input, process frames, and either classify selected frames or use object detection models like YOLO for real-time detection.

### Q50. How would you integrate this with hardware?

The backend prediction could be connected to a smart bin using a camera and a microcontroller. Based on the predicted class, the bin could move a servo motor to direct waste to the correct compartment.

### Q51. What security improvements would you add?

I would validate file type, limit file size, sanitize filenames, avoid saving unnecessary uploads, add rate limiting, and restrict CORS origins in production.

### Q52. Why is `safe_filename` used?

It prevents unsafe path characters in uploaded filenames and avoids path traversal issues.

### Q53. What is path traversal?

Path traversal is when a malicious filename tries to access files outside the intended upload folder, such as `../../secret.txt`.

### Q54. Why use `FileResponse`?

`FileResponse` sends frontend files like `index.html`, CSS, and JavaScript from the FastAPI backend.

### Q55. What is `/docs`?

`/docs` is FastAPI's automatically generated Swagger UI, where API endpoints can be tested in the browser.

### Q56. What is the difference between training and inference?

Training is when the model learns from data. Inference is when the trained model predicts the class of a new image.

### Q57. Why is the model loaded globally?

Loading a model is expensive. Loading it once and reusing it makes predictions faster.

### Q58. What is JSON?

JSON is a lightweight data format used to send structured data between backend and frontend.

### Q59. What is REST API?

A REST API exposes endpoints over HTTP. In this project, `/api/segregate` is the prediction API endpoint.

### Q60. How would you explain this project in one minute?

"This is a deployed AI web app for smart waste segregation. Users upload a waste image, and the frontend sends it to a FastAPI backend. The backend preprocesses the image and uses a trained TensorFlow/Keras MobileNetV2 model to classify it as Biodegradable, Non-Biodegradable, or E-Waste. It returns confidence scores as JSON, and the frontend displays the results. The project uses transfer learning, image preprocessing, API development, and cloud deployment on Render."

## 17. Short Answers To Remember

- Project type: supervised image classification.
- Backend: FastAPI.
- Server: Uvicorn.
- Model: MobileNetV2 transfer learning.
- ML library: TensorFlow/Keras.
- Input size: `224 x 224`.
- Output: 3 classes.
- API endpoint: `POST /api/segregate`.
- Deployment: Render.
- Python version on Render: `3.11.9`.
- Frontend: HTML, CSS, JavaScript.

## 18. Best Closing Statement

"This project gave me experience in building an end-to-end machine learning application: dataset preparation, model training, backend API development, frontend integration, and cloud deployment. It is a practical prototype that can be improved further with better data, camera support, and IoT integration."
