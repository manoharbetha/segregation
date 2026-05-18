# Smart Waste Segregation

Smart Waste Segregation is a machine learning web application that classifies a waste item image into one of three categories:

- Biodegradable
- Non-Biodegradable
- E-Waste

The project uses a TensorFlow/Keras image classification model with a FastAPI backend and a simple HTML/CSS/JavaScript frontend. A user uploads an image, clicks **Classify**, and the app returns confidence scores for each waste category.

## Project Overview

This project helps automate basic waste segregation using image classification. It can be used as a prototype for smart bins, recycling awareness tools, or waste sorting assistance systems.

The application has three main parts:

1. **Dataset**
   - Training images should be stored locally inside `waste_dataset/`.
   - The dataset is divided into class folders:
     - `Biodegradable`
     - `E-waste`
     - `Non-Biodegradable`
   - The dataset folder is ignored by Git because it is large. Add it locally before retraining.

2. **Training Script**
   - `train_model.py` trains a MobileNetV2-based image classifier.
   - It uses transfer learning, data augmentation, class weights, early stopping, and fine-tuning.
   - The trained model is saved as `backend/saved_model/waste_classifier.h5`.

3. **FastAPI Web Application**
   - `backend/app.py` runs the FastAPI application.
   - The frontend is served from the `frontend/` folder.
   - Users upload an image through the browser.
   - The backend preprocesses the image, predicts its waste category, and sends JSON results back to the frontend.

## Folder Structure

```text
Smart_Waste_Seggregation-main/
├── backend/
│   ├── app.py
│   ├── saved_model/
│   │   └── waste_classifier.h5
│   └── uploads/
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── waste_dataset/
│   ├── Biodegradable/
│   ├── E-waste/
│   └── Non-Biodegradable/
├── requirements.txt
├── run_server.ps1
├── train_model.py
└── README.md
```

## Technologies Used

- Python
- FastAPI
- Uvicorn
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pillow
- scikit-learn
- HTML
- CSS
- JavaScript

## How It Works

1. The user opens the web page.
2. The user selects an image of a waste item.
3. The frontend previews the selected image.
4. When the user clicks **Classify**, the image is sent to the FastAPI endpoint.
5. The backend:
   - reads the uploaded image,
   - saves a copy in `backend/uploads/`,
   - resizes it to `224 x 224`,
   - normalizes pixel values,
   - passes it to the trained Keras model,
   - calculates confidence scores.
6. The frontend displays the confidence percentage for each category.

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.\.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

From the project root, run:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Then open this URL in your browser:

```text
http://127.0.0.1:8000
```

FastAPI also gives automatic API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Deploying on Render

Render's default Python version can be newer than TensorFlow supports. This project includes a `.python-version` file pinned to Python `3.11.9` so TensorFlow installs correctly during deployment.

Create a Render **Web Service** from this GitHub repository and use:

```text
Build Command: pip install -r requirements.txt
Start Command: uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

After deployment, open your Render URL. The API docs will be available at:

```text
https://your-render-service.onrender.com/docs
```

## API Endpoint

### `POST /api/segregate`

Accepts an image file and returns the predicted waste category with confidence values.

Request format:

```text
multipart/form-data
file: uploaded image
```

Example response:

```json
{
  "category": "Biodegradable",
  "confidence": 0.94,
  "class_probs": [
    {
      "class": "Biodegradable",
      "probability": 0.94,
      "percentage": 94.0
    },
    {
      "class": "E-Waste",
      "probability": 0.03,
      "percentage": 3.0
    },
    {
      "class": "Non-Biodegradable",
      "probability": 0.03,
      "percentage": 3.0
    }
  ]
}
```

## Training the Model

To retrain the model, make sure the dataset is available in this format:

```text
waste_dataset/
├── Biodegradable/
├── E-waste/
└── Non-Biodegradable/
```

Then run:

```bash
python train_model.py
```

The training script performs two stages:

1. **Feature extraction**
   - MobileNetV2 base layers are frozen.
   - Only the custom classifier layers are trained.

2. **Fine-tuning**
   - The last 40 layers of MobileNetV2 are unfrozen.
   - The model is trained with a lower learning rate.

After training, the model is saved here:

```text
backend/saved_model/waste_classifier.h5
```

## Model Details

- Base model: MobileNetV2
- Input image size: `224 x 224`
- Output classes: 3
- Loss function: categorical crossentropy
- Optimizer: Adam
- Data augmentation:
  - rotation
  - width shift
  - height shift
  - shear
  - zoom
  - horizontal flip
  - brightness adjustment

## Why FastAPI?

FastAPI is a good fit because this project is API-focused. The frontend sends an image to the backend, and the backend returns JSON prediction data. FastAPI also provides:

- automatic Swagger docs at `/docs`,
- clean request handling for file uploads,
- better type hints and validation,
- easy CORS configuration,
- simple deployment with Uvicorn or Gunicorn/Uvicorn workers.

## Important Notes

- The class order in the backend must match the order used during model training.
- Uploaded images are saved in `backend/uploads/`.
- The current model file is expected at `backend/saved_model/waste_classifier.h5`.
- If you move the model, set the `WASTE_MODEL_PATH` environment variable before running the server.

Windows example:

```bash
set WASTE_MODEL_PATH=path\to\waste_classifier.h5
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## Limitations

- Prediction accuracy depends on the quality and balance of the dataset.
- The model may misclassify images that are blurry, dark, cropped, or very different from the training data.
- TensorFlow can make deployment heavier than a normal web API.
- This is a prototype and should be tested further before real-world deployment.

## Future Improvements

- Add model accuracy and training charts.
- Add drag-and-drop image upload.
- Add camera capture support.
- Store prediction history.
- Improve dataset balance, especially for E-Waste images.
- Add production deployment instructions.

## Author

Smart Waste Segregation project.
