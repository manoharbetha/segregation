# Smart Waste Segregation

Smart Waste Segregation is an end-to-end image classification project that predicts whether an uploaded waste item belongs to one of three categories:

- `Biodegradable`
- `Non-Biodegradable`
- `E-Waste`

The project combines a lightweight FastAPI backend, a simple browser-based frontend, and a TensorFlow/Keras model built with MobileNetV2 transfer learning.

## Accuracy Summary

| Metric | Value |
| --- | --- |
| Previous observed validation accuracy | `92.02%` (`3542 / 3849`) |
| Class-wise evaluation | Precision, recall, F1-score, confusion matrix |
| Fresh verified score in this repo | Not available until the dataset is added and retraining is run |

If you want to add a screenshot later, place it in the repo root or a `docs/` folder and link it here.

## Highlights

- Three-class waste image classification
- FastAPI inference API with file upload support
- HTML, CSS, and JavaScript frontend
- MobileNetV2-based transfer learning pipeline
- Saved production model included in the repository
- Retraining pipeline with class-wise evaluation output

## Tech Stack

- `Python`
- `FastAPI`
- `Uvicorn`
- `TensorFlow / Keras`
- `MobileNetV2`
- `NumPy`
- `Pillow`
- `scikit-learn`
- `HTML`
- `CSS`
- `JavaScript`

## Project Structure

```text
segregation/
|-- backend/
|   |-- app.py
|   |-- __init__.py
|   `-- saved_model/
|       `-- waste_classifier.h5
|-- frontend/
|   |-- index.html
|   |-- script.js
|   `-- styles.css
|-- INTERVIEW_GUIDE.md
|-- README.md
|-- requirements.txt
|-- run_server.ps1
`-- train_model.py
```

## How It Works

1. A user uploads a waste image from the web interface.
2. The frontend sends the image to `POST /api/segregate`.
3. The backend preprocesses the image and runs model inference.
4. The API returns the predicted class and probabilities for all three categories.
5. The frontend displays the result and confidence breakdown.

## API Response

Example response from `POST /api/segregate`:

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

## Run Locally

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Model Training

The repository includes the trained model file, but the original dataset is intentionally excluded from Git.

To retrain the classifier, add your local dataset in this format:

```text
waste_dataset/
|-- Biodegradable/
|-- E-waste/
`-- Non-Biodegradable/
```

Then run:

```bash
python train_model.py
```

The training script uses:

- MobileNetV2 transfer learning
- data augmentation for training images
- clean validation evaluation without random augmentation
- class weights for imbalanced classes
- early stopping
- learning rate reduction
- fine-tuning of the top MobileNetV2 layers

## Accuracy And Evaluation

The current repository does not include the dataset, so a fresh accuracy number cannot be reproduced directly from GitHub alone.

After retraining with the local dataset, the script saves:

```text
backend/saved_model/model_metadata.json
backend/saved_model/training_metrics.json
backend/saved_model/training_log.csv
```

The evaluation checks:

- validation accuracy
- correct predictions out of total validation samples
- precision for each class
- recall for each class
- F1-score for each class
- confusion matrix

Accuracy is calculated as:

```text
Accuracy = Correct Predictions / Total Validation Predictions
```

The improved evaluation flow in `train_model.py`:

1. splits the dataset using `validation_split=0.2`
2. trains the classifier head
3. fine-tunes the top MobileNetV2 layers
4. evaluates the model on the validation set
5. generates class-wise metrics from validation predictions
6. saves the final metrics in `training_metrics.json`

## Deployment Note

The repository includes `.python-version` pinned to `3.11.9` to avoid TensorFlow compatibility issues during deployment on platforms such as Render.

Suggested commands for Render:

```text
Build Command: pip install -r requirements.txt
Start Command: uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

## Limitations

- accuracy depends heavily on dataset quality and class balance
- the dataset is not bundled in the repository
- the current app classifies one uploaded image at a time
- real-world performance may drop for blurry, dark, or cluttered images

## Future Improvements

- add drag-and-drop upload
- support camera capture
- store prediction history
- add Docker support
- improve dataset balance for E-Waste
- export confusion matrix visuals after training

## Interview Support

For a detailed explanation of the architecture, training flow, technology choices, and interview questions, see [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md).
