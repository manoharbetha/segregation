import json
import os
import warnings

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")


# ==========================
# CONFIGURATION
# ==========================
DATA_ROOT = "waste_dataset"
MODEL_SAVE_PATH = "backend/saved_model/waste_classifier.h5"
METADATA_SAVE_PATH = "backend/saved_model/model_metadata.json"
METRICS_SAVE_PATH = "backend/saved_model/training_metrics.json"
TRAINING_LOG_PATH = "backend/saved_model/training_log.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 15
SEED = 42

tf.keras.utils.set_random_seed(SEED)


# ==========================
# CHECK DATASET
# ==========================
if not os.path.exists(DATA_ROOT):
    raise FileNotFoundError(
        f"Dataset folder '{DATA_ROOT}' was not found. Add your local dataset before training."
    )

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


# ==========================
# DATA GENERATORS
# ==========================
# MobileNetV2 was pre-trained with this preprocessing, so it usually performs
# better than simple rescaling for transfer-learning projects.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=35,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

# Do not augment validation images. Validation should measure the model on
# stable, real examples instead of random transformed copies.
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    DATA_ROOT,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED,
)

print("Loading validation data...")
validation_generator = validation_datagen.flow_from_directory(
    DATA_ROOT,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED,
)


# ==========================
# CLASSES & CLASS WEIGHTS
# ==========================
NUM_CLASSES = train_generator.num_classes
WASTE_CLASSES = list(train_generator.class_indices.keys())
print(f"Classes detected: {WASTE_CLASSES}")

with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as metadata_file:
    json.dump(
        {
            "class_indices": train_generator.class_indices,
            "class_names": WASTE_CLASSES,
            "image_size": IMAGE_SIZE,
            "preprocessing": "mobilenet_v2.preprocess_input",
            "validation_split": 0.2,
            "seed": SEED,
        },
        metadata_file,
        indent=2,
    )

classes = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(classes),
    y=classes,
)
class_weights = dict(enumerate(class_weights_array))
print(f"Computed class weights: {class_weights}")


# ==========================
# MODEL BUILDING
# ==========================
print("\nBuilding MobileNetV2 model...")
base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)


# ==========================
# CALLBACKS
# ==========================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
    CSVLogger(TRAINING_LOG_PATH, append=False),
]


# ==========================
# PHASE 1: TRAIN CLASSIFIER
# ==========================
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)

print(f"\nStarting Phase 1 training for {PHASE1_EPOCHS} epochs...")
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)


# ==========================
# PHASE 2: FINE-TUNING
# ==========================
print("\nUnfreezing top MobileNetV2 layers for fine-tuning...")
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)

print(f"\nStarting Phase 2 fine-tuning for {PHASE2_EPOCHS} epochs...")
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)


# ==========================
# FINAL SAVE
# ==========================
model.save(MODEL_SAVE_PATH, include_optimizer=False)
print(f"Final model saved at: {MODEL_SAVE_PATH}")


# ==========================
# EVALUATE MODEL
# ==========================
print("\nEvaluating model on validation data...")
val_loss, val_acc = model.evaluate(validation_generator, verbose=1)

validation_generator.reset()
y_true = validation_generator.classes
y_prob = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

correct_predictions = int(np.sum(y_true == y_pred))
total_predictions = int(len(y_true))
report = classification_report(
    y_true,
    y_pred,
    target_names=WASTE_CLASSES,
    output_dict=True,
    zero_division=0,
)
matrix = confusion_matrix(y_true, y_pred).tolist()

metrics = {
    "validation_loss": float(val_loss),
    "validation_accuracy": float(accuracy_score(y_true, y_pred)),
    "correct_predictions": correct_predictions,
    "total_predictions": total_predictions,
    "classification_report": report,
    "confusion_matrix": matrix,
    "class_names": WASTE_CLASSES,
}

with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as metrics_file:
    json.dump(metrics, metrics_file, indent=2)

print(f"Validation accuracy: {val_acc * 100:.2f}%")
print(f"Correct predictions: {correct_predictions} of {total_predictions}")
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=WASTE_CLASSES, zero_division=0))
print(f"Metrics saved at: {METRICS_SAVE_PATH}")
