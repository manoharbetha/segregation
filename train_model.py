import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ==========================
# CONFIGURATION
# ==========================
DATA_ROOT = 'waste_dataset'  # Main dataset folder (should have subfolders)
MODEL_SAVE_PATH = 'backend/saved_model/waste_classifier.h5'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
PHASE1_EPOCHS = 10  # Classifier training
PHASE2_EPOCHS = 15  # Fine-tuning

# ==========================
# CHECK DATASET
# ==========================
if not os.path.exists(DATA_ROOT):
    raise FileNotFoundError(f"❌ Dataset folder '{DATA_ROOT}' not found. Please check your path.")

# ==========================
# DATA GENERATORS
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=35,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

print("📂 Loading training data...")
train_generator = train_datagen.flow_from_directory(
    DATA_ROOT,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("📂 Loading validation data...")
validation_generator = train_datagen.flow_from_directory(
    DATA_ROOT,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ==========================
# CLASSES & CLASS WEIGHTS
# ==========================
NUM_CLASSES = train_generator.num_classes
WASTE_CLASSES = list(train_generator.class_indices.keys())
print(f"✅ Classes detected: {WASTE_CLASSES}")

classes = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)
class_weights = dict(enumerate(class_weights_array))
print(f"⚖️ Computed class weights: {class_weights}")

# ==========================
# MODEL BUILDING
# ==========================
print("\n🔧 Building MobileNetV2 model...")
base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

# Common top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ==========================
# CALLBACKS
# ==========================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)
]

# ==========================
# PHASE 1: Train classifier layers
# ==========================
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n🚀 Starting Phase 1 training (frozen base) for {PHASE1_EPOCHS} epochs...")
history1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ==========================
# PHASE 2: Fine-tuning
# ==========================
print("\n🔓 Unfreezing top layers for fine-tuning...")
for layer in base_model.layers[-40:]:  # last 40 layers
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n🚀 Starting Phase 2 fine-tuning for {PHASE2_EPOCHS} epochs...")
history2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ==========================
# FINAL SAVE
# ==========================
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH, include_optimizer=False)
print(f"✅ Final model saved at: {MODEL_SAVE_PATH}")

# ==========================
# EVALUATE MODEL
# ==========================
print("\n🧪 Evaluating model on validation data...")
val_loss, val_acc = model.evaluate(validation_generator)
print(f"🔬 Validation Accuracy: {val_acc * 100:.2f}%")
