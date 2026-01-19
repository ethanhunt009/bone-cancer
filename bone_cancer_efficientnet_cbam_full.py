import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. EDITABLE DATASET PATH (CHANGE ONLY THIS)
# =========================================================
BASE_DATASET_PATH = r"D:\bone cancer\archive"

TRAIN_DIR = os.path.join(BASE_DATASET_PATH, "train")
VALID_DIR = os.path.join(BASE_DATASET_PATH, "valid")
TEST_DIR  = os.path.join(BASE_DATASET_PATH, "test")

# =========================================================
# 2. CHECKPOINT / BACKUP CONFIGURATION (RESUMABLE)
# =========================================================
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_CKPT_PATH = os.path.join(
    CHECKPOINT_DIR, "bone_cancer_best_model.keras"
)

BACKUP_DIR = os.path.join(
    CHECKPOINT_DIR, "backup"
)

# =========================================================
# 3. TRAINING CONFIG
# =========================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# =========================================================
# 4. LOAD CSV AND BUILD DATAFRAME
# =========================================================
def load_dataframe(data_dir):
    csv_path = os.path.join(data_dir, "_classes.csv")
    df = pd.read_csv(csv_path)

    # Binary label: 1 = cancer, 0 = normal
    df["label"] = df["cancer"]

    df["filepath"] = df["filename"].apply(
        lambda x: os.path.join(data_dir, x)
    )

    return df[["filepath", "label"]]

train_df = load_dataframe(TRAIN_DIR)
valid_df = load_dataframe(VALID_DIR)
test_df  = load_dataframe(TEST_DIR)

# =========================================================
# 5. CLASS IMBALANCE HANDLING
# =========================================================
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_df["label"].values
)

class_weights = {
    0: class_weights_array[0],
    1: class_weights_array[1]
}

print("Class Weights:", class_weights)

# =========================================================
# 6. IMAGE PIPELINE
# =========================================================
def process_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label

def make_dataset(df, training=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (df["filepath"].values, df["label"].values)
    )
    ds = ds.map(process_image, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(1024)
        ds = ds.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y),
            num_parallel_calls=AUTOTUNE
        )

    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_dataset(train_df, training=True)
valid_ds = make_dataset(valid_df)
test_ds  = make_dataset(test_df)

# =========================================================
# 7. CBAM ATTENTION MODULE
# =========================================================
def cbam_block(feature_map, ratio=8):
    channel = feature_map.shape[-1]

    shared_dense_1 = layers.Dense(channel // ratio, activation="relu")
    shared_dense_2 = layers.Dense(channel)

    avg_pool = layers.GlobalAveragePooling2D()(feature_map)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_2(shared_dense_1(avg_pool))

    max_pool = layers.GlobalMaxPooling2D()(feature_map)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_2(shared_dense_1(max_pool))

    channel_attention = layers.Add()([avg_pool, max_pool])
    channel_attention = layers.Activation("sigmoid")(channel_attention)
    channel_refined = layers.Multiply()([feature_map, channel_attention])

    spatial_attention = layers.Conv2D(
        1, kernel_size=7, padding="same", activation="sigmoid"
    )(channel_refined)

    return layers.Multiply()([channel_refined, spatial_attention])

# =========================================================
# 8. MODEL DEFINITION
# =========================================================
def build_model():
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs)

model = build_model()

# =========================================================
# 9. COMPILE MODEL
# =========================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

# =========================================================
# 10. CALLBACKS (RESUMABLE TRAINING)
# =========================================================
backup_callback = tf.keras.callbacks.BackupAndRestore(
    backup_dir=BACKUP_DIR
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    MODEL_CKPT_PATH,
    monitor="val_auc",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    patience=5,
    restore_best_weights=True
)

# =========================================================
# 11. TRAINING – PHASE 1
# =========================================================
print("\nPhase 1 Training...\n")

model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=[backup_callback, checkpoint_callback, early_stopping]
)

# =========================================================
# 12. FINE-TUNING – PHASE 2
# =========================================================
print("\nFine-Tuning...\n")

model.get_layer("efficientnetb0").trainable = True
for layer in model.get_layer("efficientnetb0").layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    class_weight=class_weights,
    callbacks=[backup_callback, checkpoint_callback, early_stopping]
)

# =========================================================
# 13. LOAD BEST MODEL AND EVALUATE
# =========================================================
best_model = tf.keras.models.load_model(MODEL_CKPT_PATH)

print("\nEvaluating on Test Set...\n")
best_model.evaluate(test_ds)

# =========================================================
# 14. METRICS, ANALYSIS, AND PLOTS
# =========================================================
y_true = []
y_pred_prob = []

for images, labels in test_ds:
    preds = best_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred_prob.extend(preds.flatten())

y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Classification report
print("\nClassification Report:\n")
print(classification_report(
    y_true, y_pred,
    target_names=["Normal", "Cancer"],
    digits=4
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Normal", "Cancer"],
    yticklabels=["Normal", "Cancer"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.tight_layout()
plt.show()

# Sensitivity and Specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# Save results
with open("test_results.txt", "w") as f:
    f.write("Bone Cancer Classification Results\n\n")
    f.write(classification_report(
        y_true, y_pred,
        target_names=["Normal", "Cancer"],
        digits=4
    ))
    f.write("\n")
    f.write(f"Sensitivity: {sensitivity:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write(f"AUC: {roc_auc:.4f}\n")

print("\nResults saved to test_results.txt")
