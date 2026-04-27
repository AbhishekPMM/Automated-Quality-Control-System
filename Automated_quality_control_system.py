import pandas as pd
import numpy as np
import cv2  # OpenCV for creating simple images
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 0: Check for GPU ---
print("Checking for GPU...")
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('\nWARNING: GPU device not found. Training will be VERY slow.')
    print('Go to Runtime > Change runtime type and select "GPU".')
else:
    print(f'Found GPU at: {device_name}')


# --- Step 1: Generate Synthetic Dataset ---
# We will create our own "Pass" and "Fail" images.
# "Pass" = A perfect circle
# "Fail" = A circle with a defect (a scratch)

print("\nStep 1: Generating synthetic 'Pass' and 'Fail' images...")
IMG_SIZE = 224  # Standard for ResNet50
NUM_IMAGES = 1000  # 500 Pass, 500 Fail
images = []
labels = []

def create_good_image():
    # Create a blank black image
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # Draw a "good" product (a perfect white circle)
    center = (IMG_SIZE // 2, IMG_SIZE // 2)
    radius = 60
    cv2.circle(img, center, radius, (255, 255, 255), thickness=-1) # -1 fills it
    return img

def create_bad_image():
    # Create a "good" product
    img = create_good_image()
    # Add a "defect" (a random black line or "scratch")
    x1 = np.random.randint(80, 150)
    y1 = np.random.randint(80, 150)
    x2 = np.random.randint(x1 + 10, x1 + 40)
    y2 = np.random.randint(y1 + 10, y1 + 40)
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=5)
    return img

for _ in range(NUM_IMAGES // 2):
    images.append(create_good_image())
    labels.append(0)  # 0 = Pass (Good)

for _ in range(NUM_IMAGES // 2):
    images.append(create_bad_image())
    labels.append(1)  # 1 = Fail (Defect)

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

print(f"Generated {X.shape[0]} images ({np.sum(y)} defects).")


# --- Step 2: Preprocessing and Splitting ---
print("\nStep 2: Splitting and preprocessing data...")

# Split: 70% Train, 15% Validation, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# IMPORTANT: Apply ResNet50 preprocessing
# This normalizes the images in the specific way ResNet50 was trained
X_train_p = preprocess_input(X_train)
X_val_p = preprocess_input(X_val)
X_test_p = preprocess_input(X_test)


# --- Step 3: Build the Transfer Learning Model ---
print("\nStep 3: Building the ResNet50 (Transfer Learning) model...")

# Load the base ResNet50 model, pre-trained on ImageNet
# We don't include the final classification layer (include_top=False)
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base model layers (so we only train our new "head")
base_model.trainable = False

# Create our custom "head"
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)  # Run in inference mode
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Regularization
x = Dense(256, activation='relu')(x)
# Final output layer: 1 neuron with sigmoid for Pass/Fail
outputs = Dense(1, activation='sigmoid')(x)

# Build the final model
model = Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- Step 4: Train the Model ---
print("\nStep 4: Training the model...")

EPOCHS = 10  # 10 is fast and should be enough for this simple problem
history = model.fit(
    X_train_p, y_train,
    epochs=EPOCHS,
    validation_data=(X_val_p, y_val),
    batch_size=32
)

# Plot training history
pd.DataFrame(history.history).plot(figsize=(10, 5))
plt.title("Model Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.grid(True)
plt.show()


# --- Step 5: Evaluate the Model on Test Data ---
print("\nStep 5: Evaluating the model on the (unseen) test set...")

# Get probability predictions (0.0 to 1.0)
y_pred_proba = model.predict(X_test_p)
# Convert probabilities to binary class (0 or 1)
y_pred = (y_pred_proba > 0.5).astype(int)

# Print the Classification Report (Precision, Recall, F1-Score)
print(classification_report(y_test, y_pred, target_names=['Pass (Class 0)', 'Fail (Class 1)']))

# Generate and plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Pass', 'Predicted Fail'],
            yticklabels=['Actual Pass', 'Actual Fail'])
plt.title('Confusion Matrix', fontsize=16)
plt.show()


# --- Step 6: Show Qualitative (Visual) Results ---
print("\nStep 6: Showing visual results on test images...")

# We use the *original* X_test images for plotting, not the preprocessed ones
plt.figure(figsize=(15, 8))
for i in range(10):  # Show 10 random examples
    ax = plt.subplot(2, 5, i + 1)
    # Select a random image from the test set
    idx = np.random.randint(0, len(X_test))

    # Get the actual label and predicted label
    actual_label = "Pass" if y_test[idx] == 0 else "Fail"
    pred_label = "Pass" if y_pred[idx] == 0 else "Fail"

    # Set title color
    color = "green" if actual_label == pred_label else "red"

    plt.imshow(X_test[idx])
    plt.title(f"Actual: {actual_label}\nPredicted: {pred_label}", color=color)
    plt.axis("off")

plt.tight_layout()
plt.show()