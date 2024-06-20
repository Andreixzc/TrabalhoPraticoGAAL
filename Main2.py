import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    print("No TPUs detected")
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

INPUT_DIR = "input/butterfly-image-classification"
TRAIN_DIR = f"{INPUT_DIR}/train"
TEST_DIR = f"{INPUT_DIR}/test"

meta_df = pd.read_csv(os.path.join(INPUT_DIR, "Training_set.csv"))

# Initialize and fit the label encoder
label_encoder = LabelEncoder()
meta_df["label_encoding"] = label_encoder.fit_transform(meta_df["label"])

# Split the data
train_df, val_df = train_test_split(meta_df, test_size=0.3, stratify=meta_df["label_encoding"])

def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image, label

def load_dataset(dataframe, directory):
    image_paths = [os.path.join(directory, fname) for fname in dataframe["filename"]]
    labels = dataframe["label_encoding"].values
    return tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(preprocess_image)

train_dataset = load_dataset(train_df, TRAIN_DIR).shuffle(len(train_df)).batch(64).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_df, TRAIN_DIR).batch(64).prefetch(tf.data.AUTOTUNE)

with strategy.scope():
    efficient_net = hub.KerasLayer("https://tfhub.dev/google/efficientnet/b3/feature-vector/1", trainable=False)
    
    model = keras.Sequential([
        efficient_net,
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(len(label_encoder.classes_), activation='softmax'),
    ])
    
    model.build([None, 224, 224, 3])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

EPOCHS = 5
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)

# Save the trained model
model.save('butterfly_classifier_model.h5')

# Evaluation and visualization
test_dataset = load_dataset(meta_df, TEST_DIR).batch(1).prefetch(tf.data.AUTOTUNE)

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25))
sample_test_dataset = val_dataset.take(25)
predictions = model.predict(sample_test_dataset).argmax(axis=1)

for ax, (image, act_label_encoding), pred_label_encoding in zip(axes.flat, sample_test_dataset, predictions):
    actual_label_encoding_np = act_label_encoding.numpy().astype(int)
    actual_label_name = label_encoder.inverse_transform([actual_label_encoding_np])[0]
    pred_label_encoding_np = pred_label_encoding.astype(int)
    pred_label_name = label_encoder.inverse_transform([pred_label_encoding_np])[0]
    ax.imshow(image.numpy().reshape(224, 224, 3))
    ax.set(title=f"Predicted: {pred_label_name}\nActual: {actual_label_name}")
    ax.set_xticks([])
    ax.set_yticks([])

test_predictions = model.predict(test_dataset).argmax(axis=1).astype(int)
test_actual = [label_encoding.numpy() for _, label_encoding in test_dataset]

print(f"Accuracy: {accuracy_score(test_predictions, test_actual)}")
print(f"Precision: {precision_score(test_predictions, test_actual, average='macro')}")
print(f"Recall: {recall_score(test_predictions, test_actual, average='macro')}")
print(f"F1 score: {f1_score(test_predictions, test_actual, average='macro')}")
