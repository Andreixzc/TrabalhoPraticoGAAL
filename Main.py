import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random

import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from keras.layers import Dense, Dropout, Conv2D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tqdm import tqdm

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    print("No TPUs detected")
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)


INPUT_DIR = "input/butterfly-image-classification"
TRAIN_DIR = f"{INPUT_DIR}/train"
TEST_DIR = f"{INPUT_DIR}/test"

# we will only use the training set provided, split this into a training and validation
# set, and use these to train our model
meta_df = pd.read_csv(os.path.join(INPUT_DIR, "Training_set.csv"))
meta_df.head()



images = []
labels = []
label_encodings = []

label_encoder = LabelEncoder()
meta_df["label_encoding"] = label_encoder.fit_transform(meta_df["label"])
pbar = tqdm(list(meta_df.iterrows()))

for index, entry in pbar: 
    image = np.asarray(Image.open(os.path.join(TRAIN_DIR, entry["filename"])))
    label = entry["label"]
    label_encoding = entry["label_encoding"]
    
    images.append(image)
    labels.append(label)
    label_encodings.append(label_encoding)




with strategy.scope():
    efficient_net = hub.KerasLayer("https://www.kaggle.com/models/google/efficientnet-v2/frameworks/TensorFlow2/variations/imagenet21k-b3-feature-vector/versions/1",
                   trainable=False)
    
    model = keras.Sequential([
        efficient_net,
#         keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(75, activation='softmax'),
    ])
    
    model.build([None, 224, 224, 3])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])


np_images = np.asarray(images).astype('float64') / 255
np_label_encodings = np.asarray(label_encodings).astype('float64')
print(f"Shape of images: {np_images.shape}")
print(f"Shape of label encodings: {np_label_encodings.shape}")
images_train, images_test, label_encodings_train, label_encodings_test = train_test_split(np_images, np_label_encodings, train_size=0.7)
print(f"Shape of training images: {images_train.shape}")
print(f"Shape of training label encodings: {label_encodings_train.shape}")
print(f"Shape of validation images: {images_test.shape}")
print(f"Shape of validation label encodings: {label_encodings_test.shape}")
train_dataset = tf.data.Dataset.from_tensor_slices((images_train, label_encodings_train)).repeat().shuffle(10000).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((images_test, label_encodings_test)).batch(1)
EPOCHS = 5
STEPS_PER_EPOCH = 2000
VALIDATION_STEPS = 1000

history = model.fit(train_dataset, epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=test_dataset)


fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25,25))

sample_test_dataset = test_dataset.take(25)
sample_test_dataset_np = [(image, label_encoding) for (image, label_encoding) in sample_test_dataset]
predictions = model.predict(sample_test_dataset).argmax(axis=1)

for ax, (image, act_label_encoding), pred_label_encoding in zip(axes.flat, sample_test_dataset, predictions):
    actual_label_encoding_np = act_label_encoding.numpy().astype(int)
    actual_label_name = label_encoder.inverse_transform(actual_label_encoding_np)[0]
    
    pred_label_encoding_np = pred_label_encoding.astype(int)
    pred_label_name = label_encoder.inverse_transform([pred_label_encoding_np])[0]
    
    ax.imshow(image.numpy().reshape(224, 224, 3))
    ax.set(title=f"Predicted: {pred_label_name}\nActual: {actual_label_name}")
    ax.set_xticks([])
    ax.set_yticks([])

test_predictions = model.predict(test_dataset).argmax(axis=1).astype(int)
test_actual = [label_encoding.numpy()[0].astype(int) for (image, label_encoding) in test_dataset]

print(f"Accuracy: {accuracy_score(test_predictions, test_actual)}")
print(f"Precision: {precision_score(test_predictions, test_actual, average='macro')}")
print(f"Recall: {recall_score(test_predictions, test_actual, average='macro')}")
print(f"F1 score: {f1_score(test_predictions, test_actual, average='macro')}")