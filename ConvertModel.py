import tensorflow as tf
import tensorflowjs as tfjs

# Path to the saved Keras model
keras_model_path = 'butterfly_classifier_model.h5'

# Path where the TensorFlow.js model will be saved
tfjs_target_dir = 'tfjs_model'

# Load the Keras model
model = tf.keras.models.load_model(keras_model_path, custom_objects={'KerasLayer': tf.keras.layers.Layer})

# Convert the model to TensorFlow.js format
tfjs.converters.save_keras_model(model, tfjs_target_dir)

print(f"Model successfully converted and saved to {tfjs_target_dir}")
