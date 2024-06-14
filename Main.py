import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Configurações básicas
batch_size = 32
img_size = (224, 224)
num_epochs = 10
learning_rate = 1e-4

# Caminhos para os dados
train_dir = 'Datasets/train'
test_dir = 'Datasets/test'
train_csv = 'Datasets/Training_set.csv'
test_csv = 'Datasets/Testing_set.csv'

# Carregar dados de treinamento e teste
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Criação dos geradores de dados
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Dividir o conjunto de treinamento em treinamento e validação
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_dir,
    x_col='filename',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_dir,
    x_col='filename',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=test_dir,
    x_col='filename',
    y_col=None,
    target_size=img_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

# Obter o número de classes
num_classes = len(train_generator.class_indices)

# Carregar o modelo EfficientNetB0 pré-treinado
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Congelar as camadas base
base_model.trainable = False

# Adicionar camadas de classificação
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=num_epochs
)

# Descongelar as camadas do modelo base para fine-tuning
base_model.trainable = True

# Recompilar o modelo com uma taxa de aprendizado menor
model.compile(optimizer=Adam(learning_rate=learning_rate / 10),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continuar o treinamento (fine-tuning)
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=num_epochs
)

# Salvar o modelo treinado
model.save('fine_tuned_efficientnet.h5')
