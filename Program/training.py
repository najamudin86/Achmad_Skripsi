import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Judul Program Skripsi

print("="*60)
print("KLASIFIKASI CITRA WILAYAH PERKOTAAN")
print("MENGGUNAKAN CNN EFFICIENTNETB2")
print("="*60)

# preprocesing dan augmentasi

train_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=10,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2
)

# data latih

train_generator = train_datagen.flow_from_directory(

    'dataset',

    target_size=(224,224),

    batch_size=16,

    class_mode='categorical',

    subset='training',

    shuffle=True
)
print("Class Indices:")
print(train_generator.class_indices)


# data validasi

val_generator = val_datagen.flow_from_directory(

    'dataset',

    target_size=(224,224),

    batch_size=16,

    class_mode='categorical',

    subset='validation',

    shuffle=False
)

# data tes / data uji

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(

    'data_test',

    target_size=(224,224),

    batch_size=16,

    class_mode='categorical',

    shuffle=False
)

# Arsitektur EfficientB2

base_model = EfficientNetB2(

    weights='imagenet',

    include_top=False,

    input_shape=(224,224,3)
)

base_model.trainable = True

for layer in base_model.layers[:-20]:

    layer.trainable = False


x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)

x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)

x = Dropout(0.3)(x)

output = Dense(3, activation='softmax')(x)


model = Model(

    inputs=base_model.input,

    outputs=output
)

# Variasi data

model.compile(

    optimizer=Adam(
        learning_rate=0.0002
    ),

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

callbacks = [

    EarlyStopping(

        monitor='val_loss',

        patience=30,

        restore_best_weights=True,

        verbose=1
    ),

    ModelCheckpoint(

        'best_model.keras',

        monitor='val_accuracy',

        save_best_only=True,

        verbose=1
    )
]

# mentraining model 

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=75,
)

# model di simpan

model.save('model.keras')


# membuat grafik accuracy

plt.figure(figsize=(8,5))

plt.plot(

    history.history['accuracy'],

    label='Training Accuracy'
)

plt.plot(

    history.history['val_accuracy'],

    label='Validation Accuracy'
)

plt.title('Grafik Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.savefig('accuracy.png')


# membuat grafik loss

plt.figure(figsize=(8,5))

plt.plot(

    history.history['loss'],

    label='Training Loss'
)

plt.plot(

    history.history['val_loss'],

    label='Validation Loss'
)

plt.title('Grafik Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.savefig('loss.png')

# prediksi data test

test_generator.reset()

pred = model.predict(test_generator)

pred_classes = np.argmax(

    pred,

    axis=1
)

true_classes = test_generator.classes

class_labels = list(

    test_generator.class_indices.keys()
)

# confusion matrix

cm = confusion_matrix(

    true_classes,

    pred_classes
)

print("\n========== CONFUSION MATRIX ==========\n")

print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=class_labels,

    yticklabels=class_labels
)

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix')

plt.savefig('confusion_matrix.png')

# klasifikasi report

print("\n========== CLASSIFICATION REPORT ==========\n")

print(

    classification_report(

        true_classes,

        pred_classes,

        target_names=class_labels
    )
)

print("\n========== TRAINING SELESAI ==========")

print("Model berhasil disimpan")