import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from PIL import Image

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
dataset_dir = 'dataset'

def create_dummy_data():
    """Creates dummy data for testing the pipeline if no dataset exists."""
    if not os.path.exists(dataset_dir):
        print("Dataset directory not found. Creating dummy data for demonstration...")
        categories = ['yes', 'no']
        for cat in categories:
            os.makedirs(os.path.join(dataset_dir, cat), exist_ok=True)
            for i in range(10): # Create 10 dummy images per category
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(os.path.join(dataset_dir, cat, f'dummy_{i}.jpg'))
        print("Dummy data created.")
    else:
        print("Dataset directory found.")

def build_model():
    """Builds the MobileNetV2 based model for Transfer Learning."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x) # 2 classes: Tumor, No Tumor
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train():
    create_dummy_data()
    
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    print("Loading validation data...")
    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    if train_generator.samples == 0:
        print("No images found! Please add images to dataset/yes and dataset/no folders.")
        return

    model = build_model()
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE if train_generator.samples > BATCH_SIZE else 1,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE if validation_generator.samples > BATCH_SIZE else 1,
        epochs=EPOCHS
    )
    
    model.save('model.h5')
    print("Model saved as model.h5")

if __name__ == '__main__':
    train()
