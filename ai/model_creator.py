from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.layers import BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
import json


from properties import HUMAN_MODEL_PATH, HUMAN_DATASET_PATH, ANIMAL_DATASET_PATH, ANIMAL_MODEL_PATH, \
    HUMAN_BEST_MODEL_PATH, ANIMAL_BEST_MODEL_PATH


def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(is_human: bool = True):
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(
        (HUMAN_DATASET_PATH if is_human else ANIMAL_DATASET_PATH) + 'train/',
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        (HUMAN_DATASET_PATH if is_human else ANIMAL_DATASET_PATH) + 'test/',
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Save mapping for inference
    with open(('human' if is_human else 'animal') + '_class_indices.json', 'w') as file:
        json.dump(train_gen.class_indices, file)

    model = build_model(num_classes=len(train_gen.class_indices))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    checkpoint = ModelCheckpoint(HUMAN_BEST_MODEL_PATH if is_human else ANIMAL_BEST_MODEL_PATH, monitor='val_loss', save_best_only=True)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=[early_stopping, checkpoint]
    )

    model.save(HUMAN_MODEL_PATH if is_human else ANIMAL_MODEL_PATH, overwrite=True)

if __name__ == '__main__':
    train_model(is_human=False)
