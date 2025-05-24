import glob
import os

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.losses import CategoricalCrossentropy
from keras.src.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json

from keras.regularizers import l2
from keras.src.optimizers import Adam
from sklearn.utils import class_weight

from properties import HUMAN_MODEL_PATH, HUMAN_DATASET_PATH, ANIMAL_DATASET_PATH, ANIMAL_MODEL_PATH, \
    HUMAN_BEST_MODEL_PATH, ANIMAL_BEST_MODEL_PATH

HUMAN_MODEL_PATH = '../' + HUMAN_MODEL_PATH
HUMAN_DATASET_PATH = '../' + HUMAN_DATASET_PATH
ANIMAL_DATASET_PATH = '../' + ANIMAL_DATASET_PATH
ANIMAL_MODEL_PATH = '../' + ANIMAL_MODEL_PATH
HUMAN_BEST_MODEL_PATH = '../' + HUMAN_BEST_MODEL_PATH
ANIMAL_BEST_MODEL_PATH = '../' + ANIMAL_BEST_MODEL_PATH

HUMAN_DATASET_AFF_PATH = HUMAN_DATASET_PATH.rstrip('/') + '_affectnet/'

def build_model(input_shape=(48, 48, 1), num_classes=7):
    inputs = Input(input_shape)

    x = Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


def train_model(is_human: bool = True):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    if is_human:
        def make_dataframe(root_dir):
            records = []
            for cls in os.listdir(root_dir):
                class_dir = os.path.join(root_dir, cls)
                if not os.path.isdir(class_dir): continue
                for img_path in glob.glob(os.path.join(class_dir, '*.*')):
                    records.append({'filename': img_path, 'class': cls})
            return pd.DataFrame(records)

        df_train_human = make_dataframe(HUMAN_DATASET_PATH + 'train/')
        df_train_aff = make_dataframe(HUMAN_DATASET_AFF_PATH + 'train/')
        df_train = pd.concat([df_train_human, df_train_aff], ignore_index=True)

        df_val_human = make_dataframe(HUMAN_DATASET_PATH + 'test/')
        df_val_aff = make_dataframe(HUMAN_DATASET_AFF_PATH + 'test/')
        df_val = pd.concat([df_val_human, df_val_aff], ignore_index=True)


        train_gen = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            x_col='filename', y_col='class',
            target_size=(48, 48), color_mode='grayscale',
            batch_size=64, class_mode='categorical',
            shuffle=True
        )
        val_gen = val_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col='filename', y_col='class',
            target_size=(48, 48), color_mode='grayscale',
            batch_size=64, class_mode='categorical',
            shuffle=False
        )
    else:
        train_gen = train_datagen.flow_from_directory(
            (HUMAN_DATASET_PATH if is_human else ANIMAL_DATASET_PATH) + 'train/',
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=64,
            class_mode='categorical',
            shuffle=True
        )

        val_gen = val_datagen.flow_from_directory(
            (HUMAN_DATASET_PATH if is_human else ANIMAL_DATASET_PATH) + 'test/',
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=64,
            class_mode='categorical',
            shuffle=False
        )

    # Save mapping for inference
    with open(('human' if is_human else 'animal') + '_class_indices.json', 'w') as file:
        json.dump(train_gen.class_indices, file)

    model = build_model(num_classes=len(train_gen.class_indices))
    model.compile(optimizer=Adam(1e-3),
                  loss=CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=['accuracy', 'Precision', 'Recall'])


    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(HUMAN_BEST_MODEL_PATH if is_human else ANIMAL_BEST_MODEL_PATH, monitor='val_loss', save_best_only=True)
    ]

    unique_classes = np.unique(train_gen.classes)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_gen.classes
    )
    class_weight_dict = {
        int(cls): float(wt)
        for cls, wt in zip(unique_classes, weights)
    }

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    model.save(HUMAN_MODEL_PATH if is_human else ANIMAL_MODEL_PATH, overwrite=True)

if __name__ == '__main__':
    train_model(is_human=True)
