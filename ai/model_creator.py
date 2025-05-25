import glob
import os

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.layers import BatchNormalization, Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, \
    SeparableConv2D, Add, Activation
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
from tensorflow.keras import backend as K
from keras.src.optimizers import Adam, AdamW
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

def conv_bn_act(x, filters, k=3, s=1):
    x = Conv2D(filters, k, strides=s, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    return Activation('relu')(x)

def sep_res_block(x, filters):
    shortcut = conv_bn_act(x, filters, k=1, s=2)

    x = SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    x = Add()([x, shortcut])
    return x

def build_mini_xception(input_shape=(48, 48, 1), num_classes=7):
    inp = Input(shape=input_shape)
    x   = conv_bn_act(inp, 32, 3, 1)
    x   = conv_bn_act(x,   64, 3, 2)

    for f in [128, 256, 512, 728]:
        x = sep_res_block(x, f)

    x = GlobalAveragePooling2D()(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inp, outputs=out)

def categorical_focal_loss(alpha=0.25, gamma=2.0):
    """
      FL = -alpha * (1 - p_t)**gamma * log(p_t)

    Args:
      alpha: balancing factor, default 0.25
      gamma: focusing parameter, default 2.0

    Returns:
      a loss function expecting y_true, y_pred one-hot.
    """

    def loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        # compute cross-entropy term: -y_true * log(p)
        ce = -y_true * K.log(y_pred)
        # compute weighting term: alpha * (1 - p)^gamma
        weight = alpha * K.pow(1.0 - y_pred, gamma)
        # combine
        fl = weight * ce
        # sum over classes
        return K.sum(fl, axis=-1)

    return loss

def train_model(is_human: bool = True):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=10,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        fill_mode='nearest',
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

    model = build_mini_xception(num_classes=len(train_gen.class_indices))
    optimizer = AdamW(learning_rate=0.003, weight_decay=5e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss=categorical_focal_loss(alpha=0.25, gamma=2.0),
                  metrics=['accuracy', 'Precision', 'Recall', 'F1Score'])

    if is_human and os.path.exists('../models/curr_best/human_best_model.keras'):
        model.load_weights('../models/curr_best/human_best_model.keras')


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
