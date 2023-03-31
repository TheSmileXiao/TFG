import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import os

dir = "../dataset/"
folder_names = os.listdir(dir)

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    dir,
    labels="inferred",
    label_mode="int",
    class_names=folder_names,
    color_mode="rgb",
    batch_size=32,
    image_size=(150, 150),
    shuffle=True,
    seed=0,
    validation_split=0.2,
    subset="both",
    interpolation="bilinear",
    follow_links=True,
    crop_to_aspect_ratio=False,
)
print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
print(
    "Number of validation samples: %d" % tf.data.experimental.cardinality(val_ds)
)
#First, instantiate a base model with pre-trained weights.
base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

#Then, freeze the base model.
base_model.trainable = False

#Create a new model on top.
inputs = keras.Input(shape=(150, 150, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
x = keras.layers.Dense(512, activation="relu")(x)

outputs = keras.layers.Dense(1)(x)

#no funciona
model = keras.Model(inputs, outputs)
model.summary()

# Train the model on new data.
# model.compile(optimizer=keras.optimizers.Adam(),
#               loss=keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=[keras.metrics.BinaryAccuracy()])
# model.fit(train_ds, epochs=20, validation_data=val_ds)

#edited
model.compile(optimizer=keras.optimizers.Adam(),
                loss="categorical_crossentropy", 
                metrics=["accuracy"])
model.fit(train_ds, epochs=20, validation_data=val_ds)

# Unfreeze the base model
  #base_model.trainable = True
  #model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
  #              loss=keras.losses.BinaryCrossentropy(from_logits=True),
  #              metrics=[keras.metrics.BinaryAccuracy()])
  #Train end-to-end. Be careful to stop before you overfit!
  #model.fit(new_dataset, epochs=10, validation_data=...)