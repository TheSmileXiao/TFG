# from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# Directorio que contiene las carpetas de imágenes

# folder_names = os.listdir(dir)

# def load_dataset(directory):
#     datagen = ImageDataGenerator(rescale=1./255)
#     dataset = datagen.flow_from_directory(directory=directory,
#                                            target_size=(150, 150),
#                                            class_mode='categorical',
#                                            shuffle=True,
#                                            seed=0,
#                                            classes=folder_names
#                                            )
#     return dataset
# dataset = load_dataset(dir)

# for data_batch, labels_batch in dataset:
#     image = data_batch[0]
#     label = labels_batch[0]
#     plt.imshow(image)
#     plt.title(f'Label: {label}')
#     plt.show()
#     break
import tensorflow as tf
import os
import matplotlib.pyplot as plt

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

for data_batch, labels_batch in train_ds:
    image = data_batch[0]
    label = labels_batch[0]
    plt.imshow(image)
    plt.title(f'Label: {label}')
    plt.show()
    break