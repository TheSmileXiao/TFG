import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

tfds.disable_progress_bar()

train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # Reserve 10% for validation and 10% for test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Include labels
)
print(train_ds)
print(validation_ds)
print(test_ds)
print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
print(
    "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
)
print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))

for image, label in train_ds.take(1):
  print("Image shape: ", image.shape)
  print("Label: ", label.numpy())


  #plt.figure(figsize=(10, 10))
