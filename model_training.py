import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import keras.utils as image
from pathlib import Path
import numpy as np

train_images = []
train_labels = []
test_images = []
test_labels = []


def load_training_image():
    i = 0
    while i < 32:
        # Child Path to the sub-folder for training data
        training_path = Path(
            "SINGAPORE_TRAFFIC_SIGNS_CLASSIFICATION") / "SINGAPORE_TRAFFIC_SIGNS_CLASSIFICATION" / "TRAIN" / str(i)

        # Load all the training images
        for images in training_path.glob("*.jpg"):
            # Load the image from the path
            img = image.load_img(images, target_size=(64, 64))

            # Convert the image to numpy array
            image_array = image.img_to_array(img)

            # Reshape the array to 4-dims so that it can work with the Keras API
            image_array = image_array.reshape(64, 64, 3)

            # Normalize dataset to 0-to-1 range
            image_array = image_array.astype("float32")
            image_array = image_array / 255

            # Add the image to the training list
            train_images.append(image_array)

            # Add the sub-folder name as training label
            train_labels.append(i)

        i += 1


def load_test_image():
    i = 0
    while i < 32:
        # Child Path to the sub-folder for training data
        testing_path = Path(
            "SINGAPORE_TRAFFIC_SIGNS_CLASSIFICATION") / "SINGAPORE_TRAFFIC_SIGNS_CLASSIFICATION" / "TEST" / str(i)

        # Load all the training images
        for images in testing_path.glob("*.jpg"):
            # Load the image from the path
            img = image.load_img(images, target_size=(64, 64))

            # Convert the image to numpy array
            image_array = image.img_to_array(img)

            # Reshape the array to 4-dims so that it can work with the Keras API
            image_array = image_array.reshape(64, 64, 3)

            # Normalize dataset to 0-to-1 range
            image_array = image_array.astype("float32")
            image_array = image_array / 255

            # Add the image to the training list
            test_images.append(image_array)

            # Add the sub-folder name as training label
            test_labels.append(i)

        i += 1


load_training_image()
load_test_image()

print("Training Image Size:", len(train_images))
print("Training Label Size", len(train_labels))
print("Testing Image Size:", len(test_images))
print("Testing Label Size", len(test_labels))

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, 32)
test_labels = keras.utils.to_categorical(test_labels, 32)

input_shape = (64, 64, 3)

# create sequential model4
model4 = Sequential()

model4.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation="relu"))
model4.add(Conv2D(32, (3, 3), activation="relu"))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.25))

model4.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model4.add(Conv2D(64, (3, 3), activation="relu"))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.25))

model4.add(Flatten())
model4.add(Dense(64, activation='relu'))
model4.add(Dropout(0.25))
model4.add(Dense(128, activation='relu'))
model4.add(Dropout(0.25))
model4.add(Dense(32, activation='softmax'))

model4.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)

# Model summary
model4.summary()

model4.fit(
    train_images,
    train_labels,
    batch_size=32,
    epochs=20,
    validation_data=(test_images, test_labels),
    shuffle=True
)

# Evaluate modal
model4.evaluate(test_images, test_labels)

# Save neural network structure
model_structure = model4.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model4.save_weights("model_weights.h5")
