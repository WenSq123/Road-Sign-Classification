from keras.models import model_from_json
from pathlib import Path
import keras.utils as image
import numpy as np
import os

# These are the CIFAR10 class labels from the training data (in order from 0 to 9)
class_labels = [
    "End of expressway",
    "No jaywalking",
    "Speed limit enforcement cameras",
    "Speed limit of 90km/hr",
    "Turn left",
    "Electronic Road Pricing (road toll) gantry ahead",
    "Speed limit of 70km/hr",
    "U-turn lane",
    "Split-way",
    "Stop",
    "Speed limit of 50km/hr",
    "Right Chevron",
    "Crosswalk",
    "Rain shelter for motorcyclists",
    "No entry for all vehicular traffic",
    "Keep left",
    "Coupon parking",
    "Pedestrian use crossing",
    "Restricted Zone ahead",
    "Left Chevron",
    "Expressway",
    "Give Way",
    "No vehicles over height 4.5m",
    "Speed limit of 40km/hr",
    "Maintain a slow speed to anticipate hazards ahead",
    "Bump ahead",
    "No left turn",
    "One-way traffic in right",
    "One-way traffic in left",
    "Slow down",
    "Lanes merge ahead",
    "No right turn"
]

image_label_dict = {1: "No jaywalking",
                    2: "Speed limit of 40km/hr",
                    3: "Speed limit enforcement cameras",
                    4: "Speed limit of 90km/hr",
                    5: "Bump ahead",
                    6: "Expressway",
                    7: "Expressway",
                    8: "U-turn lane",
                    9: "Split-way",
                    10: "No left turn"}

# Load the json file that contains the model's structure
f = Path("model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("model_weights.h5")

# Change directory
os.chdir('Test Model Image')

i = 1
true = 0
# for images in path.glob("*.jpg"):
while i < 11:
    # Load an image file to test, resizing it to 64x64 pixels (as required by this model)
    img = image.load_img("Test " + str(i) + ".jpg", target_size=(64, 64))

    # Convert the image to a numpy array
    image_to_test = image.img_to_array(img)

    # Normalize dataset to 0-to-1 range
    image_to_test = image_to_test.astype("float32")
    image_to_test = image_to_test / 255

    # Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
    list_of_images = np.expand_dims(image_to_test, axis=0)

    # Make a prediction using the model
    results = model.predict(list_of_images)

    # Since only testing one image per loop, only need to check the first result
    single_result = results[0]

    # Get a likelihood score for all 10 possible classes. Find out which class had the highest score.
    most_likely_class_index = int(np.argmax(single_result))
    class_likelihood = single_result[most_likely_class_index]

    # Get the name of the most likely class
    class_label = class_labels[most_likely_class_index]

    # Print the result
    print("Test " + str(i) + ".jpg" + " is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))

    # Check if the result is correct prediction
    print((class_label == image_label_dict[i]), "prediction")

    if class_label == image_label_dict[i]:
        true += 1

    i += 1

print(str(true) + " out of " + str(i-1) + " is correct prediction")
