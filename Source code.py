import warnings

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model("children_adult_classifier.h5")

# Function to preprocess an image before making predictions
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize the pixel values


image_path = "child.jpg"
processed_image = preprocess_image(image_path)

# Make predictions
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    predictions = model.predict(processed_image)

# Adjust the threshold value
threshold = 0.5

# Display the image
img = image.load_img(image_path)
plt.imshow(img)
plt.axis('off')  # Turn off axis labels
plt.show()

# Display the prediction result
if predictions[0][0] > threshold:
    print("The model predicts that the person in the image is a child.")
else:
    print("The model predicts that the person in the image is not a child.")
