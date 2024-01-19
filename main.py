import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import warnings

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier App")

        self.label = tk.Label(root, text="Image Classifier", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.browse_button = tk.Button(root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_image)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=20)

        self.image_label = tk.Label(root)
        self.image_label.pack()

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.result_label.config(text=f"Selected Image: {file_path}")

            # Display the selected image
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            self.image_label.img = img
            self.image_label.config(image=img)

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0

    def predict_image(self):
        if hasattr(self, 'image_path'):
            # Load the pre-trained model
            model = tf.keras.models.load_model("children_adult_classifier.h5")

            # Preprocess the image
            processed_image = self.preprocess_image(self.image_path)

            # Make predictions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictions = model.predict(processed_image)

            # Adjust the threshold value
            threshold = 0.5

            # Display the prediction result
            if predictions[0][0] > threshold:
                result = "The model predicts that the person in the image is a child."
            else:
                result = "The model predicts that the person in the image is not a child."

            self.result_label.config(text=result)
        else:
            self.result_label.config(text="Please select an image first.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
