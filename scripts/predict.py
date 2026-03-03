import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("models/mnist_model.h5")

# Load sample image (28x28 grayscale)
img = Image.open("sample_digit.png").convert("L")
img = img.resize((28,28))
img_array = np.array(img).reshape(1,28,28,1) / 255.0

# Predict
prediction = model.predict(img_array)
digit = np.argmax(prediction)
print("Predicted Digit:", digit)