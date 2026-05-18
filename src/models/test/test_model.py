from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np

model = load_model("models/digit_cnn_model.h5")

(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

prediction = model.predict(x_test[0:1])

predicted_digit = np.argmax(prediction)

print("Predicted:", predicted_digit)
print("Actual:", y_test[0])