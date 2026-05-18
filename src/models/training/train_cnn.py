from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)

# Display the first image and its label
plt.imshow(x_train[0], cmap='gray')
plt.title('Label: {}'.format(y_train[0]))
plt.show()

# Normalize the pixel values to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape images for CNN input (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)



# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
# Conv2D learns images feature through egdes curves and shapes.
# MaxPooling2D reduces the images size
# Flatten turns images features into a list
# Dense fianle decison maker layer 
# Softmax gives ouput probailoties for each class (0-9)


# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(x_train, y_train, epochs = 5) # The AI looks at the images 5 times to learn the patterns
# Compares preditions to correct answers 
# improves itselft repeatedlt

# Evaluate accuracy
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy:", accuracy)
# expected result: 97 - 99% accuracy
# usually for CNN on MNIST

# Save the model
model.save("src/models/digit_cnn_model.h5")