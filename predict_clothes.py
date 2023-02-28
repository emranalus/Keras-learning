import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

targets = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocessing to map every grayscale pixel(0-255) between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the neural net model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),             # Flatten layer takes matrix as input and generates 28x28 input neurons in a single layer hence the name 'Flatten'.
    keras.layers.Dense(128, activation='relu'),             # Dense layer is all connected neurons amount of neurons are variable but usually lower than amount of input neurons
    keras.layers.Dense(len(targets), activation='softmax')  # Output layer densely connected amount of neurons is equal to length to the target labels length
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=1) 

# Run predictions then print the given one
predictions = model.predict(test_images)

pred_index = 5001

print("\nPrediction:", targets[np.argmax(predictions[pred_index])])

plt.figure()
plt.imshow(test_images[pred_index])
plt.colorbar()
plt.grid(False)
plt.show()
