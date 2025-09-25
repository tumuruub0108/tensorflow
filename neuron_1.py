import tensorflow as tf
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
# First 55,000 → training set.
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
# Last 5,000 → validation se
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# hidden layers
tf.random.set_seed(42)

"""
option 1 

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

"""

# option 2
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# rint(model.summary())
# print(model.layers)
hidden1 = model.layers[1]
# print(model.layers[1])
# print(model.get_layer("dense")) # by name

# All the parameters of a layer can be accessed using its get_weights() and set_weights()
weights, biases = hidden1.get_weights()
# print(weights.shape)
# print(biases.shape)


# Compiling the model
# sgd = stochastic gradient descent
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Training and evaluating the model
# history = model.fit(X_train,y_train, epochs=30, validation_data=(X_valid, y_valid))

"""
pd.DataFrame(history.history).plot(
figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
style=["r--", "r--.", "b-", "b-*"])
plt.show()

"""

# print(model.evaluate(X_test, y_test))

"""
sequence[start:stop:step]
    start → index where slicing begins (default = 0).
    stop → index where slicing stops (but not included).
    step → how many elements to skip each time (default = 1).
    
nums = [10, 20, 30, 40, 50]

print(nums[1:4])    # [20, 30, 40] → elements at indices 1,2,3
print(nums[:])      # [10, 20, 30, 40, 50] → copy the whole list
print(nums[::2])    # [10, 30, 50] → every 2nd element
print(nums[-3:])    # [30, 40, 50] → last 3 elements
print(nums[::-1])   # [50, 40, 30, 20, 10] → reverse the list

"""

# Using the model to make predictions
X_new = X_test[:3]
y_proba = model.predict(X_new)
# print(y_proba.round(2))

y_pred = y_proba.argmax(axis=-1)
# sprint(y_pred)
# print(np.array(class_names)[y_pred]) 506