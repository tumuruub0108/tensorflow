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
# print(y_pred)
# print(np.array(class_names)[y_pred]) 506

y_new = y_test[:3]
# print(y_new)

# Building a Regression MLP Using the Sequential API
tf.random.set_seed(42)

"""

norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
print(y_pred)

"""


# Building Complex Models Using the Functional API
"""

# Wide & Deep neural network
normalization_layer = tf.keras.layers.Normalization()
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)

input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalized, hidden2])
output = output_layer(concat)

model = tf.keras.Model(inputs=[input_], outputs=[output])

"""

# handling multiple inputs
input_wide = tf.keras.layers.Input(shape=[5]) # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6]) # features 2 to 7

norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()

norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)

hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)

concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)

model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])

print(model) # 511