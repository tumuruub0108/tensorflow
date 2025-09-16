import tensorflow as tf
import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)

layer = layers.Dense(3)
print(layer.weights)


# Call layer on a test input
x = tf.ones((1, 4))
y = layer(x)
print(layer.weights) # Now it has weights, of shape (4, 3) and (3,)