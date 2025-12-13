import tensorflow as tf
import numpy as np
from tensorflow import keras


# Basics on how to train a neural network using tensorflow & numpy
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Optimizer vs Loss
model.compile(optimizer='sgd', loss='mean_squared_error')

# Give it some data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Run it 500 times
model.fit(xs, ys, epochs=500)

# Test it
print(model.predict(np.array([10.0])))

# Extra note: Result will not be 100% accurate, but very close

