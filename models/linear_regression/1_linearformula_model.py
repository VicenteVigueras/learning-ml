import tensorflow as tf
import numpy as np
from tensorflow import keras

"""
Neuron with 1 input and 1 output - Modeling Linear Formula: y = mx + b
"""

# Create the model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Optimizer vs Loss 
model.compile(optimizer='sgd', loss='mean_squared_error')

# Input data 
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Train the model
model.fit(xs, ys, epochs=500)

# Test it
def linear_function(x, m, b):
  return m * x + b

expected = linear_function(10.0, 3.0, 1.0)  
actual = model.predict(np.array([10.0]))[0][0]

print(f"Expected: {expected}, Actual: {actual} ")  

"""
Expected output: 31.0
Extra note: Result will not be 100% accurate, but very close
"""


