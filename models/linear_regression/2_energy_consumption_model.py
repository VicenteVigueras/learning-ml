import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd

"""
TODO: Apply linear regression to energy consumption dataset
Understand how features like temperature, humidity, and time of day affect energy usage
"""

df = pd.read_csv("models/linear_regression/data/train_energy_data.csv")
for column in df.columns:
    print(column)

df = df["Energy Consumption"]
print(df.head())

"""
1. Building Type : Categorical feature representing the type of building.
2. Square Footage : Numeric feature representing the total square footage of the building.
3. Number of Occupants : Numeric feature indicating the number of people occupying the building.
4. Appliances Used: Numeric feature representing the number of appliances used in the building.
5. Average Temperature : Numeric feature representing the average temperature of the building or climate area (in Celsius).
6. Days of the week : Categorical feature representing whether the data point corresponds to a weekday or weekend.
7. Energy Consumption : Numeric target variable representing the energy consumption of the building in kWh (kilowatt-hours). This is the value
"""


# Based on this I'll do Square Footage vs Energy Consumption


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array(df["Square Footage"], dtype=float)
ys = np.array(df["Energy Consumption"], dtype=float)

model.fit(xs, ys, epochs=500)

w, b = model.layers[0].get_weights()
print(f"weight: {w}, intercept{b}")