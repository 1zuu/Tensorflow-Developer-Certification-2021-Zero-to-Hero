import numpy as np
import tensorflow as tf

"""

In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.

So, imagine if house pricing was as easy as a house costs,
            -> 50k + 50k per bedroom,
                        so that a 1 bedroom house costs 100k,
                                a 2 bedroom house costs 150k etc.

"""

X = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([1, 1.5, 2, 2.5, 3, 3.5])


def house_prices(x, y):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
    model.compile(
            optimizer='sgd',
            loss='mse'
                )
    model.fit(x, y, epochs=500)
    return model


model = house_prices(X, Y)
x_pred = 7
y_pred = model.predict([x_pred]).squeeze()
print(y_pred)