# import tensorflow & required libraries
import numpy as np
import tensorflow as tf

"""
Consider Y = 2X -1 relation ship.
    X = [-1,0,1,2,3,4]
    Y = [-3,-1,1,3,5,7]

    These X, Y lists satisfy this relationship. Now let's build & train simple perceptron network to
    recognize this pattern ( Y = 2X - 1) between these X & Y arrays.
"""

# Input & Output Data
X = [-1, 0, 1, 2, 3, 4]
Y = [-3, -1, 1, 3, 5, 7]

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)
# create model using sequential API
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

# compile the model
model.compile(
        optimizer='sgd',
        loss='mse'
            )

# train the model
model.fit(X, Y, epochs=500)

# testing the model
x = 10
y = model.predict([x])
print(y)




