import numpy as np
import tensorflow as tf

"""
About MNIST dataset

        ★ 70000 Images
        ★ 10 different items (classes)
        ★ Images are (28,28) {after flattening only a vector with the length 784}
        ★ Image pixel intensities in the range of (0,255)

"""

# load fashion mnist data
mnist = tf.keras.datasets.mnist
(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

# In order to achieve more efficient training process we standardize our data into (0,1) scale.
Xtrain = Xtrain / 255.0
Xtest = Xtest / 255.0

# print shape of data
print(Xtrain.shape)
print(Ytrain.shape)
print(Xtest.shape)
print(Ytest.shape)


# Create Callback inorder to avoid model overfitting


class MyCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nReached 99% train accuracy.So stop training!")
            self.model.stop_training = True


def fashion_mnist_model(x, y):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callback = MyCallBack()
    model.fit(x, y, epochs=150, batch_size=64, callbacks=[callback])
    return model


model = fashion_mnist_model(Xtrain, Ytrain)

# obtain model predictions
Predictions = model.predict(Xtest).squeeze()
Predictions = Predictions.argmax(axis=-1).squeeze()
print(Predictions)

# model evaluation
model.evaluate(Xtest, Ytest)