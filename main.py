# Main.py in the tensorflow-2-test repo.
# Replace with useful ML code!
import sys
import torch
import tensorflow
import pysptk
import numpy as np
tensorflow.__version__, pysptk.__version__, np.__version__

def main():
# The following is stolen verbatim  from
# https://www.tensorflow.org/tutorials

# Replace with something actually useful
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

if __name__ == '__main__':
    sys.exit(main())