import matplotlib.pyplot as plt
import numpy as np
import io
import tensorflow as tf
import itertools

def to_img(figure):
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


def plot_confusion_matrix(cm):
    figure = plt.figure(figsize=(8, 8))
    im = plt.imshow(cm, cmap='hot', interpolation='nearest')
    plt.colorbar(im)

    return to_img(figure)

def plot_dots(x, y):
    figure = plt.figure(figsize=(8, 8))
    plt.scatter(x, y)

    return to_img(figure)