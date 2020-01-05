import matplotlib.pyplot as plt
import numpy as np
import io
import tensorflow as tf


def to_img(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


def plot_confusion_matrix(a, dist=True):

    if dist:
        figure = plt.figure(figsize=(8, 8))
        im = plt.imshow(a, interpolation='nearest', cmap='RdBu')
        plt.colorbar(im)
    else:
        figure = plt.figure(figsize=(12, 6))
        plt.imshow(a, interpolation='nearest', cmap='Blues')

    return to_img(figure)


def plot_dots(x, y):
    figure = plt.figure(figsize=(8, 8))
    plt.xlabel('% coh toward choice 1')
    plt.ylabel('percent choice 1')
    plt.scatter(x, y)

    return to_img(figure)
