import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


class MaskMeanSquaredError(tf.losses.Loss):
    def call(self, y_true, y_pred, masks):
        y_pred = tf.convert_to_tensor(y_pred,dtype=np.float32)
        y_true = tf.dtypes.cast(y_true, y_pred.dtype)
        return K.mean(K.square(tf.multiply((y_pred - y_true), masks)))

# Test
# y = np.array([0,2,2,0])
# pred = np.array([2,2,2,2])
# masks = np.array([0,1,1,1])
# # print(np.mean(np.square(y-pred)*masks))
# lossfun = MaskMeanSquaredError()
# error = lossfun.call(y, pred, masks)
# print(error)