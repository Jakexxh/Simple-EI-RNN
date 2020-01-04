import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


def MaskMeanSquaredError(y_true, y_pred, masks):
        # y_pred = tf.convert_to_tensor(y_pred,dtype=np.float32)
        # y_true = tf.dtypes.cast(y_true, y_pred.dtype)
        loss =  K.mean(K.square(tf.multiply((y_pred - y_true), tf.expand_dims(masks, 1))))
        return loss

# Test


"""
Test

y = np.array([[0,2,2,0],[0,0,0,0]] )
pred = np.array([[2,2,2,2],[0,0,0,0]])
masks = np.array([0,1,1,1])
lossfun = MaskMeanSquaredError()
error = lossfun.call(y, pred, masks)
print(error)

"""