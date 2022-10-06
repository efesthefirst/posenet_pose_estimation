from tensorflow.keras.layers import Input, Dense, Convolution2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import ZeroPadding2D, Dropout, Flatten
from tensorflow.keras.layers import concatenate, Reshape, Activation, BatchNormalization,LSTM, Bidirectional, Layer, ReLU, Convolution2DTranspose
import helper
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import h5py
import math
import tensorflow



class Geoloss_x(Layer):
    def __init__(self, sx):
        super(Geoloss_x, self).__init__()
        self.sx = K.variable(sx)

    def get_vars(self):
        return self.sx

    def custom_loss(self, y_true, y_pred2):
        return tensorflow.math.exp(-self.sx) * K.sqrt(K.sum(K.square(y_true[:,:] - y_pred2[:,:]), axis=1, keepdims=True)) + self.sx

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'sx': self.sx.numpy()
        })
        return config

    def call(self, y_true, y_pred2):
        self.add_loss(self.custom_loss(y_true, y_pred2))
        self.add_metric(self.sx,"sx")
        self.add_metric(self.custom_loss(y_true, y_pred2),"loss_x")
        return y_pred2

class Geoloss_q(Layer):
    def __init__(self, sq):
        super(Geoloss_q, self).__init__()
        self.sq = K.variable(sq)  # or tf.Variable(var1) etc.

    def get_vars(self):
        return self.sq

    def custom_loss(self, y_true, y_pred):
        return tensorflow.math.exp(-self.sq) * K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True)) + self.sq

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'sq': self.sq.numpy()
        })
        return config

    def call(self, y_true, y_pred):
        self.add_loss(self.custom_loss(y_true, y_pred))
        self.add_metric(self.sq, "sq")
        self.add_metric(self.custom_loss(y_true, y_pred),"loss_q")
        return y_pred

class max_loss(Layer):
    def __init__(self):
        super(max_loss, self).__init__()

    def custom_loss(self, y_true_x, y_pred_x, y_true_q, y_pred_q):
        lq = K.sqrt(K.sum(K.square(y_true_x[:, :] - y_pred_x[:, :]), axis=1, keepdims=True))
        q1 = y_true_q[:, :] / tf.norm(y_true_q)
        q2 = y_pred_q[:, :] / tf.norm(y_pred_q)
        d = tf.math.abs(tf.tensordot(q1, q2,axes=[[1], [1]]))
        d=tf.linalg.diag_part(d)

        theta = 2 * tf.math.acos(d) * 180 / math.pi
        loss = tf.math.maximum(theta, lq)
        return loss

    def call(self, y_true_x, y_pred_x, y_true_q, y_pred_q):
        self.add_loss(self.custom_loss(y_true_x, y_pred_x, y_true_q, y_pred_q))
        return y_pred_x, y_true_q

class combined_loss(Layer):
    def __init__(self):
        super(combined_loss, self).__init__()

    def custom_loss(self, y_true_x, y_pred_x, y_true_q, y_pred_q):
        lq = K.sqrt(K.sum(K.square(y_true_x[:, :] - y_pred_x[:, :]), axis=1, keepdims=True))
        q1 = y_true_q[:, :] / tf.norm(y_true_q)
        q2 = y_pred_q[:, :] / tf.norm(y_pred_q)
        d = tf.math.abs(tf.tensordot(q1, q2,axes=[[1], [1]]))
        d=tf.linalg.diag_part(d)

        theta = 2 * tf.math.acos(d) * 180 / math.pi
        loss = theta*2+lq
        return loss

    def lq_loss(self,y_true_x, y_pred_x, y_true_q, y_pred_q):
        lq = K.sqrt(K.sum(K.square(y_true_x[:, :] - y_pred_x[:, :]), axis=1, keepdims=True))
        return lq

    def theta_loss(self,y_true_x, y_pred_x, y_true_q, y_pred_q):
        q1 = y_true_q[:, :] / tf.norm(y_true_q)
        q2 = y_pred_q[:, :] / tf.norm(y_pred_q)
        d = tf.math.abs(tf.tensordot(q1, q2, axes=[[1], [1]]))
        d = tf.linalg.diag_part(d)

        theta = 2 * tf.math.acos(d) * 180 / math.pi
        return theta

    def call(self, y_true_x, y_pred_x, y_true_q, y_pred_q):
        self.add_loss(self.custom_loss(y_true_x, y_pred_x, y_true_q, y_pred_q))
        self.add_metric(self.lq_loss(y_true_x, y_pred_x, y_true_q, y_pred_q),"lq")
        self.add_metric(self.theta_loss(y_true_x, y_pred_x, y_true_q, y_pred_q), "theta")
        return y_pred_x, y_true_q


def euc_loss1x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.3 * lx)

def euc_loss1q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (150 * lq)

def euc_loss2x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.3 * lx)

def euc_loss2q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (150 * lq)

def euc_loss3x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (1 * lx)

def euc_loss3q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (500 * lq)



def create_posenet(bayesian,weights_path=None, tune=False):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/gpu:0'):
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64,(7,7),strides=(2,2),padding='same',activation='relu',name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64,(1,1),padding='same',activation='relu',name='reduction2')(norm1)

        conv2 = Convolution2D(192,(3,3),padding='same',activation='relu',name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96,(1,1),padding='same',activation='relu',name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128,(3,3),padding='same',activation='relu',name='icp1_out1')(icp1_reduction1)


        icp1_reduction2 = Convolution2D(16,(1,1),padding='same',activation='relu',name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32,(5,5),padding='same',activation='relu',name='icp1_out2')(icp1_reduction2)


        icp1_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32,(1,1),padding='same',activation='relu',name='icp1_out3')(icp1_pool)


        icp1_out0 = Convolution2D(64,(1,1),padding='same',activation='relu',name='icp1_out0')(pool2)


        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3],axis=3,name='icp2_in')






        icp2_reduction1 = Convolution2D(128,(1,1),padding='same',activation='relu',name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192,(3,3),padding='same',activation='relu',name='icp2_out1')(icp2_reduction1)


        icp2_reduction2 = Convolution2D(32,(1,1),padding='same',activation='relu',name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96,(5,5),padding='same',activation='relu',name='icp2_out2')(icp2_reduction2)


        icp2_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64,(1,1),padding='same',activation='relu',name='icp2_out3')(icp2_pool)


        icp2_out0 = Convolution2D(128,(1,1),padding='same',activation='relu',name='icp2_out0')(icp2_in)


        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3],axis=3,name='icp2_out')






        icp3_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96,(1,1),padding='same',activation='relu',name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208,(3,3),padding='same',activation='relu',name='icp3_out1')(icp3_reduction1)


        icp3_reduction2 = Convolution2D(16,(1,1),padding='same',activation='relu',name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48,(5,5),padding='same',activation='relu',name='icp3_out2')(icp3_reduction2)


        icp3_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64,(1,1),padding='same',activation='relu',name='icp3_out3')(icp3_pool)


        icp3_out0 = Convolution2D(192,(1,1),padding='same',activation='relu',name='icp3_out0')(icp3_in)


        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3],axis=3,name='icp3_out')






        cls1_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls1_pool')(icp3_out)

        cls1_reduction_pose = Convolution2D(128,(1,1),padding='same',activation='relu',name='cls1_reduction_pose')(cls1_pool)


        cls1_fc1_flat = Flatten()(cls1_reduction_pose)

        cls1_fc1_pose = Dense(1024,activation='relu',name='cls1_fc1_pose')(cls1_fc1_flat)

        cls1_fc_pose_xyz = Dense(3,name='cls1_fc_pose_xyz')(cls1_fc1_pose)

        cls1_fc_pose_wpqr = Dense(4,name='cls1_fc_pose_wpqr')(cls1_fc1_pose)






        icp4_reduction1 = Convolution2D(112,(1,1),padding='same',activation='relu',name='icp4_reduction1')(icp3_out)

        icp4_out1 = Convolution2D(224,(3,3),padding='same',activation='relu',name='icp4_out1')(icp4_reduction1)


        icp4_reduction2 = Convolution2D(24,(1,1),padding='same',activation='relu',name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64,(5,5),padding='same',activation='relu',name='icp4_out2')(icp4_reduction2)


        icp4_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64,(1,1),padding='same',activation='relu',name='icp4_out3')(icp4_pool)


        icp4_out0 = Convolution2D(160,(1,1),padding='same',activation='relu',name='icp4_out0')(icp3_out)


        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3],axis=3,name='icp4_out')






        icp5_reduction1 = Convolution2D(128,(1,1),padding='same',activation='relu',name='icp5_reduction1')(icp4_out)

        icp5_out1 = Convolution2D(256,(3,3),padding='same',activation='relu',name='icp5_out1')(icp5_reduction1)


        icp5_reduction2 = Convolution2D(24,(1,1),padding='same',activation='relu',name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64,(5,5),padding='same',activation='relu',name='icp5_out2')(icp5_reduction2)


        icp5_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64,(1,1),padding='same',activation='relu',name='icp5_out3')(icp5_pool)


        icp5_out0 = Convolution2D(128,(1,1),padding='same',activation='relu',name='icp5_out0')(icp4_out)


        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3],axis=3,name='icp5_out')






        icp6_reduction1 = Convolution2D(144,(1,1),padding='same',activation='relu',name='icp6_reduction1')(icp5_out)

        icp6_out1 = Convolution2D(288,(3,3),padding='same',activation='relu',name='icp6_out1')(icp6_reduction1)


        icp6_reduction2 = Convolution2D(32,(1,1),padding='same',activation='relu',name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64,(5,5),padding='same',activation='relu',name='icp6_out2')(icp6_reduction2)


        icp6_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64,(1,1),padding='same',activation='relu',name='icp6_out3')(icp6_pool)


        icp6_out0 = Convolution2D(112,(1,1),padding='same',activation='relu',name='icp6_out0')(icp5_out)


        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3],axis=3,name='icp6_out')






        cls2_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls2_pool')(icp6_out)

        cls2_reduction_pose = Convolution2D(128,(1,1),padding='same',activation='relu',name='cls2_reduction_pose')(cls2_pool)


        cls2_fc1_flat = Flatten()(cls2_reduction_pose)

        cls2_fc1 = Dense(1024,activation='relu',name='cls2_fc1')(cls2_fc1_flat)

        cls2_fc_pose_xyz = Dense(3,name='cls2_fc_pose_xyz')(cls2_fc1)

        cls2_fc_pose_wpqr = Dense(4,name='cls2_fc_pose_wpqr')(cls2_fc1)






        icp7_reduction1 = Convolution2D(160,(1,1),padding='same',activation='relu',name='icp7_reduction1')(icp6_out)

        icp7_out1 = Convolution2D(320,(3,3),padding='same',activation='relu',name='icp7_out1')(icp7_reduction1)


        icp7_reduction2 = Convolution2D(32,(1,1),padding='same',activation='relu',name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128,(5,5),padding='same',activation='relu',name='icp7_out2')(icp7_reduction2)


        icp7_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128,(1,1),padding='same',activation='relu',name='icp7_out3')(icp7_pool)


        icp7_out0 = Convolution2D(256,(1,1),padding='same',activation='relu',name='icp7_out0')(icp6_out)


        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3],axis=3,name='icp7_out')






        icp8_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160,(1,1),padding='same',activation='relu',name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320,(3,3),padding='same',activation='relu',name='icp8_out1')(icp8_reduction1)


        icp8_reduction2 = Convolution2D(32,(1,1),padding='same',activation='relu',name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128,(5,5),padding='same',activation='relu',name='icp8_out2')(icp8_reduction2)


        icp8_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128,(1,1),padding='same',activation='relu',name='icp8_out3')(icp8_pool)


        icp8_out0 = Convolution2D(256,(1,1),padding='same',activation='relu',name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3],axis=3,name='icp8_out')






        icp9_reduction1 = Convolution2D(192,(1,1),padding='same',activation='relu',name='icp9_reduction1')(icp8_out)

        icp9_out1 = Convolution2D(384,(3,3),padding='same',activation='relu',name='icp9_out1')(icp9_reduction1)


        icp9_reduction2 = Convolution2D(48,(1,1),padding='same',activation='relu',name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128,(5,5),padding='same',activation='relu',name='icp9_out2')(icp9_reduction2)


        icp9_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128,(1,1),padding='same',activation='relu',name='icp9_out3')(icp9_pool)


        icp9_out0 = Convolution2D(384,(1,1),padding='same',activation='relu',name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3],axis=3,name='icp9_out')




        cls3_pool = AveragePooling2D(pool_size=(7,7),strides=(1,1),padding='valid',name='cls3_pool')(icp9_out)

        if bayesian==True:
            cls3_pool=Dropout(0.5)(cls3_pool,training=True)

        cls3_fc1_flat = Flatten()(cls3_pool)

        cls3_fc1_pose = Dense(2048,activation='relu',name='cls3_fc1_pose')(cls3_fc1_flat)


        cls3_fc_pose_xyz = Dense(3,name='cls3_fc_pose_xyz')(cls3_fc1_pose)

        cls3_fc_pose_wpqr = Dense(4,name='cls3_fc_pose_wpqr')(cls3_fc1_pose)






        posenet = Model(input,[cls1_fc_pose_xyz, cls1_fc_pose_wpqr, cls2_fc_pose_xyz, cls2_fc_pose_wpqr, cls3_fc_pose_xyz, cls3_fc_pose_wpqr])

    if tune:
        if weights_path:
            weights_data = np.load(weights_path, encoding='bytes', allow_pickle=True).item()
            for layer in posenet.layers:
                if layer.name in weights_data.keys():
                    layer_weights = weights_data[layer.name]
                    layer.set_weights((layer_weights[b'weights'], layer_weights[b'biases']))

            print("FINISHED SETTING THE WEIGHTS!")

    return posenet

def create_posenet_geo(sx_initial,sq_initial):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/gpu:1'):
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)

        conv2 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)

        icp1_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)

        icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)

        icp1_out0 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

        icp2_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)

        icp2_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)

        icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)

        icp2_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)

        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)

        icp3_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)

        icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)

        icp3_out0 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)

        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

        cls1_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid', name='cls1_pool')(icp3_out)

        cls1_reduction_pose = Convolution2D(128, (1, 1), padding='same', activation='relu', name='cls1_reduction_pose')(
            cls1_pool)

        cls1_fc1_flat = Flatten()(cls1_reduction_pose)

        cls1_fc1_pose = Dense(1024, activation='relu', name='cls1_fc1_pose')(cls1_fc1_flat)

        cls1_fc_pose_xyz = Dense(3, name='cls1_fc_pose_xyz')(cls1_fc1_pose)

        cls1_fc_pose_wpqr = Dense(4, name='cls1_fc_pose_wpqr')(cls1_fc1_pose)

        icp4_reduction1 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(
            icp3_out)

        icp4_out1 = Convolution2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)

        icp4_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)

        icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)

        icp4_out0 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)

        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(
            icp4_out)

        icp5_out1 = Convolution2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)

        icp5_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)

        icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)

        icp5_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)

        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Convolution2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(
            icp5_out)

        icp6_out1 = Convolution2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)

        icp6_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)

        icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)

        icp6_out0 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)

        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        cls2_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid', name='cls2_pool')(icp6_out)

        cls2_reduction_pose = Convolution2D(128, (1, 1), padding='same', activation='relu', name='cls2_reduction_pose')(
            cls2_pool)

        cls2_fc1_flat = Flatten()(cls2_reduction_pose)

        cls2_fc1 = Dense(1024, activation='relu', name='cls2_fc1')(cls2_fc1_flat)

        cls2_fc_pose_xyz = Dense(3, name='cls2_fc_pose_xyz')(cls2_fc1)

        cls2_fc_pose_wpqr = Dense(4, name='cls2_fc_pose_wpqr')(cls2_fc1)

        icp7_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(
            icp6_out)

        icp7_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)

        icp7_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)

        icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)

        icp7_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)

        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)

        icp8_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)

        icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)

        icp8_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(
            icp8_out)

        icp9_out1 = Convolution2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)

        icp9_reduction2 = Convolution2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)

        icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)

        icp9_out0 = Convolution2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)

        cls3_fc1_flat = Flatten()(cls3_pool)

        cls3_fc1_pose = Dense(2048, activation='relu', name='cls3_fc1_pose')(cls3_fc1_flat)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(cls3_fc1_pose)

        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(cls3_fc1_pose)

        y_input_x = Input(shape=(3,), name="y_input_x")
        y_input_q = Input(shape=(4,), name="y_input_q")
        with tf.name_scope("loss_x"):
            my_loss_x = Geoloss_x(sx_initial)(y_input_x, cls3_fc_pose_xyz)
        with tf.name_scope("loss_q"):
            my_loss_q = Geoloss_q(sq_initial)(y_input_q, cls3_fc_pose_wpqr)
        posenet = Model(inputs=[input, y_input_x, y_input_q],
                        outputs=[my_loss_x, my_loss_q])


    return posenet

def create_posenet_max(type):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/gpu:1'):
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)

        conv2 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)

        icp1_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)

        icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)

        icp1_out0 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

        icp2_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)

        icp2_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)

        icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)

        icp2_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)

        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)

        icp3_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)

        icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)

        icp3_out0 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)

        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

        cls1_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid', name='cls1_pool')(icp3_out)

        cls1_reduction_pose = Convolution2D(128, (1, 1), padding='same', activation='relu', name='cls1_reduction_pose')(
            cls1_pool)

        cls1_fc1_flat = Flatten()(cls1_reduction_pose)

        cls1_fc1_pose = Dense(1024, activation='relu', name='cls1_fc1_pose')(cls1_fc1_flat)

        cls1_fc_pose_xyz = Dense(3, name='cls1_fc_pose_xyz')(cls1_fc1_pose)

        cls1_fc_pose_wpqr = Dense(4, name='cls1_fc_pose_wpqr')(cls1_fc1_pose)

        icp4_reduction1 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(
            icp3_out)

        icp4_out1 = Convolution2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)

        icp4_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)

        icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)

        icp4_out0 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)

        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(
            icp4_out)

        icp5_out1 = Convolution2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)

        icp5_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)

        icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)

        icp5_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)

        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Convolution2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(
            icp5_out)

        icp6_out1 = Convolution2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)

        icp6_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)

        icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)

        icp6_out0 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)

        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        cls2_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid', name='cls2_pool')(icp6_out)

        cls2_reduction_pose = Convolution2D(128, (1, 1), padding='same', activation='relu', name='cls2_reduction_pose')(
            cls2_pool)

        cls2_fc1_flat = Flatten()(cls2_reduction_pose)

        cls2_fc1 = Dense(1024, activation='relu', name='cls2_fc1')(cls2_fc1_flat)

        cls2_fc_pose_xyz = Dense(3, name='cls2_fc_pose_xyz')(cls2_fc1)

        cls2_fc_pose_wpqr = Dense(4, name='cls2_fc_pose_wpqr')(cls2_fc1)

        icp7_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(
            icp6_out)

        icp7_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)

        icp7_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)

        icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)

        icp7_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)

        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)

        icp8_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)

        icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)

        icp8_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(
            icp8_out)

        icp9_out1 = Convolution2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)

        icp9_reduction2 = Convolution2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)

        icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)

        icp9_out0 = Convolution2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)

        cls3_fc1_flat = Flatten()(cls3_pool)

        cls3_fc1_pose = Dense(2048, activation='relu', name='cls3_fc1_pose')(cls3_fc1_flat)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(cls3_fc1_pose)

        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(cls3_fc1_pose)

        y_input_x = Input(shape=(3,), name="y_input_x")
        y_input_q = Input(shape=(4,), name="y_input_q")
        with tf.name_scope("loss_x"):
            if type=="max":
                my_loss_x,my_loss_q = max_loss()(y_input_x, cls3_fc_pose_xyz,y_input_q,cls3_fc_pose_wpqr)
            elif type=="combined":
                my_loss_x, my_loss_q = combined_loss()(y_input_x, cls3_fc_pose_xyz, y_input_q, cls3_fc_pose_wpqr)
        posenet = Model(inputs=[input, y_input_x, y_input_q],
                        outputs=[my_loss_x, my_loss_q])


    return posenet


def create_poselstm():
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    if True:
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)

        conv2 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)

        icp1_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)

        icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)

        icp1_out0 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

        icp2_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)

        icp2_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)

        icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)

        icp2_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)

        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)

        icp3_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)

        icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)

        icp3_out0 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)

        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

        icp4_reduction1 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(
            icp3_out)

        icp4_out1 = Convolution2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)

        icp4_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)

        icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)

        icp4_out0 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)

        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(
            icp4_out)

        icp5_out1 = Convolution2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)

        icp5_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)

        icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)

        icp5_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)

        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Convolution2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(
            icp5_out)

        icp6_out1 = Convolution2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)

        icp6_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)

        icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)

        icp6_out0 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)

        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        icp7_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(
            icp6_out)

        icp7_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)

        icp7_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)

        icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)

        icp7_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)

        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)

        icp8_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)

        icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)

        icp8_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(
            icp8_out)

        icp9_out1 = Convolution2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)

        icp9_reduction2 = Convolution2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)

        icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)

        icp9_out0 = Convolution2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)

        cls3_fc1_flat = Flatten()(cls3_pool)

        cls3_fc1_pose = Dense(2048, activation='relu', name='cls3_fc1_pose')(cls3_fc1_flat)
        #lstm
        cls3_fc1_pose_reshaped = tf.reshape(cls3_fc1_pose,[-1,32,64])

        bilstm1, state_h1, state_c1, state_h2, state_c2 = Bidirectional(LSTM(128,name='lstm_lr',return_sequences=True, return_state=True)) (tf.transpose(cls3_fc1_pose_reshaped, perm=[0, 1, 2]))
        bilstm2, state_h3, state_c3, state_h4, state_c4= Bidirectional(LSTM(128, name='lstm_ud', return_sequences=True, return_state=True))(tf.transpose(cls3_fc1_pose_reshaped, perm=[0, 2, 1]))

        lstm_concat = concatenate([state_h1,state_h2,state_h3,state_h4],axis=1)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(lstm_concat)

        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(lstm_concat)

        posenet = Model(input,[cls3_fc_pose_xyz,cls3_fc_pose_wpqr])

    return posenet

def create_poselstm_with_geo_loss(sx_initial,sq_initial):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    if True:
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)

        conv2 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)

        icp1_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)

        icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)

        icp1_out0 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

        icp2_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)

        icp2_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)

        icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)

        icp2_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)

        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)

        icp3_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)

        icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)

        icp3_out0 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)

        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

        icp4_reduction1 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(
            icp3_out)

        icp4_out1 = Convolution2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)

        icp4_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)

        icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)

        icp4_out0 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)

        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(
            icp4_out)

        icp5_out1 = Convolution2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)

        icp5_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)

        icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)

        icp5_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)

        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Convolution2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(
            icp5_out)

        icp6_out1 = Convolution2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)

        icp6_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)

        icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)

        icp6_out0 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)

        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        icp7_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(
            icp6_out)

        icp7_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)

        icp7_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)

        icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)

        icp7_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)

        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)

        icp8_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)

        icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)

        icp8_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(
            icp8_out)

        icp9_out1 = Convolution2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)

        icp9_reduction2 = Convolution2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)

        icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)

        icp9_out0 = Convolution2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)

        cls3_fc1_flat = Flatten()(cls3_pool)

        cls3_fc1_pose = Dense(2048, activation='relu', name='cls3_fc1_pose')(cls3_fc1_flat)
        #lstm
        cls3_fc1_pose_reshaped = tf.reshape(cls3_fc1_pose,[-1,32,64])

        bilstm1, state_h1, state_c1, state_h2, state_c2 = Bidirectional(LSTM(128,name='lstm_lr',return_sequences=True, return_state=True)) (tf.transpose(cls3_fc1_pose_reshaped, perm=[0, 1, 2]))
        bilstm2, state_h3, state_c3, state_h4, state_c4= Bidirectional(LSTM(128, name='lstm_ud', return_sequences=True, return_state=True))(tf.transpose(cls3_fc1_pose_reshaped, perm=[0, 2, 1]))

        lstm_concat = concatenate([state_h1,state_h2,state_h3,state_h4],axis=1)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(lstm_concat)

        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(lstm_concat)

        y_input_x = Input(shape=(3,), name="y_input_x")
        y_input_q = Input(shape=(4,), name="y_input_q")
        with tf.name_scope("loss_x"):
            my_loss_x = Geoloss_x(sx_initial)(y_input_x, cls3_fc_pose_xyz)
        with tf.name_scope("loss_q"):
            my_loss_q = Geoloss_q(sq_initial)(y_input_q, cls3_fc_pose_wpqr)
        model = tf.keras.models.Model(inputs=[input, y_input_x, y_input_q], outputs=[my_loss_x, my_loss_q])

    return model

def create_poselstm_with_max_loss(type):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    if True:
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)

        conv2 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)

        icp1_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)

        icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)

        icp1_out0 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

        icp2_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)

        icp2_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)

        icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)

        icp2_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)

        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)

        icp3_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)

        icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)

        icp3_out0 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)

        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

        icp4_reduction1 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(
            icp3_out)

        icp4_out1 = Convolution2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)

        icp4_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)

        icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)

        icp4_out0 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)

        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(
            icp4_out)

        icp5_out1 = Convolution2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)

        icp5_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)

        icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)

        icp5_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)

        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Convolution2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(
            icp5_out)

        icp6_out1 = Convolution2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)

        icp6_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)

        icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)

        icp6_out0 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)

        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        icp7_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(
            icp6_out)

        icp7_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)

        icp7_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)

        icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)

        icp7_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)

        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)

        icp8_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)

        icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)

        icp8_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(
            icp8_out)

        icp9_out1 = Convolution2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)

        icp9_reduction2 = Convolution2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)

        icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)

        icp9_out0 = Convolution2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)

        cls3_fc1_flat = Flatten()(cls3_pool)

        cls3_fc1_pose = Dense(2048, activation='relu', name='cls3_fc1_pose')(cls3_fc1_flat)
        #lstm
        cls3_fc1_pose_reshaped = tf.reshape(cls3_fc1_pose,[-1,32,64])

        bilstm1, state_h1, state_c1, state_h2, state_c2 = Bidirectional(LSTM(128,name='lstm_lr',return_sequences=True, return_state=True)) (tf.transpose(cls3_fc1_pose_reshaped, perm=[0, 1, 2]))
        bilstm2, state_h3, state_c3, state_h4, state_c4= Bidirectional(LSTM(128, name='lstm_ud', return_sequences=True, return_state=True))(tf.transpose(cls3_fc1_pose_reshaped, perm=[0, 2, 1]))

        lstm_concat = concatenate([state_h1,state_h2,state_h3,state_h4],axis=1)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(lstm_concat)

        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(lstm_concat)

        y_input_x = Input(shape=(3,), name="y_input_x")
        y_input_q = Input(shape=(4,), name="y_input_q")
        with tf.name_scope("loss_x"):
            if type=="max":
                my_loss_x,my_loss_q = max_loss()(y_input_x, cls3_fc_pose_xyz,y_input_q,cls3_fc_pose_wpqr)
            elif type == "combined":
                my_loss_x, my_loss_q = combined_loss()(y_input_x, cls3_fc_pose_xyz, y_input_q, cls3_fc_pose_wpqr)
        posenet = Model(inputs=[input, y_input_x, y_input_q],
                        outputs=[my_loss_x, my_loss_q])

    return posenet


def create_poselstm_with_2_lstm():
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    if True:
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)

        conv2 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)

        icp1_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)

        icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)

        icp1_out0 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

        icp2_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)

        icp2_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)

        icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)

        icp2_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)

        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)

        icp3_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)

        icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)

        icp3_out0 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)

        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

        icp4_reduction1 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(
            icp3_out)

        icp4_out1 = Convolution2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)

        icp4_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)

        icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)

        icp4_out0 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)

        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(
            icp4_out)

        icp5_out1 = Convolution2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)

        icp5_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)

        icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)

        icp5_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)

        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Convolution2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(
            icp5_out)

        icp6_out1 = Convolution2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)

        icp6_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)

        icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)

        icp6_out0 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)

        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        icp7_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(
            icp6_out)

        icp7_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)

        icp7_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)

        icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)

        icp7_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)

        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)

        icp8_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)

        icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)

        icp8_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(
            icp8_out)

        icp9_out1 = Convolution2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)

        icp9_reduction2 = Convolution2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)

        icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)

        icp9_out0 = Convolution2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)

        cls3_fc1_flat = Flatten()(cls3_pool)

        cls3_fc1_flat_reshaped = tf.reshape(cls3_fc1_flat, [-1, 1, 1024])


        bilstm1, state_h1, state_c1 = LSTM(64,name='lstm_pose',return_sequences=True, return_state=True) (cls3_fc1_flat_reshaped)
        bilstm2, state_h2, state_c2= LSTM(64, name='lstm_orientation', return_sequences=True, return_state=True)(cls3_fc1_flat_reshaped)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(state_h1)

        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(state_h2)

        posenet = Model(input,[cls3_fc_pose_xyz,cls3_fc_pose_wpqr])

    return posenet

def create_poselstm_with_2_lstm_with_geo_loss(sx_initial,sq_initial):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    if True:
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)

        conv2 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)

        icp1_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)

        icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)

        icp1_out0 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

        icp2_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)

        icp2_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)

        icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)

        icp2_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)

        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)

        icp3_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)

        icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)

        icp3_out0 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)

        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

        icp4_reduction1 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(
            icp3_out)

        icp4_out1 = Convolution2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)

        icp4_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)

        icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)

        icp4_out0 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)

        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(
            icp4_out)

        icp5_out1 = Convolution2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)

        icp5_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)

        icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)

        icp5_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)

        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Convolution2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(
            icp5_out)

        icp6_out1 = Convolution2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)

        icp6_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)

        icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)

        icp6_out0 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)

        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        icp7_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(
            icp6_out)

        icp7_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)

        icp7_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)

        icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)

        icp7_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)

        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)

        icp8_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)

        icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)

        icp8_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(
            icp8_out)

        icp9_out1 = Convolution2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)

        icp9_reduction2 = Convolution2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)

        icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)

        icp9_out0 = Convolution2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)

        cls3_fc1_flat = Flatten()(cls3_pool)

        cls3_fc1_flat_reshaped = tf.reshape(cls3_fc1_flat, [-1, 1, 1024])


        bilstm1, state_h1, state_c1 = LSTM(64,name='lstm_pose',return_sequences=True, return_state=True) (cls3_fc1_flat_reshaped)
        bilstm2, state_h2, state_c2= LSTM(64, name='lstm_orientation', return_sequences=True, return_state=True)(cls3_fc1_flat_reshaped)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(state_h1)

        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(state_h2)

        y_input_x = Input(shape=(3,), name="y_input_x")
        y_input_q = Input(shape=(4,), name="y_input_q")
        with tf.name_scope("loss_x"):
            my_loss_x = Geoloss_x(sx_initial)(y_input_x, cls3_fc_pose_xyz)
        with tf.name_scope("loss_q"):
            my_loss_q = Geoloss_q(sq_initial)(y_input_q, cls3_fc_pose_wpqr)

        model = tf.keras.models.Model(inputs=[input,y_input_x,y_input_q], outputs=[my_loss_x,my_loss_q] )

    return model

def create_poselstm_with_2_lstm_with_max_loss(type):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    if True:
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)

        conv2 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)

        icp1_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)

        icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)

        icp1_out0 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

        icp2_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)

        icp2_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)

        icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)

        icp2_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)

        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)

        icp3_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)

        icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)

        icp3_out0 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)

        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

        icp4_reduction1 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(
            icp3_out)

        icp4_out1 = Convolution2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)

        icp4_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)

        icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)

        icp4_out0 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)

        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(
            icp4_out)

        icp5_out1 = Convolution2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)

        icp5_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)

        icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)

        icp5_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)

        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Convolution2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(
            icp5_out)

        icp6_out1 = Convolution2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)

        icp6_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)

        icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)

        icp6_out0 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)

        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        icp7_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(
            icp6_out)

        icp7_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)

        icp7_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)

        icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)

        icp7_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)

        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)

        icp8_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)

        icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)

        icp8_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(
            icp8_out)

        icp9_out1 = Convolution2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)

        icp9_reduction2 = Convolution2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)

        icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)

        icp9_out0 = Convolution2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)

        cls3_fc1_flat = Flatten()(cls3_pool)

        cls3_fc1_flat_reshaped = tf.reshape(cls3_fc1_flat, [-1, 1, 1024])


        bilstm1, state_h1, state_c1 = LSTM(64,name='lstm_pose',return_sequences=True, return_state=True) (cls3_fc1_flat_reshaped)
        bilstm2, state_h2, state_c2= LSTM(64, name='lstm_orientation', return_sequences=True, return_state=True)(cls3_fc1_flat_reshaped)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(state_h1)

        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(state_h2)

        y_input_x = Input(shape=(3,), name="y_input_x")
        y_input_q = Input(shape=(4,), name="y_input_q")
        with tf.name_scope("loss_x"):
            if type=="max":
                my_loss_x,my_loss_q = max_loss()(y_input_x, cls3_fc_pose_xyz,y_input_q,cls3_fc_pose_wpqr)
            elif type == "combined":
                my_loss_x, my_loss_q = combined_loss()(y_input_x, cls3_fc_pose_xyz, y_input_q, cls3_fc_pose_wpqr)
        posenet = Model(inputs=[input, y_input_x, y_input_q],
                        outputs=[my_loss_x, my_loss_q])

    return posenet


def create_inception_v3():


    input = Input(shape=(224, 224, 3))

    inc = tf.keras.applications.inception_v3.InceptionV3(input_tensor=input,weights='imagenet', include_top=True)
    last = inc.layers[-2].output
    cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(last)
    cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(last)


    model = tf.keras.models.Model(inputs=input, outputs=[cls3_fc_pose_xyz,cls3_fc_pose_wpqr] )

    return model

def create_inception_v3_with_geo_loss(sx_initial,sq_initial):


    input = Input(shape=(224, 224, 3))

    inc = tf.keras.applications.inception_v3.InceptionV3(input_tensor=input,weights='imagenet', include_top=True)
    last = inc.layers[-2].output
    cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(last)
    cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(last)

    y_input_x = Input(shape=(3,),name="y_input_x")
    y_input_q = Input(shape=(4,),name="y_input_q")
    with tf.name_scope("loss_x"):
        my_loss_x = Geoloss_x(sx_initial)(y_input_x, cls3_fc_pose_xyz)
    with tf.name_scope("loss_q"):
        my_loss_q = Geoloss_q(sq_initial)(y_input_q, cls3_fc_pose_wpqr)
    model = tf.keras.models.Model(inputs=[input,y_input_x,y_input_q], outputs=[my_loss_x,my_loss_q] )

    return model

def create_inception_v3_with_max_loss(type):


    input = Input(shape=(224, 224, 3))

    inc = tf.keras.applications.inception_v3.InceptionV3(input_tensor=input,weights='imagenet', include_top=True)
    last = inc.layers[-2].output
    cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(last)
    cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(last)

    y_input_x = Input(shape=(3,),name="y_input_x")
    y_input_q = Input(shape=(4,),name="y_input_q")
    with tf.name_scope("loss_x"):
        if type=="max":
            my_loss_x, my_loss_q = max_loss()(y_input_x, cls3_fc_pose_xyz, y_input_q, cls3_fc_pose_wpqr)
        elif type == "combined":
            my_loss_x, my_loss_q = combined_loss()(y_input_x, cls3_fc_pose_xyz, y_input_q, cls3_fc_pose_wpqr)
    posenet = Model(inputs=[input, y_input_x, y_input_q],
                    outputs=[my_loss_x, my_loss_q])

    return posenet


def create_vidloc():
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/gpu:1'):
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)

        conv2 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)

        icp1_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)

        icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)

        icp1_out0 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

        icp2_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)

        icp2_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)

        icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)

        icp2_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)

        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)

        icp3_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)

        icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)

        icp3_out0 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)

        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

        cls1_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid', name='cls1_pool')(icp3_out)

        cls1_reduction_pose = Convolution2D(128, (1, 1), padding='same', activation='relu', name='cls1_reduction_pose')(
            cls1_pool)

        cls1_fc1_flat = Flatten()(cls1_reduction_pose)

        cls1_fc1_pose = Dense(1024, activation='relu', name='cls1_fc1_pose')(cls1_fc1_flat)

        cls1_fc_pose_xyz = Dense(3, name='cls1_fc_pose_xyz')(cls1_fc1_pose)

        cls1_fc_pose_wpqr = Dense(4, name='cls1_fc_pose_wpqr')(cls1_fc1_pose)

        icp4_reduction1 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(
            icp3_out)

        icp4_out1 = Convolution2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)

        icp4_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)

        icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)

        icp4_out0 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)

        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(
            icp4_out)

        icp5_out1 = Convolution2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)

        icp5_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)

        icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)

        icp5_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)

        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Convolution2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(
            icp5_out)

        icp6_out1 = Convolution2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)

        icp6_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)

        icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)

        icp6_out0 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)

        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        cls2_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid', name='cls2_pool')(icp6_out)

        cls2_reduction_pose = Convolution2D(128, (1, 1), padding='same', activation='relu', name='cls2_reduction_pose')(
            cls2_pool)

        cls2_fc1_flat = Flatten()(cls2_reduction_pose)

        cls2_fc1 = Dense(1024, activation='relu', name='cls2_fc1')(cls2_fc1_flat)

        cls2_fc_pose_xyz = Dense(3, name='cls2_fc_pose_xyz')(cls2_fc1)

        cls2_fc_pose_wpqr = Dense(4, name='cls2_fc_pose_wpqr')(cls2_fc1)

        icp7_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(
            icp6_out)

        icp7_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)

        icp7_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)

        icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)

        icp7_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)

        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)

        icp8_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)

        icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)

        icp8_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(
            icp8_out)

        icp9_out1 = Convolution2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)

        icp9_reduction2 = Convolution2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)

        icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)

        icp9_out0 = Convolution2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)


        cls3_fc1_flat_reshaped = tf.reshape(cls3_pool, [-1,1, 1024])
        bilstm1, state_h1, state_c1, state_h2, state_c2 = Bidirectional(LSTM(256,name='lstm_1',return_sequences=True, return_state=True)) (cls3_fc1_flat_reshaped)

        lstm_concat = concatenate([state_h1,state_h2],axis=1)
        lstm_concat = tf.reshape(lstm_concat, [-1, 1, 512])


        bilstm2, state_h3, state_c3, state_h4, state_c4 = Bidirectional(
            LSTM(128, name='lstm_2', return_sequences=True, return_state=True))(lstm_concat)

        lstm_concat2 = concatenate([state_h3, state_h4], axis=1)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(lstm_concat2)

        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(lstm_concat2)

        posenet = Model(input,[cls3_fc_pose_xyz,cls3_fc_pose_wpqr])


    return posenet

def create_vidloc_geo(sx_initial,sq_initial):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/gpu:0'):
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)

        conv2 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)

        icp1_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)

        icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)

        icp1_out0 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

        icp2_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)

        icp2_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)

        icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)

        icp2_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)

        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)

        icp3_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)

        icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)

        icp3_out0 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)

        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

        cls1_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid', name='cls1_pool')(icp3_out)

        cls1_reduction_pose = Convolution2D(128, (1, 1), padding='same', activation='relu', name='cls1_reduction_pose')(
            cls1_pool)

        cls1_fc1_flat = Flatten()(cls1_reduction_pose)

        cls1_fc1_pose = Dense(1024, activation='relu', name='cls1_fc1_pose')(cls1_fc1_flat)

        cls1_fc_pose_xyz = Dense(3, name='cls1_fc_pose_xyz')(cls1_fc1_pose)

        cls1_fc_pose_wpqr = Dense(4, name='cls1_fc_pose_wpqr')(cls1_fc1_pose)

        icp4_reduction1 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(
            icp3_out)

        icp4_out1 = Convolution2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)

        icp4_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)

        icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)

        icp4_out0 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)

        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(
            icp4_out)

        icp5_out1 = Convolution2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)

        icp5_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)

        icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)

        icp5_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)

        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Convolution2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(
            icp5_out)

        icp6_out1 = Convolution2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)

        icp6_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)

        icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)

        icp6_out0 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)

        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        cls2_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid', name='cls2_pool')(icp6_out)

        cls2_reduction_pose = Convolution2D(128, (1, 1), padding='same', activation='relu', name='cls2_reduction_pose')(
            cls2_pool)

        cls2_fc1_flat = Flatten()(cls2_reduction_pose)

        cls2_fc1 = Dense(1024, activation='relu', name='cls2_fc1')(cls2_fc1_flat)

        cls2_fc_pose_xyz = Dense(3, name='cls2_fc_pose_xyz')(cls2_fc1)

        cls2_fc_pose_wpqr = Dense(4, name='cls2_fc_pose_wpqr')(cls2_fc1)

        icp7_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(
            icp6_out)

        icp7_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)

        icp7_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)

        icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)

        icp7_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)

        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)

        icp8_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)

        icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)

        icp8_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(
            icp8_out)

        icp9_out1 = Convolution2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)

        icp9_reduction2 = Convolution2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)

        icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)

        icp9_out0 = Convolution2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)


        cls3_fc1_flat_reshaped = tf.reshape(cls3_pool, [-1,1, 1024])
        bilstm1, state_h1, state_c1, state_h2, state_c2 = Bidirectional(LSTM(256,name='lstm_1',return_sequences=True, return_state=True)) (cls3_fc1_flat_reshaped)

        lstm_concat = concatenate([state_h1,state_h2],axis=1)
        lstm_concat = tf.reshape(lstm_concat, [-1, 1, 512])


        bilstm2, state_h3, state_c3, state_h4, state_c4 = Bidirectional(
            LSTM(128, name='lstm_2', return_sequences=True, return_state=True))(lstm_concat)

        lstm_concat2 = concatenate([state_h3, state_h4], axis=1)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(lstm_concat2)

        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(lstm_concat2)

        y_input_x = Input(shape=(3,), name="y_input_x")
        y_input_q = Input(shape=(4,), name="y_input_q")
        with tf.name_scope("loss_x"):
            my_loss_x = Geoloss_x(sx_initial)(y_input_x, cls3_fc_pose_xyz)
        with tf.name_scope("loss_q"):
            my_loss_q = Geoloss_q(sq_initial)(y_input_q, cls3_fc_pose_wpqr)
        posenet = Model(inputs=[input, y_input_x, y_input_q],
                        outputs=[my_loss_x, my_loss_q])


    return posenet

def create_vidloc_max(type):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/gpu:0'):
        input = Input(shape=(224, 224, 3))

        conv1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

        reduction2 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)

        conv2 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)

        icp1_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)

        icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)

        icp1_out0 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

        icp2_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)

        icp2_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)

        icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)

        icp2_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)

        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)

        icp3_reduction2 = Convolution2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)

        icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)

        icp3_out0 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)

        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

        cls1_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid', name='cls1_pool')(icp3_out)

        cls1_reduction_pose = Convolution2D(128, (1, 1), padding='same', activation='relu', name='cls1_reduction_pose')(
            cls1_pool)

        cls1_fc1_flat = Flatten()(cls1_reduction_pose)

        cls1_fc1_pose = Dense(1024, activation='relu', name='cls1_fc1_pose')(cls1_fc1_flat)

        cls1_fc_pose_xyz = Dense(3, name='cls1_fc_pose_xyz')(cls1_fc1_pose)

        cls1_fc_pose_wpqr = Dense(4, name='cls1_fc_pose_wpqr')(cls1_fc1_pose)

        icp4_reduction1 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(
            icp3_out)

        icp4_out1 = Convolution2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)

        icp4_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)

        icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)

        icp4_out0 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)

        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(
            icp4_out)

        icp5_out1 = Convolution2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)

        icp5_reduction2 = Convolution2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)

        icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)

        icp5_out0 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)

        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Convolution2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(
            icp5_out)

        icp6_out1 = Convolution2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)

        icp6_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)

        icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)

        icp6_out0 = Convolution2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)

        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        cls2_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid', name='cls2_pool')(icp6_out)

        cls2_reduction_pose = Convolution2D(128, (1, 1), padding='same', activation='relu', name='cls2_reduction_pose')(
            cls2_pool)

        cls2_fc1_flat = Flatten()(cls2_reduction_pose)

        cls2_fc1 = Dense(1024, activation='relu', name='cls2_fc1')(cls2_fc1_flat)

        cls2_fc_pose_xyz = Dense(3, name='cls2_fc_pose_xyz')(cls2_fc1)

        cls2_fc_pose_wpqr = Dense(4, name='cls2_fc_pose_wpqr')(cls2_fc1)

        icp7_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(
            icp6_out)

        icp7_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)

        icp7_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)

        icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)

        icp7_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)

        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)

        icp8_reduction1 = Convolution2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)

        icp8_reduction2 = Convolution2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)

        icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)

        icp8_out0 = Convolution2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)

        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Convolution2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(
            icp8_out)

        icp9_out1 = Convolution2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)

        icp9_reduction2 = Convolution2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)

        icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)

        icp9_out0 = Convolution2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)

        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)


        cls3_fc1_flat_reshaped = tf.reshape(cls3_pool, [-1,1, 1024])
        bilstm1, state_h1, state_c1, state_h2, state_c2 = Bidirectional(LSTM(256,name='lstm_1',return_sequences=True, return_state=True)) (cls3_fc1_flat_reshaped)

        lstm_concat = concatenate([state_h1,state_h2],axis=1)
        lstm_concat = tf.reshape(lstm_concat, [-1, 1, 512])


        bilstm2, state_h3, state_c3, state_h4, state_c4 = Bidirectional(
            LSTM(128, name='lstm_2', return_sequences=True, return_state=True))(lstm_concat)

        lstm_concat2 = concatenate([state_h3, state_h4], axis=1)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(lstm_concat2)

        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(lstm_concat2)

        y_input_x = Input(shape=(3,), name="y_input_x")
        y_input_q = Input(shape=(4,), name="y_input_q")
        with tf.name_scope("loss_x"):
            if type=="max":
                my_loss_x,my_loss_q = max_loss()(y_input_x, cls3_fc_pose_xyz,y_input_q,cls3_fc_pose_wpqr)
            elif type=="combined":
                my_loss_x, my_loss_q = combined_loss()(y_input_x, cls3_fc_pose_xyz, y_input_q, cls3_fc_pose_wpqr)
        posenet = Model(inputs=[input, y_input_x, y_input_q],
                        outputs=[my_loss_x, my_loss_q])


    return posenet



def basic_block(channel,downsample,input):
    input = tf.pad(input, ((0, 0), (1, 1), (1, 1), (0, 0)))

    conv1 = Convolution2D(channel, (3, 3), strides=(1, 1))(input)

    norm1 = BatchNormalization(axis=3)(conv1)

    relu = ReLU()(norm1)

    relu = tf.pad(relu, ((0, 0), (1, 1), (1, 1), (0, 0)))

    conv2 = Convolution2D(channel, (3, 3), strides=(1, 1))(relu)

    norm2 = BatchNormalization(axis=3)(conv2)

    if downsample:
        conv3 = Convolution2D(channel, (1, 1), strides=(2, 2))(norm2)

        norm3 = BatchNormalization(axis=3)(conv3)

        return norm3
    return norm2

def create_hourglass_pose(bayesian=False):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/gpu:0'):
        input = Input(shape=(224, 224, 3))

        input_padded = tf.pad(input, ((0, 0), (3, 3), (3, 3), (0, 0)))

        conv0 = Convolution2D(64, (7, 7), strides=(2, 2), name='conv0')(input_padded)

        norm0 = BatchNormalization(axis=3, name='norm0')(conv0)

        relu0 = ReLU(name='relu0')(norm0)

        relu0 = tf.pad(relu0, ((0, 0), (1, 1), (1, 1), (0, 0)))

        pool0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool0')(relu0)

        x = basic_block(64, False, pool0)

        x = basic_block(64, False, x)

        x = basic_block(64, False, x)

        x = basic_block(128, True, x)

        x = basic_block(128, False, x)

        x = basic_block(128, False, x)

        x = basic_block(128, False, x)

        x = basic_block(256, True, x)

        x = basic_block(256, False, x)

        x = basic_block(256, False, x)

        x = basic_block(256, False, x)

        x = basic_block(256, False, x)

        x = basic_block(256, False, x)

        x = basic_block(512, True, x)

        x = basic_block(512, False, x)

        x = basic_block(512, False, x)

        x = tf.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))

        x = Convolution2DTranspose(256, (3, 3), strides=(2, 2))(x)

        x = tf.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))

        x = Convolution2DTranspose(128, (3, 3), strides=(2, 2))(x)

        x = tf.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))

        x = Convolution2DTranspose(64, (3, 3), strides=(2, 2))(x)

        x = tf.pad(x, ((0, 0), (3, 3), (3, 3), (0, 0)))

        x = Convolution2D(32, (3,3), strides=(1, 1))(x)

        if bayesian:
            x = Dropout(0.5)(x, training=True)

        x = Flatten()(x)

        x = Dense(1024)(x)

        pose_xyz = Dense(3, name='pose_xyz')(x)

        pose_wpqr = Dense(4, name='pose_wpqr')(x)

        hourglass_pose = Model(input, [pose_xyz, pose_wpqr])


    return hourglass_pose


def create_vgg16():


    input = Input(shape=(224, 224, 3))

    inc = tf.keras.applications.vgg16.VGG16(input_tensor=input,weights='imagenet', include_top=True)
    last = inc.layers[-3].output
    dense_layer= Dense(2048, name='dense_layer')(last)
    cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(dense_layer)
    cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(dense_layer)


    model = tf.keras.models.Model(inputs=input, outputs=[cls3_fc_pose_xyz,cls3_fc_pose_wpqr] )

    return model

def create_vgg16_geo(sx_initial,sq_initial):


    input = Input(shape=(224, 224, 3))

    inc = tf.keras.applications.vgg16.VGG16(input_tensor=input,weights='imagenet', include_top=True)
    last = inc.layers[-3].output
    dense_layer= Dense(2048, name='dense_layer')(last)
    cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(dense_layer)
    cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(dense_layer)

    y_input_x = Input(shape=(3,),name="y_input_x")
    y_input_q = Input(shape=(4,),name="y_input_q")
    with tf.name_scope("loss_x"):
        my_loss_x = Geoloss_x(sx_initial)(y_input_x, cls3_fc_pose_xyz)
    with tf.name_scope("loss_q"):
        my_loss_q = Geoloss_q(sq_initial)(y_input_q, cls3_fc_pose_wpqr)
    model = tf.keras.models.Model(inputs=[input,y_input_x,y_input_q], outputs=[my_loss_x,my_loss_q] )



    return model

def create_vgg16_max(sx_initial, sq_initial):
    input = Input(shape=(224, 224, 3))

    inc = tf.keras.applications.vgg16.VGG16(input_tensor=input, weights='imagenet', include_top=True)
    last = inc.layers[-3].output
    dense_layer = Dense(2048, name='dense_layer')(last)
    cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(dense_layer)
    cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(dense_layer)

    y_input_x = Input(shape=(3,),name="y_input_x")
    y_input_q = Input(shape=(4,),name="y_input_q")
    with tf.name_scope("loss_x"):
        if type=="max":
            my_loss_x, my_loss_q = max_loss()(y_input_x, cls3_fc_pose_xyz, y_input_q, cls3_fc_pose_wpqr)
        elif type == "combined":
            my_loss_x, my_loss_q = combined_loss()(y_input_x, cls3_fc_pose_xyz, y_input_q, cls3_fc_pose_wpqr)
    posenet = Model(inputs=[input, y_input_x, y_input_q],
                    outputs=[my_loss_x, my_loss_q])

    return posenet


if __name__ == "__main__":
    print("Please run either test.py or train.py to evaluate or fine-tune the network!")
