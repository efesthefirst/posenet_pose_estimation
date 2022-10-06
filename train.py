import helper
import posenet
import os
import numpy as np
import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
from  tensorflow.keras.utils import plot_model
import sys
import datetime
import matplotlib.pyplot as plt
import time
from posenet import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == "__main__":
    # Variables
    batch_size = 1
    epoch_num= 1000
    learning_rate=1e-5
    load_weights="random" # random, load_h5 or npy(for posenet only)
    model_arc= "posenet" # "posenet", "poselstm", "poseincepv3", "poselstm_with_2_lstm" "vidloc" "hourglass_pose" "vgg16"
    loss_type="euc" # "euc" "geo" "max" "combined" "msl" (mean squared logarithmics) "huber"  or "logcosh"
    bayesian=False

    sx_initial = 0
    sq_initial = -3

    msl_coeff=300

    huber_coef=1e3
    huber_delta_x=50
    huber_delta_q=50

    cosh_coef=1500

    # Train model
    if model_arc == "posenet":

        if loss_type=="euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            if load_weights == "npy":
                model = posenet.create_posenet(bayesian,'posenet.npy', True)
            elif load_weights == "load_h5":
                model = posenet.create_posenet(bayesian)
                model.load_weights('custom_trained_weights.h5')
            elif load_weights == "random":
                model = posenet.create_posenet(bayesian)

        elif loss_type == "geo":
            if load_weights == "random":
                model = posenet.create_posenet_geo(sx_initial,sq_initial)
            elif load_weights == "load_h5":
                model = posenet.create_posenet_geo (sx_initial,sq_initial)
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type == "max":
            if load_weights == "random":
                model = posenet.create_posenet_max(type="max")
            elif load_weights == "load_h5":
                model = posenet.create_posenet_max (type="max")
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type == "combined":
            if load_weights == "random":
                model = posenet.create_posenet_max(type="combined")
            elif load_weights == "load_h5":
                model = posenet.create_posenet_max (type="combined")
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

    elif model_arc == "poselstm":
        if loss_type=="euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            if load_weights == "random":
                model = posenet.create_poselstm()
            elif load_weights == "load_h5":
                model = posenet.create_poselstm()
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type=="geo":
            if load_weights == "random":
                model = posenet.create_poselstm_with_geo_loss(sx_initial,sq_initial)
            elif load_weights == "load_h5":
                model = posenet.create_poselstm_with_geo_loss(sx_initial,sq_initial)
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type == "max":
            if load_weights == "random":
                model = posenet.create_poselstm_with_max_loss(type="max")
            elif load_weights == "load_h5":
                model = posenet.create_poselstm_with_max_loss(type="max")
                model.load_weights('custom_trained_weights.h5', by_name=True, skip_mismatch=True)

        elif loss_type == "combined":
            if load_weights == "random":
                model = posenet.create_poselstm_with_max_loss(type="combined")
            elif load_weights == "load_h5":
                model = posenet.create_poselstm_with_max_loss(type="combined")
                model.load_weights('custom_trained_weights.h5', by_name=True, skip_mismatch=True)

    elif model_arc == "poseincepv3":
        if loss_type == "euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            if load_weights == "random":
                model = posenet.create_inception_v3()
            elif load_weights == "load_h5":
                model = posenet.create_inception_v3 ()
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type == "geo":
            if load_weights == "random":
                model = posenet.create_inception_v3_with_geo_loss(sx_initial,sq_initial)
            elif load_weights == "load_h5":
                model = posenet.create_inception_v3_with_geo_loss (sx_initial,sq_initial)
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type == "max":
            if load_weights == "random":
                model = posenet.create_inception_v3_with_max_loss(type="max")
            elif load_weights == "load_h5":
                model = posenet.create_inception_v3_with_max_loss(type="max")
                model.load_weights('custom_trained_weights.h5', by_name=True, skip_mismatch=True)

        elif loss_type == "combined":
            if load_weights == "random":
                model = posenet.create_inception_v3_with_max_loss(type="combined")
            elif load_weights == "load_h5":
                model = posenet.create_inception_v3_with_max_loss(type="combined")
                model.load_weights('custom_trained_weights.h5', by_name=True, skip_mismatch=True)

    elif model_arc == "vgg16":
        if loss_type == "euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            if load_weights == "random":
                model = posenet.create_vgg16()
            elif load_weights == "load_h5":
                model = posenet.create_vgg16 ()
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type == "geo":
            if load_weights == "random":
                model = posenet.create_vgg16_geo(sx_initial,sq_initial)
            elif load_weights == "load_h5":
                model = posenet.create_vgg16_geo (sx_initial,sq_initial)
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type == "max":
            if load_weights == "random":
                model = posenet.create_vgg16_max(type="max")
            elif load_weights == "load_h5":
                model = posenet.create_vgg16_max(type="max")
                model.load_weights('custom_trained_weights.h5', by_name=True, skip_mismatch=True)

        elif loss_type == "combined":
            if load_weights == "random":
                model = posenet.create_vgg16_max(type="combined")
            elif load_weights == "load_h5":
                model = posenet.create_vgg16_max(type="combined")
                model.load_weights('custom_trained_weights.h5', by_name=True, skip_mismatch=True)

    elif model_arc == "poselstm_with_2_lstm":
        if loss_type == "euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            if load_weights == "random":
                model = posenet.create_poselstm_with_2_lstm()
            elif load_weights == "load_h5":
                model = posenet.create_poselstm_with_2_lstm ()
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type == "geo":
            if load_weights == "random":
                model = posenet.create_poselstm_with_2_lstm_with_geo_loss(sx_initial,sq_initial)
            elif load_weights == "load_h5":
                model = posenet.create_poselstm_with_2_lstm_with_geo_loss (sx_initial,sq_initial)
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type == "max":
            if load_weights == "random":
                model = posenet.create_poselstm_with_2_lstm_with_max_loss(type="max")
            elif load_weights == "load_h5":
                model = posenet.create_poselstm_with_2_lstm_with_max_loss(type="max")
                model.load_weights('custom_trained_weights.h5', by_name=True, skip_mismatch=True)
        elif loss_type == "combined":
            if load_weights == "random":
                model = posenet.create_poselstm_with_2_lstm_with_max_loss(type="combined")
            elif load_weights == "load_h5":
                model = posenet.create_poselstm_with_2_lstm_with_max_loss(type="combined")
                model.load_weights('custom_trained_weights.h5', by_name=True, skip_mismatch=True)

    elif model_arc == "vidloc":
        if loss_type == "euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            if load_weights == "random":
                model = posenet.create_vidloc()
            elif load_weights == "load_h5":
                model = posenet.create_vidloc ()
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type == "geo":
            if load_weights == "random":
                model = posenet.create_vidloc_geo(sx_initial,sq_initial)
            elif load_weights == "load_h5":
                model = posenet.create_vidloc_geo (sx_initial,sq_initial)
                model.load_weights('custom_trained_weights.h5',by_name=True,skip_mismatch=True)

        elif loss_type == "max":
            if load_weights == "random":
                model = posenet.create_vidloc_max(type="max")
            elif load_weights == "load_h5":
                model = posenet.create_vidloc_max(type="max")
                model.load_weights('custom_trained_weights.h5', by_name=True, skip_mismatch=True)
        elif loss_type == "combined":
            if load_weights == "random":
                model = posenet.create_vidloc_max(type="combined")
            elif load_weights == "load_h5":
                model = posenet.create_vidloc_max(type="combined")
                model.load_weights('custom_trained_weights.h5', by_name=True, skip_mismatch=True)

    elif model_arc == "hourglass_pose":
        if loss_type == "euc":
            model = posenet.create_hourglass_pose(bayesian)
        else:
            raise ValueError('The only valid loss function for hourglass_pose model is euclidean !')




    adam = Adam(learning_rate=learning_rate, clipvalue=1.5)

    # model plots and summary
    #model.summary()
    if model_arc=="posenet":
        plot_model(model, to_file="my_model_posenet.png", show_shapes=True)
    elif model_arc == "poselstm":
        plot_model(model, to_file="my_model_lstm.png", show_shapes=True)
    elif model_arc == "poseincepv3":
        plot_model(model, to_file="my_model_incepv3.png", show_shapes=True)
    elif model_arc == "vgg16":
        plot_model(model, to_file="my_model_vgg16.png", show_shapes=True)
    elif model_arc == "poselstm_with_2_lstm":
        plot_model(model, to_file="my_model_poselstm_with_2_lstm.png", show_shapes=True)
    elif model_arc == "vidloc":
        plot_model(model, to_file="my_model_vidloc.png", show_shapes=True)
    elif model_arc == "hourglass_pose":
        plot_model(model, to_file="my_model_hourglass_pose.png", show_shapes=True)

    #Model.compile(loss_weights=None)
    # We can add weights to our losses

    dataset_train, dataset_test = helper.getKings()

    X_train = np.squeeze(np.array(dataset_train.images))
    np.save('x_train.npy',X_train)
    y_train = np.squeeze(np.array(dataset_train.poses))


    #print(y_train.shape)
    #In order to check if the input image number is correct or not

    y_train_x = y_train[:,0:3]
    np.save('y_train_x.npy', y_train_x)
    y_train_q = y_train[:,3:7]
    np.save('y_train_q.npy', y_train_q)


    #X_test = np.squeeze(np.array(dataset_test.images))
    #y_test = np.squeeze(np.array(dataset_test.poses))

    #y_test_x = y_test[:,0:3]
    #y_test_q = y_test[:,3:7]

    #sys.exit("Error message")

    # Setup checkpointing
    checkpointer = ModelCheckpoint(filepath="checkpoint_weights.h5", verbose=0, save_best_only=True, save_weights_only=False)

    # We can also monitor our losses on tensorboard. For more details visit https://www.tensorflow.org/tensorboard/graphs

    logdir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    # We can add an early stopping criteria
    early_stopping_callback=EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=100,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=False
    )

    # A callback for printing total training time

    class TRA(tensorflow.keras.callbacks.Callback):
        def __init__(self):
            super(TRA, self).__init__()

        def on_train_begin(self, logs=None):
            self.start_time = time.time()

        def on_train_end(self, logs=None):
            stop_time = time.time()
            tr_duration = stop_time - self.start_time
            hours = tr_duration // 3600
            minutes = (tr_duration - (hours * 3600)) // 60
            seconds = tr_duration - ((hours * 3600) + (minutes * 60))
            msg = f'Training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
            print(msg)

    if model_arc=="posenet":

        if loss_type == "euc":
            model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                            'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                                    'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})
        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":
            model.compile(optimizer=adam)

        elif loss_type == "msl":
            model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': tf.keras.losses.MeanSquaredLogarithmicError(), 'cls1_fc_pose_wpqr': tf.keras.losses.MeanSquaredLogarithmicError(),
                                            'cls2_fc_pose_xyz': tf.keras.losses.MeanSquaredLogarithmicError(), 'cls2_fc_pose_wpqr': tf.keras.losses.MeanSquaredLogarithmicError(),
                                            'cls3_fc_pose_xyz': tf.keras.losses.MeanSquaredLogarithmicError(), 'cls3_fc_pose_wpqr': tf.keras.losses.MeanSquaredLogarithmicError()},
                          loss_weights=[1,msl_coeff,1,msl_coeff,1,msl_coeff])
        elif loss_type == "huber":
            model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': tf.keras.losses.Huber(delta=huber_delta_x), 'cls1_fc_pose_wpqr': tf.keras.losses.Huber(delta=huber_delta_q),
                                            'cls2_fc_pose_xyz': tf.keras.losses.Huber(delta=huber_delta_x), 'cls2_fc_pose_wpqr': tf.keras.losses.Huber(delta=huber_delta_q),
                                            'cls3_fc_pose_xyz': tf.keras.losses.Huber(delta=huber_delta_x), 'cls3_fc_pose_wpqr': tf.keras.losses.Huber(delta=huber_delta_q)},
                          loss_weights=[1e-2,huber_coef,1e-2,huber_coef,1e-2,huber_coef])
        elif loss_type == "logcosh":
            model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': tf.keras.losses.LogCosh(), 'cls1_fc_pose_wpqr': tf.keras.losses.LogCosh(),
                                            'cls2_fc_pose_xyz': tf.keras.losses.LogCosh(), 'cls2_fc_pose_wpqr': tf.keras.losses.LogCosh(),
                                            'cls3_fc_pose_xyz': tf.keras.losses.LogCosh(), 'cls3_fc_pose_wpqr': tf.keras.losses.LogCosh()},
                          loss_weights=[1,cosh_coef,1,cosh_coef,1,cosh_coef])

    elif model_arc=="vidloc":

        if loss_type == "euc":
            model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                            'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                                    'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})
        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":
            model.compile(optimizer=adam)

        elif loss_type == "msl":
            model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': tf.keras.losses.MeanSquaredLogarithmicError(), 'cls1_fc_pose_wpqr': tf.keras.losses.MeanSquaredLogarithmicError(),
                                            'cls2_fc_pose_xyz': tf.keras.losses.MeanSquaredLogarithmicError(), 'cls2_fc_pose_wpqr': tf.keras.losses.MeanSquaredLogarithmicError(),
                                            'cls3_fc_pose_xyz': tf.keras.losses.MeanSquaredLogarithmicError(), 'cls3_fc_pose_wpqr': tf.keras.losses.MeanSquaredLogarithmicError()},
                          loss_weights=[1,msl_coeff,1,msl_coeff,1,msl_coeff])
        elif loss_type == "huber":
            model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': tf.keras.losses.Huber(delta=huber_delta_x), 'cls1_fc_pose_wpqr': tf.keras.losses.Huber(delta=huber_delta_q),
                                            'cls2_fc_pose_xyz': tf.keras.losses.Huber(delta=huber_delta_x), 'cls2_fc_pose_wpqr': tf.keras.losses.Huber(delta=huber_delta_q),
                                            'cls3_fc_pose_xyz': tf.keras.losses.Huber(delta=huber_delta_x), 'cls3_fc_pose_wpqr': tf.keras.losses.Huber(delta=huber_delta_q)},
                          loss_weights=[1e-2,huber_coef,1e-2,huber_coef,1e-2,huber_coef])
        elif loss_type == "logcosh":
            model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': tf.keras.losses.LogCosh(), 'cls1_fc_pose_wpqr': tf.keras.losses.LogCosh(),
                                            'cls2_fc_pose_xyz': tf.keras.losses.LogCosh(), 'cls2_fc_pose_wpqr': tf.keras.losses.LogCosh(),
                                            'cls3_fc_pose_xyz': tf.keras.losses.LogCosh(), 'cls3_fc_pose_wpqr': tf.keras.losses.LogCosh()},
                          loss_weights=[1,cosh_coef,1,cosh_coef,1,cosh_coef])

    elif model_arc=="poselstm":

        if  loss_type == "euc":
            model.compile(optimizer=adam,
                      loss={'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})


        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":
            model.compile(optimizer=adam)

        elif  loss_type == "msl":
            model.compile(optimizer=adam,
                      loss={'cls3_fc_pose_xyz': tf.keras.losses.MeanSquaredLogarithmicError(), 'cls3_fc_pose_wpqr': tf.keras.losses.MeanSquaredLogarithmicError()},
                          loss_weights=[1,msl_coeff])
        elif  loss_type == "huber":
            model.compile(optimizer=adam,
                      loss={'cls3_fc_pose_xyz': tf.keras.losses.Huber(delta=huber_delta_x), 'cls3_fc_pose_wpqr': tf.keras.losses.Huber(delta=huber_delta_q)},
                          loss_weights=[1e-2,huber_coef])
        elif  loss_type == "logcosh":
            model.compile(optimizer=adam,
                      loss={'cls3_fc_pose_xyz': tf.keras.losses.LogCosh(), 'cls3_fc_pose_wpqr': tf.keras.losses.LogCosh()},
                          loss_weights=[1,cosh_coef])

    elif model_arc=="poselstm_with_2_lstm":
        if loss_type == "euc":
            model.compile(optimizer=adam,
                          loss={'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})
        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":
            model.compile(optimizer=adam)

        elif loss_type == "msl":
            model.compile(optimizer=adam,
                          loss={'cls3_fc_pose_xyz': tf.keras.losses.MeanSquaredLogarithmicError(),
                                'cls3_fc_pose_wpqr': tf.keras.losses.MeanSquaredLogarithmicError()},loss_weights=[1,msl_coeff])
        elif  loss_type == "huber":
            model.compile(optimizer=adam,
                      loss={'cls3_fc_pose_xyz': tf.keras.losses.Huber(delta=huber_delta_x), 'cls3_fc_pose_wpqr': tf.keras.losses.Huber(delta=huber_delta_q)},
                          loss_weights=[1e-2,huber_coef])

        elif  loss_type == "logcosh":
            model.compile(optimizer=adam,
                      loss={'cls3_fc_pose_xyz': tf.keras.losses.LogCosh(), 'cls3_fc_pose_wpqr': tf.keras.losses.LogCosh()},
                          loss_weights=[1,cosh_coef])

    elif model_arc=="poseincepv3":
        if loss_type == "euc":
            model.compile(optimizer=adam,
                          loss={'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})
        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":
            model.compile(optimizer=adam)

        elif loss_type == "msl":
            model.compile(optimizer=adam,
                          loss={'cls3_fc_pose_xyz': tf.keras.losses.MeanSquaredLogarithmicError(),
                                'cls3_fc_pose_wpqr': tf.keras.losses.MeanSquaredLogarithmicError()},loss_weights=[1,msl_coeff])
        elif  loss_type == "huber":
            model.compile(optimizer=adam,
                      loss={'cls3_fc_pose_xyz': tf.keras.losses.Huber(delta=huber_delta_x), 'cls3_fc_pose_wpqr': tf.keras.losses.Huber(delta=huber_delta_q)},
                          loss_weights=[1e-2,huber_coef])
        elif  loss_type == "logcosh":
            model.compile(optimizer=adam,
                      loss={'cls3_fc_pose_xyz': tf.keras.losses.LogCosh(), 'cls3_fc_pose_wpqr': tf.keras.losses.LogCosh()},
                          loss_weights=[1,cosh_coef])

    elif model_arc == "hourglass_pose":
        if loss_type == "euc":
            model.compile(optimizer=adam,
                          loss={'pose_xyz': posenet.euc_loss3x, 'pose_wpqr': posenet.euc_loss3q})

    elif model_arc=="vgg16":
        if loss_type == "euc":
            model.compile(optimizer=adam,
                          loss={'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})
        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":
            model.compile(optimizer=adam)

        elif loss_type == "msl":
            model.compile(optimizer=adam,
                          loss={'cls3_fc_pose_xyz': tf.keras.losses.MeanSquaredLogarithmicError(),
                                'cls3_fc_pose_wpqr': tf.keras.losses.MeanSquaredLogarithmicError()},loss_weights=[1,msl_coeff])
        elif  loss_type == "huber":
            model.compile(optimizer=adam,
                      loss={'cls3_fc_pose_xyz': tf.keras.losses.Huber(delta=huber_delta_x), 'cls3_fc_pose_wpqr': tf.keras.losses.Huber(delta=huber_delta_q)},
                          loss_weights=[1e-2,huber_coef])
        elif  loss_type == "logcosh":
            model.compile(optimizer=adam,
                      loss={'cls3_fc_pose_xyz': tf.keras.losses.LogCosh(), 'cls3_fc_pose_wpqr': tf.keras.losses.LogCosh()},
                          loss_weights=[1,cosh_coef])



    if model_arc=="posenet":
        if loss_type == "euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            model.fit(X_train, [y_train_x, y_train_q, y_train_x, y_train_q, y_train_x, y_train_q],
                  batch_size=batch_size,
                  epochs=epoch_num,
                  validation_data=(X_train, [y_train_x, y_train_q, y_train_x, y_train_q, y_train_x, y_train_q]),
                  callbacks=[checkpointer,tensorboard_callback,early_stopping_callback,TRA()])

        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":
            model.fit([X_train, y_train_x, y_train_q], None,
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=([X_test,y_test_x, y_test_q],None),
                      callbacks=[checkpointer,tensorboard_callback, early_stopping_callback, TRA()])

    elif model_arc == "vidloc":
        if loss_type == "euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            model.fit(X_train, [y_train_x, y_train_q],
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=(X_test, [y_test_x, y_test_q]),
                      callbacks=[checkpointer,tensorboard_callback,early_stopping_callback,TRA()])


        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":
            model.fit([X_train, y_train_x, y_train_q], None,
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=([X_test,y_test_x, y_test_q],None),
                      callbacks=[checkpointer,tensorboard_callback, early_stopping_callback, TRA()])

    elif model_arc == "poselstm":
        if loss_type == "euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            model.fit(X_train, [y_train_x, y_train_q],
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=(X_test, [y_test_x, y_test_q]),
                      callbacks=[checkpointer,tensorboard_callback,early_stopping_callback,TRA()])


        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":
            model.fit([X_train, y_train_x, y_train_q], None,
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=([X_test,y_test_x, y_test_q],None),
                      callbacks=[checkpointer,tensorboard_callback, early_stopping_callback, TRA()])

    elif model_arc == "poseincepv3":
        if loss_type == "euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            model.fit(X_train, [y_train_x, y_train_q],
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=(X_test, [y_test_x, y_test_q]),
                      callbacks=[checkpointer,tensorboard_callback,early_stopping_callback,TRA()])

        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":

            model.fit([X_train, y_train_x, y_train_q], None,
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=([X_test,y_test_x, y_test_q],None),
                      callbacks=[checkpointer,tensorboard_callback, early_stopping_callback, TRA()])

    elif model_arc == "vgg16":
        if loss_type == "euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            model.fit(X_train, [y_train_x, y_train_q],
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=(X_test, [y_test_x, y_test_q]),
                      callbacks=[checkpointer,tensorboard_callback,early_stopping_callback,TRA()])

        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":

            model.fit([X_train, y_train_x, y_train_q], None,
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=([X_test,y_test_x, y_test_q],None),
                      callbacks=[checkpointer,tensorboard_callback, early_stopping_callback, TRA()])

    elif model_arc == "poselstm_with_2_lstm":
        if loss_type == "euc" or loss_type == "msl" or loss_type == "huber" or loss_type == "logcosh":
            model.fit(X_train, [y_train_x, y_train_q],
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=(X_test, [y_test_x, y_test_q]),
                      callbacks=[checkpointer,tensorboard_callback,early_stopping_callback,TRA()])

        elif loss_type == "geo" or loss_type == "max" or loss_type=="combined":
            model.fit([X_train, y_train_x, y_train_q], None,
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=([X_test,y_test_x, y_test_q],None),
                      callbacks=[checkpointer,tensorboard_callback, early_stopping_callback, TRA()])

    elif model_arc == "hourglass_pose":
        if loss_type == "euc":
            model.fit(X_train, [y_train_x, y_train_q],
                      batch_size=batch_size,
                      epochs=epoch_num,
                      validation_data=(X_test, [y_test_x, y_test_q]),
                      callbacks=[checkpointer,tensorboard_callback,early_stopping_callback,TRA()])



    model.save_weights("custom_trained_weights.h5")
