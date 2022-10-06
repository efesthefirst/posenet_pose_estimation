import math
import helper
import posenet
import numpy as np
from tensorflow.keras.optimizers import Adam
from scipy.spatial.transform import Rotation

directory = 'D:\\keras-posenet-master\\datasets\\rgb_real\\'
dataset_train_direct = 'dataset_train.txt'
dataset_test_direct = 'dataset_test.txt'
model_arc="posenet"

if model_arc=="posenet":
    f1 = open('D:\\keras-posenet-master\\posenet_angle.txt', 'w')
    f2 = open('D:\\keras-posenet-master\\posenet_coord.txt', 'w')
elif model_arc=="lstm-pose":
    f1 = open('D:\\keras-posenet-master\\lstmpose_angle.txt', 'w')
    f2 = open('D:\\keras-posenet-master\\lstmpose_coord.txt', 'w')
elif model_arc=="poseincepv3":
    f1 = open('D:\\keras-posenet-master\\poseincepv3_angle.txt', 'w')
    f2 = open('D:\\keras-posenet-master\\poseincepv3_coord.txt', 'w')
elif model_arc=="poselstm_with_2_lstm" :
    f1 = open('D:\\keras-posenet-master\\poselstm_with_2_lstm_angle.txt', 'w')
    f2 = open('D:\\keras-posenet-master\\poselstm_with_2_lstm_coord.txt', 'w')

if __name__ == "__main__":
    # Test model



    model = posenet.create_posenet(False)
    model.load_weights('custom_trained_weights.h5')
    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=2.0)
    model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                        'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                        'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})

    dataset_train, dataset_test = helper.getKings()

    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.poses))

    testPredict = model.predict(X_test)

    valsx = testPredict[4]
    valsq = testPredict[5]

    train_image_names = []

    with open(directory + dataset_train_direct) as f:
        next(f)  # skip the 3 header lines
        next(f)
        next(f)
        for line in f:
            fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
            train_image_names.append(fname)


    def angle_diff_abs(x, y):
        x = np.array(x)
        y = np.array(y)
        error = 180 - abs(abs(x - y) - 180);

        return (error)


    # Get results... :/
    results = np.zeros((len(dataset_test.images),6))
    for i in range(len(dataset_test.images)):

        pose_q= np.asarray(dataset_test.poses[i][3:7])
        pose_x= np.asarray(dataset_test.poses[i][0:3])
        predicted_x = valsx[i]
        predicted_q = valsq[i]

        pose_q = np.squeeze(pose_q)
        pose_x = np.squeeze(pose_x)
        predicted_q = np.squeeze(predicted_q)
        predicted_x = np.squeeze(predicted_x)
        print(predicted_x)
        #Compute Individual Sample Error
        q1 = pose_q / np.linalg.norm(pose_q)
        q2 = predicted_q / np.linalg.norm(predicted_q)
        d = abs(np.sum(np.multiply(q1,q2)))
        theta = 2 * np.arccos(d) * 180/math.pi

        error_distance = np.linalg.norm(pose_x-predicted_x)



        predicted_q = np.roll(predicted_q, -1)
        pose_q = np.roll(pose_q, -1)


        rot_pred = Rotation.from_quat(predicted_q)
        rot_real = Rotation.from_quat(pose_q)
        rotation_pred = rot_pred.as_euler('zyx', degrees=True)
        rotation_real = rot_real.as_euler('zyx', degrees=True)

        results[i, :] = [ abs((pose_x-predicted_x)[0]), abs((pose_x-predicted_x)[1]),abs((pose_x-predicted_x)[2]),angle_diff_abs(rotation_pred[0], rotation_real[0]),angle_diff_abs(rotation_pred[1],
                            rotation_real[1]),angle_diff_abs(rotation_pred[2], rotation_real[2]),error_distance,theta]




        f2.write(train_image_names[i] + ' ' + str(predicted_x[0]) + ' ' + str(predicted_x[1]) + ' ' + str(predicted_x[2]) + '\n')
        f1.write(train_image_names[i] + ' ' + str(rotation_pred[0]) + ' ' + str(rotation_pred[1]) + ' ' + str(rotation_pred[2]) + '\n')

        # print('Image Name:  ', train_image_names[i], '  Error X (m):  ', abs((pose_x-predicted_x)[0]),'  Error Y (m):  ', abs((pose_x-predicted_x)[1]),'  Error Z (m):  ', abs((pose_x-predicted_x)[2]),
        #           '  Error Yaw (degrees):  ', angle_diff_abs(rotation_pred[0], rotation_real[0]),'  Error Pitch (degrees):  '
        #           , angle_diff_abs(rotation_pred[1], rotation_real[1]),'  Error Roll (degrees):  ', angle_diff_abs(rotation_pred[2], rotation_real[2]) )

    median_result = np.median(results,axis=0)
    print('Median error X ', median_result[0], 'm  and ', 'Median error Y ', median_result[1], 'm  and ','Median error Z ', median_result[2],
          'm  and Yaw error ',median_result[3], 'degrees and Pitch error', median_result[4], 'degrees and Roll error', median_result[5], 'degrees.',"angular error",median_result[7]
          ,"positional error",median_result[6])

    f1.close()
    f2.close()

f1 = open('D:\\keras-posenet-master\\posenet_angle.txt', 'r')
f3 = open('D:\\keras-posenet-master\\posenet_angle_sorted.txt', 'w')
lines = f1.readlines()
lines_sorted = sorted(lines)
for i in range(len(lines)):
    f3.write(lines_sorted[i])
f3.close()

f2 = open('D:\\keras-posenet-master\\posenet_coord.txt', 'r')
f4 = open('D:\\keras-posenet-master\\posenet_coord_sorted.txt', 'w')
lines = f2.readlines()
lines_sorted = sorted(lines)
for i in range(len(lines)):
    f4.write(lines_sorted[i])
f4.close()