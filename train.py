import os
import sys
import numpy as np
from keras.optimizers import SGD,Adam, Adagrad
from sklearn.metrics import roc_auc_score, roc_curve,auc
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
from ObjectiveFunctions import *
from DL_train import *
from DL_test import *
from model import *

seed = 7
np.random.seed(seed)



lrn = 0.0001
l3 = 0.001
nuron = 96
l1 = 0.9
l2 = 1-l1
segment_size = 32

############################################# CUSTOME CALLBACK ################################
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.ALL_AUC = []
        self.ALL_EPOCH = []
        self.ALL_acc1 = []
        self.ALL_acc2 = []
        self.ALL_acc3 = []
        self.ALL_acc4 = []
        self.ALL_Avg_ACC = []
    
    
    def get_GT(self,test_file):
        test_file=test_file.split()
        video_name=test_file[0][:-4]
        s1 = int(test_file[2])
        e1 = int(test_file[3])
        s2 = int(test_file[4])
        e2 = int(test_file[5])
        Test_video_name = np.load("./data/Test_video_name_UCF.npy")
        Test_frame_number = np.load("./data/Test_frame_number_UCF.npy")
        video_index = int(np.argwhere(Test_video_name==video_name))
        gt = np.zeros((Test_frame_number[video_index], 1))  # Initially all normal     #NORMAL = 1  # ABNORMAL =0
        if s1 != -1 and e1 != -1:
            gt[s1:e1, 0] = 1
        if s2 != -1 and e2 != -1:
            gt[s2:e2, 0] = 1
        return gt


    def my_detection_metrics(self):
        test_gen = DataLoader_test_detect(segment_size)
        score = model.predict_generator(test_gen)

        ############ DETECTION PART ################
        detection_score=np.asarray(score[0])
        temp_annotation = './data/Temporal_Anomaly_Annotation_for_Testing_Videos_UCF.txt'
        test_files = [i.strip() for i in open(temp_annotation).readlines()]
        ALL_GT = np.array([])
        ALL_score = np.array([])
        for i in range(len(test_files)):
            video_file = test_files[i]
            video_file = video_file.split()
            video_name = video_file[0][:-4]
            video_segment_score = detection_score[i]
            video_GT = self.get_GT(test_files[i])
            video_score = np.array([])
            for k in range(segment_size):
                dummy_score = np.repeat(video_segment_score[k, 0], np.floor(video_GT.shape[0] / segment_size))
                video_score = np.concatenate([video_score, dummy_score])
            if video_GT.shape[0] % segment_size != 0:
                dummy_remain_score = np.repeat(video_segment_score[segment_size-1, 0], video_GT.shape[0] - np.floor(video_GT.shape[0] / segment_size) * segment_size)
                video_score = np.concatenate([video_score, dummy_remain_score])
            video_GT = np.squeeze(video_GT, axis=1)
            ALL_GT = np.concatenate([ALL_GT, video_GT])
            ALL_score = np.concatenate([ALL_score, video_score])
        AUC = roc_auc_score(ALL_GT, ALL_score)
        ############## RECOGNITION PART ##############
        recognition_class=np.argmax(np.asarray(score[1]),axis=-1)
        recognition_GT =np.array([])
        for i in range(29):
            recognition_GT =np.concatenate([recognition_GT,np.asarray(test_gen[i][2])])
        cm =confusion_matrix(recognition_GT,recognition_class)
        cmn = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]
        mean_avg_acc=np.sum(cmn.diagonal())/14

        return AUC, mean_avg_acc

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % 1 == 0:
            AUC , acc = self.my_detection_metrics()
            print("###############  AUC : " + str(AUC))
            print("TEST RECOGNITION mAA on Detection Set: " + str(acc))
            self.ALL_EPOCH.append(epoch)
            self.ALL_AUC.append(AUC)
            self.ALL_Avg_ACC.append(acc)
            AUC_new = np.asarray(self.ALL_AUC)
            EPOCH_new = np.asarray(self.ALL_EPOCH)
            acc_avg_new = np.asarray(self.ALL_Avg_ACC)
            total = np.asarray(list(zip(EPOCH_new, AUC_new, acc_avg_new)))
            
            
            np.savetxt("./AUC/AUC_JointD&C.txt", total, delimiter=',')
            # prev_AUC = np.max(np.array(AUC_new))
            # if AUC >= prev_AUC :
            #     model.save('./model/MIL_model_JointD&C.h5')
            #     print('MIL model weights saved Sucessfully')
            prev_acc = np.max(np.array(acc_avg_new))
            if acc >= prev_acc:
                model.save('./Model/Model_JointD&C.h5')
                print('Recognition model weights saved Sucessfully')

        return
############################################ DEFINE MODEL ###################################

weights = [0.97018634, 0.97204969, 0.97453416, 0.97080745, 0.94596273,0.98198758, 0.97204969, 0.50310559, 0.92111801, 0.90993789,0.98322981, 0.98198758, 0.94099379, 0.97204969]
custom_cce = weighted_categorical_crossentropy(weights)
losses = [RankingLoss,custom_cce]
adam = Adam(lr = lrn, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0)
model = JointDandC_model(l3,nuron)
model.summary()
model.compile(loss = losses, loss_weights = [l1, l2], optimizer = adam)
metrics = Metrics()
print("Starting training...")

########################################## TRAINING #####################################################
num_epoch = 70000
stp_epc = 1
train_generator = DataLoader_MIL_train(segment_size)
loss = model.fit_generator(train_generator, steps_per_epoch=stp_epc, epochs=num_epoch, verbose=1,callbacks=[metrics])

