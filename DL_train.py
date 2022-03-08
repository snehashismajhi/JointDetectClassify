import numpy as np
import os
from keras.utils import Sequence,to_categorical
from keras.preprocessing import image
import glob
import h5py
from sklearn.preprocessing import normalize




class DataLoader_MIL_train(Sequence):

    def __init__(self,segment_size):
        self.batch_size = 30
        self.segment_size = segment_size
        self.video_path_normal = './data/Normal_Train_UCF.txt' ######### video_path_normal
        self.video_path_anomaly = './data/Abnormal_Train_UCF.txt' ############ video_path_anomaly
        self.normal_files = [i.strip()[:-4] for i in open(self.video_path_normal).readlines()]
        self.anomaly_files = [i.strip()[:-4] for i in open(self.video_path_anomaly).readlines()]
        self.path_frame_feature = "E:\\Research\\Features\\UCF\\I3D_center_crop_GAP\\"#"./data/I3D_UCF/"

    def __len__(self):
        return int(len(self.normal_files)/self.batch_size)

    def _name_to_int(self,class_name):
       if class_name == 'Abuse':
           integer= 0
       elif class_name == 'Arrest':
           integer= 1
       elif class_name == 'Arson':
           integer= 2
       elif class_name == 'Assault':
           integer= 3
       elif class_name == 'Burglary':
           integer= 4
       elif class_name == 'Explosion':
           integer= 5
       elif class_name == 'Fighting':
           integer= 6
       elif class_name == 'Normal_Videos_event' or class_name == 'Training_Normal_Videos_Anomaly' or class_name == 'Testing_Normal_Videos_Anomaly':
           integer= 7
       elif class_name == 'RoadAccidents':
           integer= 8
       elif class_name == 'Robbery':
           integer= 9
       elif class_name == 'Shooting':
           integer= 10
       elif class_name == 'Shoplifting':
           integer= 11
       elif class_name == 'Stealing':
           integer= 12
       elif class_name == 'Vandalism':
           integer= 13
       return integer

    def _get_class_label(self,video_name):
        folder = video_name.split('/')
        class_name = folder[0]
        int_label = self._name_to_int(class_name)
        return int_label


    def __getitem__(self, item):
        batch_normal = self.normal_files[item * self.batch_size : (item+1) * self.batch_size]
        batch_anomaly = self.anomaly_files[item * self.batch_size : (item + 1) * self.batch_size]

        RGB_train_normal= [self._get_feature(i) for i in batch_normal]
        RGB_attn_normal = [self._get_feature_norm(i) for i in batch_normal]
        y_train_normal = np.ones((self.batch_size*32,1))
        label_train_normal = [self._get_class_label(i) for i in batch_normal]

        RGB_train_anomaly = [self._get_feature(i) for i in batch_anomaly]
        RGB_attn_anomaly = [self._get_feature_norm(i) for i in batch_anomaly]
        y_train_anomaly = np.zeros((self.batch_size*32,1))
        label_train_anomaly = [self._get_class_label(i) for i in batch_anomaly]

        RGB_train_normal = np.asarray(RGB_train_normal)
        RGB_attn_normal = np.asarray(RGB_attn_normal)
        y_train_normal = np.array(y_train_normal)
        label_train_normal = np.array(label_train_normal)

        RGB_train_anomaly = np.asarray(RGB_train_anomaly)
        RGB_attn_anomaly = np.asarray(RGB_attn_anomaly)
        y_train_anomaly = np.array(y_train_anomaly)
        label_train_anomaly = np.array(label_train_anomaly)

        RGB_train = np.vstack([RGB_train_anomaly,RGB_train_normal])
        RGB_attn = np.vstack([RGB_attn_anomaly,RGB_attn_normal])
        y_train = np.concatenate([y_train_anomaly,y_train_normal])
        y_train_classification = np.concatenate([label_train_anomaly,label_train_normal])
        y_train_classification = to_categorical(y_train_classification,num_classes=14)
        y_train = np.reshape(y_train,(self.batch_size*2,32,1))

        return [RGB_train,RGB_attn], [y_train,y_train_classification]



    def _get_feature(self,video_name):
        f = h5py.File(self.path_frame_feature + video_name+'.h5', 'r')
        feature = np.array(f['features'])
        feature = np.squeeze(feature,axis=0)
        feature = np.squeeze(feature,axis=3)
        feature = np.squeeze(feature,axis=2)
        feature_norm = np.reshape(feature,(feature.shape[0],feature.shape[1]*feature.shape[2]))

        if feature_norm.shape[0] < self.segment_size:
            feature_segment = feature_norm
            i = feature_norm.shape[0]
            k = feature_norm.shape[0] - 1
            while i < self.segment_size:
                feature_segment = np.vstack([feature_segment, feature_norm[k]])
                i = i+1
        else:
            feature_segment = []
            for i in range(self.segment_size):
                j = int(feature_norm.shape[0]/self.segment_size)
                temp = feature_norm[i*j:(i+1)*j]
                feature_segment.append(np.max(temp,axis=0))
            feature_segment = np.asarray(feature_segment)
        return feature_segment


    def _get_feature_norm(self,video_name):
        f = h5py.File(self.path_frame_feature + video_name+'.h5', 'r')
        feature = np.array(f['features'])
        feature = np.squeeze(feature,axis=0)
        feature = np.squeeze(feature,axis=3)
        feature = np.squeeze(feature,axis=2)
        feature = np.reshape(feature,(feature.shape[0],feature.shape[1]*feature.shape[2]))
        feature_norm = normalize(feature,norm='l2')

        if feature_norm.shape[0] < self.segment_size:
            feature_segment = feature_norm
            i = feature_norm.shape[0]
            k = feature_norm.shape[0] - 1
            while i < self.segment_size:
                feature_segment = np.vstack([feature_segment, feature_norm[k]])
                i = i+1
        else:
            feature_segment = []
            for i in range(self.segment_size):
                j = int(feature_norm.shape[0]/self.segment_size)
                temp = feature_norm[i*j:(i+1)*j]
                feature_segment.append(np.max(temp,axis=0))
            feature_segment = np.asarray(feature_segment)
        return feature_segment
