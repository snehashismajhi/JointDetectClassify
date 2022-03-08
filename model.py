import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten
from keras.layers import GlobalAveragePooling1D, RepeatVector, multiply, Reshape,Permute
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Flatten, Reshape, GRU, LSTM, Concatenate, Input, TimeDistributed
from keras.layers.convolutional import *



def model_LSTM_RGB():
    timesteps=32
    data_dim=7168
    print("Create RGB Model")
    input = Input(shape=(timesteps, data_dim))
    LSTM_1 = LSTM(units=1024, activation='tanh', return_sequences=True)(input)
    merged = Concatenate()([LSTM_1, input])
    Res_LSTM_model = Model(inputs=input, outputs=merged)
    return Res_LSTM_model

def model_attn_RGB():
    timesteps=32
    data_dim=7168
    print("Create Attention Model")
    input = Input(shape=(timesteps, data_dim))
    d1 = TimeDistributed(Dense(1024, init='glorot_normal',activation='relu'))(input)
    d1_avg = GlobalAveragePooling1D()(d1)
    output = Dense(32,init='glorot_normal',name='RGB_Attentation',activity_regularizer = l2(0.01), activation='softmax')(d1_avg)
    model = Model(inputs=input, outputs=output)
    return model


def JointDandC_model(lam3,n_neuron):
    print("Create MIL Model")
    model_RGB = model_LSTM_RGB()
    attn_RGB = model_attn_RGB()
    attn_value = attn_RGB.output
    attn_repeat = RepeatVector(8192)(attn_value)
    attn_permute = Permute((2,1))(attn_repeat)
    #################### First-Level Attention ##############
    attn_multiply = multiply([model_RGB.output,attn_permute])
    attn_add = keras.layers.add([attn_multiply,model_RGB.output])
    ############# Detection Branch ##########
    merged_dropout = Dropout(0.6)(attn_add)
    d1= Dense(n_neuron, init='glorot_normal', W_regularizer=l2(lam3))(merged_dropout)
    d1_dropout= Dropout(0.6)(d1)
    detection_output=Dense(1, init='glorot_normal', W_regularizer=l2(lam3),name='Detection_part', activation='sigmoid')(d1_dropout)
    ################### Second-Level Attention ################
    d_r = Dense(24, init='glorot_normal',name='Detection_ATTN', activation='relu')(d1_dropout)
    d_r_flatten = Flatten()(d_r)
    att_classify_output = Dense(32,activation='sigmoid')(d_r_flatten)
    d_r_repeat = RepeatVector(8192)(att_classify_output)
    d_r_permute = Permute((2,1))(d_r_repeat)
    ########### Classification Branch ###########
    attn_detection =multiply([attn_add,d_r_permute])
    attn_detection = keras.layers.add([attn_detection, attn_add])
    combine_feature = GlobalAveragePooling1D()(attn_detection)
    dc1 =Dense(256,init='glorot_normal', W_regularizer=l2(0.001), activation='relu')(combine_feature)
    dc1_dropout= Dropout(0.6)(dc1)
    prob=Dense(14,init='glorot_normal', W_regularizer=l2(0.001),name='Recognition_part', activation='softmax')(dc1_dropout)
    MIL = Model(inputs=[model_RGB.input,attn_RGB.input], outputs=[detection_output,prob])
    return MIL
