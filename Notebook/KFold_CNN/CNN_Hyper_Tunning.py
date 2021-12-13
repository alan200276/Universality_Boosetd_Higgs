#!/usr/bin/env python
# encoding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
# Install TensorFlow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten , Convolution2D, MaxPooling2D , Lambda, Conv2D, Activation,Concatenate
from tensorflow.keras.layers import ActivityRegularization
from tensorflow.keras.optimizers import Adam , SGD , Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers , initializers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import NumpyArrayIterator



from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
# from xgboost import XGBClassifier
import tensorflow.keras.backend as K
from sklearn import metrics

# !pip3 install keras-tuner --upgrade
# !pip3 install autokeras
import kerastuner as kt
# import autokeras as ak

# Import local libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import importlib
from tqdm import tqdm 
import os
import sys
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '64'


gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    logging.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
except RuntimeError as e:
# Visible devices must be set before GPUs have been initialized
    logging.info(e)


logging.info("Tensorflow Version is {}".format(tf.__version__))
logging.info("Keras Version is {}".format(tf.keras.__version__))
from tensorflow.python.client import device_lib
logging.info(device_lib.list_local_devices())
tf.device('/device:XLA_GPU:0')


def Loading_Data(data_source, datadict, start=0, stop=20000):
    x_jet, target = [], []

    time.sleep(0.5)
    for k in tqdm(range(start,len(data_source))):
        x_jet_path = savepath + "Image_Directory/"+ data_source["JetImage"].iloc[k]
        x_jet_tmp = np.load(x_jet_path)["jet_image"]
        if np.isnan(x_jet_tmp).any() == True:
            continue 

        target.append(data_source["Y"].iloc[k])
        x_jet_tmp = np.divide((x_jet_tmp - Norm_dict[datadict][0]), (np.sqrt(Norm_dict[datadict][1])+1e-5))#[0].reshape(1,40,40)
        x_jet.append(x_jet_tmp)


        if k == stop:
            break

    return np.asarray(x_jet), np.asarray(target)


############################################################################################################################################################
ticks_1 = time.time()


"""
Load Pythia Default Data
"""

HOMEPATH = "/dicos_ui_home/alanchung/Universality_Boosetd_Higgs/"
JetImagePath =  HOMEPATH + "Data_ML/" +"Image_Directory/"
savepath = HOMEPATH + "Data_ML/"


    
try:
    
    data_dict ={
#             "herwig_ang" : [0,0],
            "pythia_def" : [0,0],
#             "pythia_vin" : [0,0],
#             "pythia_dip" : [0,0],
#             "sherpa_def" : [0,0],
              }  
    
    Norm_dict ={
#             "herwig_ang" : [0,0],
            "pythia_def" : [0,0],
#             "pythia_vin" : [0,0],
#             "pythia_dip" : [0,0],
#             "sherpa_def" : [0,0],
              }  
    
    data_train = {
#             "herwig_ang_train" : 0,
            "pythia_def_train" : 0,
#             "pythia_vin_train" : 0,
#             "pythia_dip_train" : 0,
#             "sherpa_def_train" : 0
            }  
    

    for i, element in enumerate(data_dict):
        data_dict[element][0] = pd.read_csv(savepath + str(element) + "_H_dict.csv")
        data_dict[element][1] = pd.read_csv(savepath + str(element) + "_QCD_dict.csv")
#         logging.info(len(data_dict[element][0]),len(data_dict[element][1]))
    
    for i, element in enumerate(Norm_dict):
        average_H = np.load(savepath + "average" + "_" + str(element) + "_H.npy")
        variance_H = np.load(savepath + "variance" + "_" + str(element) + "_H.npy")
        average_QCD = np.load(savepath + "average" + "_" + str(element) + "_QCD.npy")
        variance_QCD = np.load(savepath + "variance" + "_" + str(element) + "_QCD.npy")
        length_H = len(data_dict[element][0])
        length_QCD = len(data_dict[element][1])
        
        Norm_dict[element][0] = (average_H*length_H + average_QCD*length_QCD)/(length_H+length_QCD)
        Norm_dict[element][1] =  variance_H + variance_QCD
        
    for i,(element, dict_element) in enumerate(zip(data_train, data_dict)):
        
        """
        Pt Range Study
        """
        pt_min, pt_max = 300, 500
        tmp = pd.read_csv(HOMEPATH + "Notebook/KFold_CNN/" + str(element) + ".csv")
        tmp = tmp[(tmp["PTJ_0"] >= pt_min)  & (tmp["PTJ_0"] < pt_max)]
        tmp = tmp[(tmp["MJ_0"] >= 110)  & (tmp["MJ_0"] < 160)]
        data_train[element] = shuffle(tmp)
        
        H_tmp = data_train[element][data_train[element]["target"] == 1]
        QCD_tmp = data_train[element][data_train[element]["target"] == 0]
        
        H_dict = data_dict[dict_element][0].iloc[H_tmp["index"].values]
        QCD_dict = data_dict[dict_element][1].iloc[QCD_tmp["index"].values]
        
        data_train[element] = pd.concat([H_dict, QCD_dict], ignore_index=True, axis=0,join='inner')
        data_train[element] = shuffle(data_train[element])
        


    logging.info("All Files are loaded!")

    logging.info("H jet : QCD jet = 1 : 1")
    logging.info("\r")
    train = [ len(data_train[element]) for j, element in enumerate(data_train)]
    logging.info("{:^8}{:^15}".format("",str(element)))
    logging.info("{:^8}{:^15}".format("Train #",train[0]))
    
    
    for i, element in enumerate(data_train):
        total_list = data_train[element].columns
        break
    
    logging.info("total_list {}".format(total_list))

except:
    
    logging.info("Please create training, test and validation datasets.")

################################################################################
    
    
"""
Load Jet Images
"""
x_jet, target = Loading_Data(data_train["pythia_def_train"], "pythia_def", start=0, stop= len(data_train["pythia_def_train"]))

################################################################################



"""
Split Training, Validation and Test Dataset
"""
length = len(x_jet)
x_train_jet, target_train = x_jet[:int(length/10*8)], target[:int(length/10*8)]
x_val_jet, target_val = x_jet[int(length/10*8):int(length/10*9)], target[int(length/10*8):int(length/10*9)]
x_test_jet, target_test = x_jet[int(length/10*9):], target[int(length/10*9):]

logging.info("training length: {}".format(len(x_train_jet)))
logging.info("validation length: {}".format(len(x_val_jet)))
logging.info("test length: {}".format(len(x_test_jet)))
logging.info("Total length: {}".format(len(x_train_jet)+len(x_val_jet)+len(x_test_jet)))
logging.info("Total length: {}".format(length))
################################################################################


"""
Define Model for Tuning
"""

def CNN_Model(hp):
    """
    Declare the Input Shape
    """
    input_shape = (3,40,40)#(1, 40,40)


    """
    Create a Sequential Model
    """
    model_CNN = Sequential(name = "Model_CNN_Pythia_Default")


    """
    Add a "Conv2D" Layer into the Sequential Model
    """
    model_CNN.add(Conv2D(
                     filters=hp.Int("Conv2D_1_filter", min_value=16, max_value=128, step=16), 
                     kernel_size=hp.Choice('Conv2D_1_kernel', values = [3,5,7]),
                     strides=hp.Choice('Conv2D_1_stride', values = [1]),
                     activation='relu',
                     data_format='channels_first',
    #                data_format='channels_last',
                    input_shape=input_shape, 
                    name = 'Conv2D_1'))

    """
    Add a "MaxPooling2D" Layer into the Sequential Model
    """
    model_CNN.add(MaxPooling2D(pool_size=(2, 2), 
                               strides=(2, 2),
#                                data_format='channels_first', 
    #                            data_format='channels_last',
                               name = 'jet_MaxPooling_1'))

    """
    Add a "Conv2D" Layer into the Sequential Model
    """
    model_CNN.add(Conv2D(
                 filters=hp.Int("Conv2D_2_filter", min_value=16, max_value=128, step=16), 
                 kernel_size=hp.Choice('Conv2D_2_kernel', values = [3,5,7]),
                 strides=hp.Choice('Conv2D_2_stride', values = [1]),
                 activation='relu',
                 data_format='channels_first',
#                data_format='channels_last',
                input_shape=input_shape, 
                name = 'Conv2D_2'))

    """
    Add a "MaxPooling2D" Layer into the Sequential Model
    """
    model_CNN.add(MaxPooling2D(pool_size=(2, 2), 
                               strides=(2, 2),
                               data_format='channels_first', 
    #                            data_format='channels_last',
                               name = 'jet_MaxPooling_2'))



    """
    Flatten
    """
    model_CNN.add(Flatten(name = 'jet_flatten'))


    """
    Put Output from Flatten Layer into "Dense" Layer with 300 neurons
    """
    for i in range(hp.Int("num_layers", 1, 3)):
        model_CNN.add(
            keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=100, max_value=500, step=50),
                activation='relu',
            )
        )

    
    if hp.Boolean("dropout"):
        model_CNN.add(keras.layers.Dropout(rate=0.01, name = 'Dropout'))

    """
    Add Output "Dense" Layer with 2 neurons into the Sequential Model
    """
    model_CNN.add(Dense(1, activation='sigmoid', name = 'output'))
    # model_CNN.add(Dense(2, activation='softmax', name = 'output'))

    """
    Declare the Optimizer
    """
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
    model_opt = keras.optimizers.Adadelta(learning_rate=learning_rate)
    # model_opt = keras.optimizers.Adam()


    """
    Compile Model
    """
    model_CNN.compile(
    #                     loss="categorical_crossentropy",
                       loss = "binary_crossentropy",
                       optimizer=model_opt,
                       metrics=["accuracy","mse"])

    """
    Print Architecture
    """
    model_CNN.summary()

    return model_CNN
################################################################################

    
"""
Start the search (RandomSearch, BayesianOptimization and Hyperband)
Here, we use RandomSearch
"""
tuner = kt.RandomSearch(hypermodel=CNN_Model,
                        objective="val_loss",
                        max_trials=50,
                        executions_per_trial=1, #The number of models that should be built and fit for each trial
                        overwrite=True,
                        directory="CNN_Model_Hyper_Tunning",
                        project_name="Universality_CNN"
                        )
################################################################################


"""
Tuner Summary
"""
tuner.search_space_summary()
################################################################################


"""
Start Tuning
"""
tuner.search(np.asarray(x_train_jet), np.asarray(target_train),
             epochs=200, 
             validation_data=(np.asarray(x_val_jet), np.asarray(target_val)))

tuner.results_summary()
################################################################################


"""
Get the Best Model
"""
best_model = tuner.get_best_models()[0]
best_model.summary()
################################################################################
    
    
"""
Save the Best Model
"""    
best_model.save("./CNN_Model_Hyper_Tuning/Universality/CNN_Tuning.h5")
    
prediction_test =  best_model.predict(np.asarray(x_test_jet))
discriminator_test = prediction_test
discriminator_test = discriminator_test/(max(discriminator_test))
auc = metrics.roc_auc_score(target_test,discriminator_test)
logging.info("auc: {}".format(auc))
    
    
ticks_2 = time.time()
############################################################################################################################################################
totaltime =  ticks_2 - ticks_1
logging.info("\n")
logging.info("\033[3;33mTime consumption : {:.4f} min for CNN\033[0;m".format(totaltime/60.))
logging.info("######################################################################################")
logging.info("\n")

