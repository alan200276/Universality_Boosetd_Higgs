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
import autokeras as ak

# Import local libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import importlib
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


############################################################################################################################################################
ticks_1 = time.time()


"""
Load Pythia Default Data
"""

HOMEPATH = "/dicos_ui_home/alanchung/Universality_Boosetd_Higgs/"
# Data_High_Level_Features_path =  HOMEPATH + "Data_High_Level_Features/"
# savepath = HOMEPATH + "Data_ML/"

try:
    
    data_train = {
#             "herwig_ang_train" : 0,
            "pythia_def_train" : 0,
#             "pythia_vin_train" : 0,
#             "pythia_dip_train" : 0,
#             "sherpa_def_train" : 0
            }  
    
    for i, element in enumerate(data_train):
#         data_train[element] = pd.read_csv(savepath + "BDT/" + str(element) + ".csv")
    
        """
        Pt Range Study
        """
        pt_min, pt_max = 300, 500
        tmp = pd.read_csv(HOMEPATH + "Notebook/KFold/" + str(element) + ".csv")
        tmp = tmp[(tmp["PTJ_0"] >= pt_min)  & (tmp["PTJ_0"] < pt_max)]
        tmp = tmp[(tmp["MJ_0"] >= 110)  & (tmp["MJ_0"] < 160)]
        data_train[element] = shuffle(tmp)
    
    
    
    logging.info("All Files are loaded!")

    logging.info("H jet : QCD jet = 1 : 1")
    
    for i, element in enumerate(data_train):
        
        logging.info("{}, # of H jet: {}".format(element, len(data_train[element][ data_train[element]["PRO"] == "H"])))
        logging.info("{}, # of QCD jet: {}".format(element, len(data_train[element][ data_train[element]["PRO"] == "QCD"])))
        
    logging.info("\r")
    
    
    train = [ len(data_train[element]) for j, element in enumerate(data_train)]
    logging.info("{:^8}{:^15}".format("",str(element)))
    logging.info("{:^8}{:^15}".format("Train #",train[0]))


    for i, element in enumerate(data_train):
        total_list = data_train[element].columns
        break
    
    logging.info("total_list: {}".format(total_list))

except:
    
    logging.info("Please create training, test and validation datasets.")
    raise ValueError("Please create training, test and validation datasets.")
################################################################################
    
    
"""
Define High-level Features 
"""
features = ["MJ_0","t21_0","D21_0","D22_0","C21_0","C22_0"]
################################################################################



"""
Split Training, Validation and Test Dataset
"""
length = len(data_train["pythia_def_train"])
training_data = data_train["pythia_def_train"][:int(length/10*8)]
validation_data = data_train["pythia_def_train"][int(length/10*8):int(length/10*9)]
test_data = data_train["pythia_def_train"][int(length/10*9):]

logging.info("training length: {}".format(len(training_data)))
logging.info("validation length: {}".format(len(validation_data)))
logging.info("test length: {}".format(len(test_data)))
logging.info("Total length: {}".format(len(training_data)+len(validation_data)+len(test_data)))
logging.info("Total length: {}".format(length))
################################################################################


"""
Define Model for Tuning
"""
def DNN_Model(hp):
    
    model_DNN = Sequential(name = "Model_DNN_Pythia_Default")
    model_DNN.add(keras.Input(shape=(len(features),), name = 'input'))
#     model_DNN.add(keras.layers.Dense(hp.Choice('units', [8, 16, 32]), activation='relu', name = 'dense_1'))
    for i in range(hp.Int("num_layers", 1, 6)):
        model_DNN.add(
            keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=1024, step=32),
#                 activation=hp.Choice("activation", ["relu", "tanh"]),
                activation='relu',
            )
        )
        
    if hp.Boolean("dropout"):
        model_DNN.add(keras.layers.Dropout(rate=0.01))
        
    model_DNN.add(Dense(1, activation='sigmoid'))
    
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
    # model_opt = keras.optimizers.Adadelta()
    model_opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model_DNN.compile(loss="binary_crossentropy",#keras.losses.binary_crossentropy
                              optimizer=model_opt,
                              metrics=['accuracy'])

    return model_DNN
################################################################################

    
"""
Start the search (RandomSearch, BayesianOptimization and Hyperband)
Here, we use RandomSearch
"""
tuner = kt.RandomSearch(hypermodel=DNN_Model,
                        objective="val_loss",
                        max_trials=50,
                        executions_per_trial=1, #The number of models that should be built and fit for each trial
                        overwrite=True,
                        directory="DNN_Model_Hyper_Tuning",
                        project_name="Universality_DNN"
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
tuner.search(training_data[features], np.asarray(training_data["target"]), 
             epochs=200, 
             validation_data=(validation_data[features], np.asarray(validation_data["target"])))
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
best_model.save("./DNN_Model_Hyper_Tuning/Universality/Tuning.h5")
    
prediction_test =  best_model.predict(np.asarray(test_data[features]))
discriminator_test = prediction_test
discriminator_test = discriminator_test/(max(discriminator_test))
auc = metrics.roc_auc_score(test_data["target"],discriminator_test)
logging.info("auc: {}".format(auc))
    
    
ticks_2 = time.time()
############################################################################################################################################################
totaltime =  ticks_2 - ticks_1
logging.info("\n")
logging.info("\033[3;33mTime consumption : {:.4f} min for DNN\033[0;m".format(totaltime/60.))
logging.info("######################################################################################")
logging.info("\n")

