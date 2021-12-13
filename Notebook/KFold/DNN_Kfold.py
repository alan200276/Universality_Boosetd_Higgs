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
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
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



# def DNN_Model(name):
    
#     model_DNN = Sequential(name = "Model_DNN_"+str(name))



#     model_DNN.add(keras.Input(shape=(len(features),), name = 'input'))
# #     model_DNN_1.add(Dense(256, activation='relu', name = 'dense_1'))
#     model_DNN.add(Dense(64, activation='relu', name = 'dense_1'))
#     model_DNN.add(Dense(32, activation='relu', name = 'dense_2'))
# #     model_DNN_1.add(Dense(32, activation='relu', name = 'dense_4'))
#     model_DNN.add(Dense(1, activation='sigmoid', name = 'dense_3'))
# #     model_DNN_1.add(ActivityRegularization(l2=0.1, name = 'Regularization'))
#     model_DNN.add(Dropout(0.00001))
    
    
#     # model_opt = keras.optimizers.Adadelta()
#     model_opt = keras.optimizers.Adam()
#     model_DNN.compile(loss="binary_crossentropy",#keras.losses.binary_crossentropy
#                               optimizer=model_opt,
#                               metrics=['accuracy'])

#     model_DNN.summary()

#     return model_DNN


def DNN_Model(name):
    
    model_DNN = Sequential(name = "Model_DNN_"+str(name))

    model_DNN.add(keras.Input(shape=(len(features),), name = 'input'))
    model_DNN.add(Dense(224, activation='relu', name = 'dense_1'))
    model_DNN.add(Dense(928, activation='relu', name = 'dense_2'))
    model_DNN.add(Dense(288, activation='relu', name = 'dense_3'))
    model_DNN.add(Dense(1024, activation='relu', name = 'dense_4'))
    model_DNN.add(keras.layers.Dropout(rate=0.01))
    model_DNN.add(Dense(1, activation='sigmoid', name = 'dense_output'))
    
    
    # model_opt = keras.optimizers.Adadelta()
    model_opt = keras.optimizers.Adam(learning_rate=6.5428e-05)
    model_DNN.compile(loss="binary_crossentropy",#keras.losses.binary_crossentropy
                              optimizer=model_opt,
                              metrics=['accuracy'])

    model_DNN.summary()

    return model_DNN




try:
    pt_min, pt_max = float(sys.argv[1]), float(sys.argv[2])
    n_splits = int(sys.argv[3])
    logging.info("PT min: {} PT max: {} N Splits: {}".format(pt_min,pt_max,n_splits))
    logging.info("\n")
    
    
except:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 DNN_Kfold.py pt_min pt_max n_splits *********")
    sys.exit(1)
    




HOMEPATH = "/dicos_ui_home/alanchung/Universality_Boosetd_Higgs/"
# Data_High_Level_Features_path =  HOMEPATH + "Data_High_Level_Features/"
# savepath = HOMEPATH + "Data_ML/"

try:
    
    data_train = {
#             "herwig_ang_train" : 0,
#             "pythia_def_train" : 0,
#             "pythia_vin_train" : 0,
            "pythia_dip_train" : 0,
#             "sherpa_def_train" : 0
            }  
    
    for i, element in enumerate(data_train):
#         data_train[element] = pd.read_csv(savepath + "BDT/" + str(element) + ".csv")
    
        """
        Pt Range Study
        """
#         pt_min, pt_max = 300, 400
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
    
    
    
    
logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
logging.info("######################################################################################")
logging.info("\n")
############################################################################################################################################################

 
features = ["MJ_0","t21_0","D21_0","D22_0","C21_0","C22_0"] #7/14
# features = ["MJ1_0","t211_0","D211_0","D221_0","C211_0","C221_0","MJ2_0","t212_0","D212_0","D222_0","C212_0","C222_0"] # 7/14
# features = ["MJ2_0","t212_0","D212_0","D222_0","C212_0","C222_0"]#7/27

DNN_Model_A1 = {
#               "herwig_ang" : 0,
#               "pythia_def" : 0, 
#               "pythia_vin" : 0, 
              "pythia_dip" : 0, 
#               "sherpa_def" : 0,
            }


kf = KFold(n_splits = n_splits)


                         
# skf = StratifiedKFold(n_splits = 5, random_state = 7, shuffle = True) 



for i,(model, trainingdata) in enumerate(zip(DNN_Model_A1, data_train)):

    logging.info("DNN Model: {}  Training Data: {}".format(model, trainingdata))
    
    for model_index, (train_index, val_index) in enumerate(kf.split(data_train[trainingdata]["target"])):
        training_data = data_train[trainingdata].iloc[train_index]
        validation_data = data_train[trainingdata].iloc[val_index]

        try:
            DNN_Model_A1[model] = load_model("./"+str(model)+"_KFold/DNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_DNN_"+str(model_index)+ ".h5")
            logging.info(str(model) + " DNN model 1 is loaded!")
            logging.info("######################################################################################")
            logging.info("\n")

        except:
            logging.info("Let's train a DNN model for {}".format(model))
            logging.info("######################################################################################")
            logging.info("\n")
            if os.path.exists("./"+str(model)+"_KFold/DNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))) == 0:
                os.mkdir("./"+str(model)+"_KFold/DNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max)))
                
                
            ticks_1 = time.time()

            model_DNN = DNN_Model(model+"_"+str(model_index))

            Performance_Frame = {
                            "AUC" : [0],
                            "max_sig" : [0],
                            "r05" : [0],
                            "time": [0]
                            }


            check_list=[]
            csv_logger = CSVLogger("./"+str(model)+"_KFold/DNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_DNN_training_log_"+str(model_index)+".csv")
            checkpoint = ModelCheckpoint(
                                filepath= "./"+str(model)+"_KFold/DNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_DNN_checkmodel_"+str(model_index)+".h5",
                                save_best_only=True,
                                verbose=0)
            earlystopping = EarlyStopping(
                                monitor="val_loss",
                                min_delta=0,
                                patience=20,
                                verbose=0,
                                mode="auto",
                                baseline=None,
                                restore_best_weights=False,
                            )

            check_list.append(checkpoint)
            check_list.append(csv_logger)
            check_list.append(earlystopping)
            History = model_DNN.fit(np.asarray(training_data[features]), np.asarray(training_data["target"]),
                            validation_data = (np.asarray(validation_data[features]), np.asarray(validation_data["target"])),
                            batch_size=64,
                            epochs=200,
                            callbacks=check_list,
                            verbose=0)


            model_DNN.save("./"+str(model)+"_KFold/DNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_DNN_"+str(model_index)+ ".h5")
            hist_df = pd.DataFrame(History.history) 
            hist_df.to_csv("./"+str(model)+"_KFold/DNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_history_DNN_"+str(model_index)+ ".csv")

            DNN_Model_A1[model] = model_DNN
            
            prediction_test =  model_DNN.predict(np.asarray(validation_data[features]))
            discriminator_test = prediction_test
            discriminator_test = discriminator_test/(max(discriminator_test))
            
            Performance_Frame["AUC"][0] = metrics.roc_auc_score(validation_data["target"],discriminator_test)
            FalsePositiveFull, TruePositiveFull, _ = metrics.roc_curve(validation_data["target"],discriminator_test)
            tmp = np.where(FalsePositiveFull != 0)
            Performance_Frame["max_sig"][0] = max(TruePositiveFull[tmp]/np.sqrt(FalsePositiveFull[tmp])) 
            tmp = np.where(TruePositiveFull >= 0.5)
            Performance_Frame["r05"][0]= 1./FalsePositiveFull[tmp[0][0]]
            
            Performance_Frame["time"][0] = (time.time() - ticks_1)/60.
            
            dataframe = pd.DataFrame(Performance_Frame)
            
            
            try:
                save_to_csvdata = pd.read_csv("./"+str(model)+"_KFold/DNN_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv")
                DATA = pd.concat([save_to_csvdata, dataframe], ignore_index=True, axis=0,join='inner')
                DATA.to_csv("./"+str(model)+"_KFold/DNN_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv", index = 0)
                
            except:
                dataframe.to_csv("./"+str(model)+"_KFold/DNN_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv", index = 0)
                
            
            ticks_2 = time.time()
            ############################################################################################################################################################
            totaltime =  ticks_2 - ticks_1
            logging.info("\n")
            logging.info("\033[3;33mTime consumption : {:.4f} min for DNN\033[0;m".format(totaltime/60.))
            logging.info("######################################################################################")
            logging.info("\n")

