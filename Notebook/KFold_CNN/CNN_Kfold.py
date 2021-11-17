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
from tqdm import tqdm 
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
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
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



try:
    pt_min, pt_max = float(sys.argv[1]), float(sys.argv[2])
    n_splits = int(sys.argv[3])
    logging.info("PT min: {} PT max: {} N Splits: {}".format(pt_min,pt_max,n_splits))
    logging.info("\n")
    
    
except:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 CNN_Kfold.py pt_min pt_max n_splits *********")
    sys.exit(1)
    




HOMEPATH = "/dicos_ui_home/alanchung/Universality_Boosetd_Higgs/"
JetImagePath =  HOMEPATH + "Data_ML/" +"Image_Directory/"
savepath = HOMEPATH + "Data_ML/"

############################################################################################################################################
"""
Define Function
"""

def CNN_Model(name):
    """
    Declare the Input Shape
    """
    input_shape = (3,40,40)#(1, 40,40)


    """
    Create a Sequential Model
    """
    model_CNN = Sequential(name = "Model_CNN_"+str(name))


    """
    Add a "Conv2D" Layer into the Sequential Model
    """
    model_CNN.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
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
    model_CNN.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
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
    model_CNN.add(Dense(300, activation='relu', name = 'jet_dense_1'))
    model_CNN.add(Dense(150, activation='relu', name = 'jet_dense_2'))


    model_CNN.add(Dropout(0.01, name = 'Dropout'))


    """
    Add Output "Dense" Layer with 2 neurons into the Sequential Model
    """
    model_CNN.add(Dense(1, activation='sigmoid', name = 'output'))
    # model_CNN.add(Dense(2, activation='softmax', name = 'output'))

    """
    Declare the Optimizer
    """
    model_opt = keras.optimizers.Adadelta()
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


def generator(data_source, batch_size, datadict):
    while True:
        for start in range(0, len(data_source), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(data_source))
            for img_path in range(start, end):
                
                x_jet_path = savepath + "Image_Directory/"+ data_source["JetImage"].iloc[img_path]
                x_jet = np.load(x_jet_path)["jet_image"]
                if np.isnan(x_jet).any() == True:
                    continue 
    
                x_jet = np.divide((x_jet - Norm_dict[datadict][0]), (np.sqrt(Norm_dict[datadict][1])+1e-5))#[0].reshape(1,40,40)
                x_batch.append(x_jet)
                y_batch.append(data_source["Y"].iloc[img_path])
                
            yield (np.array(x_batch), np.array(y_batch))#to_categorical(np.array(y_batch)))
############################################################################################################################################

    
try:
    
    data_dict ={
#             "herwig_ang" : [0,0],
#             "pythia_def" : [0,0],
#             "pythia_vin" : [0,0],
            "pythia_dip" : [0,0],
#             "sherpa_def" : [0,0],
              }  
    
    Norm_dict ={
#             "herwig_ang" : [0,0],
#             "pythia_def" : [0,0],
#             "pythia_vin" : [0,0],
            "pythia_dip" : [0,0],
#             "sherpa_def" : [0,0],
              }  
    
    data_train = {
#             "herwig_ang_train" : 0,
#             "pythia_def_train" : 0,
#             "pythia_vin_train" : 0,
            "pythia_dip_train" : 0,
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
    
    
    
    
logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
logging.info("######################################################################################")
logging.info("\n")
############################################################################################################################################################



CNN_Model_A1 = {
#               "herwig_ang" : 0,
#               "pythia_def" : 0, 
#               "pythia_vin" : 0, 
              "pythia_dip" : 0, 
#               "sherpa_def" : 0,
            }


kf = KFold(n_splits = n_splits)


for i, (model, trainingdata, datadict) in enumerate(zip(CNN_Model_A1, data_train, data_dict)):

    logging.info("CNN Model: {}  Training Data: {}".format(model, trainingdata))
    
    for model_index, (train_index, val_index) in enumerate(kf.split(data_train[trainingdata]["Y"])):
        training_data = data_train[trainingdata].iloc[train_index]
        validation_data = data_train[trainingdata].iloc[val_index]
        
#         if model_index == 1:
#             break
        
       
        try:
            CNN_Model_A1[model] = load_model("./"+str(model)+"_KFold/CNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_CNN_"+str(model_index)+ ".h5")
            logging.info(str(model) + " CNN model 1 is loaded!")
            logging.info("######################################################################################")
            logging.info("\n")

        except:
            logging.info("Let's train a CNN model for {}".format(model))
            logging.info("######################################################################################")
            logging.info("\n")
            if os.path.exists("./"+str(model)+"_KFold/CNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))) == 0:
                os.mkdir("./"+str(model)+"_KFold/CNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max)))
                
                
#             x_train_jet, target_train = Loading_Data(training_data, datadict, start=0, stop= len(training_data))
            x_val_jet, target_val = Loading_Data(validation_data, datadict, start=0, stop= len(validation_data))
    

            ticks_1 = time.time()

            model_CNN = CNN_Model(model+"_"+str(model_index))

            Performance_Frame = {
                            "AUC" : [0],
                            "max_sig" : [0],
                            "r05" : [0],
                            "time": [0]
                            }


            check_list=[]
            csv_logger = CSVLogger("./"+str(model)+"_KFold/CNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_CNN_training_log_"+str(model_index)+".csv")
            checkpoint = ModelCheckpoint(
                                filepath= "./"+str(model)+"_KFold/CNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_CNN_checkmodel_"+str(model_index)+".h5",
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
            
            
#             History = model_CNN.fit(
#                                     np.asarray(x_train_jet), np.asarray(target_train),
#                                     validation_data = (np.asarray(x_val_jet), np.asarray(target_val)),
#             #                         validation_split = 0.2,
#                                     batch_size=512,
#                                     epochs=200,
#                                     callbacks=check_list,
#                                     shuffle=True,
#                                     verbose=1
#                                     )
            batch_size = 512
            nb_train_samples = len(training_data)
            nb_val_samples = len(validation_data)
            
            History = model_CNN.fit(
                                    generator(training_data, batch_size, datadict),
                                    epochs= 200,
                                    steps_per_epoch= nb_train_samples // batch_size,
                                    validation_data= generator(validation_data, batch_size, datadict),
                                    validation_steps = nb_val_samples // batch_size,
                                    verbose=1
                                    )


            model_CNN.save("./"+str(model)+"_KFold/CNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_CNN_"+str(model_index)+ ".h5")
            hist_df = pd.DataFrame(History.history) 
            hist_df.to_csv("./"+str(model)+"_KFold/CNN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_history_CNN_"+str(model_index)+ ".csv")

            CNN_Model_A1[model] = model_CNN
            
            prediction_test =  model_CNN.predict(np.asarray(x_val_jet))
            discriminator_test = prediction_test
            discriminator_test = discriminator_test/(max(discriminator_test))
            
            Performance_Frame["AUC"][0] = metrics.roc_auc_score(validation_data["Y"],discriminator_test)
            FalsePositiveFull, TruePositiveFull, _ = metrics.roc_curve(validation_data["Y"],discriminator_test)
            tmp = np.where(FalsePositiveFull != 0)
            Performance_Frame["max_sig"][0] = max(TruePositiveFull[tmp]/np.sqrt(FalsePositiveFull[tmp])) 
            tmp = np.where(TruePositiveFull >= 0.5)
            Performance_Frame["r05"][0]= 1./FalsePositiveFull[tmp[0][0]]
            
            Performance_Frame["time"][0] = (time.time() - ticks_1)/60.
            
            dataframe = pd.DataFrame(Performance_Frame)
            
            
            try:
                save_to_csvdata = pd.read_csv("./"+str(model)+"_KFold/CNN_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv")
                DATA = pd.concat([save_to_csvdata, dataframe], ignore_index=True, axis=0,join='inner')
                DATA.to_csv("./"+str(model)+"_KFold/CNN_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv", index = 0)
                
            except:
                dataframe.to_csv("./"+str(model)+"_KFold/CNN_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv", index = 0)
            
            ticks_2 = time.time()
            ############################################################################################################################################################
            totaltime =  ticks_2 - ticks_1
            logging.info("\n")
            logging.info("\033[3;33mTime consumption : {:.4f} min for CNN\033[0;m".format(totaltime/60.))
            logging.info("######################################################################################")
            logging.info("\n")
            
            

