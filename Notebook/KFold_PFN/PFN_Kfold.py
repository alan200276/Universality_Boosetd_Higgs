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

# energyflow imports
import energyflow as ef
from energyflow.archs import PFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical


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
# import kerastuner as kt
# import autokeras as ak

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
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
except RuntimeError as e:
# Visible devices must be set before GPUs have been initialized
    logging.info(e)




logging.info("Tensorflow Version is {}".format(tf.__version__))
logging.info("Keras Version is {}".format(tf.keras.__version__))
from tensorflow.python.client import device_lib
logging.info("{}".format(device_lib.list_local_devices()))
tf.device('/device:XLA_GPU:0')

try:
    pt_min, pt_max = float(sys.argv[1]), float(sys.argv[2])
    n_splits = int(sys.argv[3])
    logging.info("PT min: {} PT max: {} N Splits: {}".format(pt_min,pt_max,n_splits))
    logging.info("\n")
    
    
except:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 PFN_Kfold.py pt_min pt_max n_splits *********")
    sys.exit(1)
    

HOMEPATH = "/dicos_ui_home/alanchung/Universality_Boosetd_Higgs/"
data_PFN =  HOMEPATH + "Data_ML/" +"Data_PFN/"
savepath = HOMEPATH + "Data_ML/"

############################################################################################################################################
def dphi(phi,phi_c):

    dphi_temp = phi - phi_c
    while dphi_temp > np.pi:
        dphi_temp = dphi_temp - 2*np.pi
    while dphi_temp < -np.pi:
        dphi_temp = dphi_temp + 2*np.pi
    return (dphi_temp)

# try:
    
data_npz ={
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

logging.info("\r")
logging.info("Load npz File")
logging.info("###############")
ticks_1 = time.time()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i, element in enumerate(data_npz):
    # np.load(data_PFN + str(element) + "_H.npz")["PFN_data_format"].shape(N, 300,
    # (N jets, 300 constituent's space, (cons.eta, cons.phi, cons.pt, cons.e, hj.eta, hj.phi, hj.pt, hj.e))
    data_npz[element][0] = np.load(data_PFN + str(element) + "_H.npz")["PFN_data_format"]
    data_npz[element][1] = np.load(data_PFN + str(element) + "_QCD.npz")["PFN_data_format"]
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
logging.info("\n")
logging.info("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))
logging.info("###############")

for i,(element, npz_element) in enumerate(zip(data_train, data_npz)):

    """
    Pt Range Study
    """
    logging.info("\r")
    logging.info("Preselection")
    logging.info("###############")
    ticks_1 = time.time()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tmp = pd.read_csv(HOMEPATH + "Notebook/KFold_PFN/" + str(element) + ".csv")
    tmp = tmp[(tmp["PTJ_0"] >= pt_min)  & (tmp["PTJ_0"] < pt_max)]
    tmp = tmp[(tmp["MJ_0"] >= 110)  & (tmp["MJ_0"] < 160)]
    data_train[element] = shuffle(tmp)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    logging.info("\n")
    logging.info("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))
    logging.info("###############")

    H_tmp = data_train[element][data_train[element]["target"] == 1]
    QCD_tmp = data_train[element][data_train[element]["target"] == 0]

    H_PFN = data_npz[npz_element][0][H_tmp["index"].values]
    QCD_PFN = data_npz[npz_element][1][QCD_tmp["index"].values]

    data_train[element] = np.concatenate([H_PFN, QCD_PFN])
    target = np.concatenate([np.full(len(H_PFN),1), np.full(len(QCD_PFN),0)])
    data_train[element], target = shuffle(data_train[element], target)
    
    target= to_categorical(target, num_classes=2)

    """
    preprocess by centering jets and normalizing pts
    """
    logging.info("\r")
    logging.info("Preprocessing")
    logging.info("###############")
    ticks_1 = time.time()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for N in range(len(data_train[element])):

        # cons.eta - Jet.eta
        data_train[element][N, :, 0] = data_train[element][N, :, 0] - data_train[element][N, :, 4]

        # dphi(cons.phi, Jet.phi)
        for i in range(len(data_train[element][N, :, 1])):
            data_train[element][N, i, 1] = dphi(data_train[element][N, i, 1], data_train[element][N, i, 5])

        # cons.pt/\SUM(cons.pt)
        data_train[element][N, :, 2] = data_train[element][N, :, 2]/np.sum(data_train[element][N, :, 2])

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    logging.info("\n")
    logging.info("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))
    logging.info("###############")
        
    


logging.info("All Files are loaded!")

logging.info("H jet : QCD jet = 1 : 1")
logging.info("\r")
train = [ len(data_train[element]) for j, element in enumerate(data_train)]
logging.info("{:^8}{:^15}".format("",str(element)))
logging.info("{:^8}{:^15}".format("Train #",train[0]))

    
#     for i, element in enumerate(data_train):
#         total_list = data_train[element].columns
#         break
    
#     logging.info("total_list {}".format(total_list))

# except:
    
#     logging.info("Please create training, test and validation datasets.")
    
    
    
    
logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
logging.info("######################################################################################")
logging.info("\n")
############################################################################################################################################################

PFN_Model_A1 = {
#               "herwig_ang" : 0,
#               "pythia_def" : 0, 
#               "pythia_vin" : 0, 
              "pythia_dip" : 0, 
            }


kf = KFold(n_splits = n_splits)

for i, (model, trainingdata) in enumerate(zip(PFN_Model_A1, data_train)):

    logging.info("PFN Model: {}  Training Data: {} ".format(model, trainingdata))
    
    for model_index, (train_index, val_index) in enumerate(kf.split(target)):

        x_train, target_train = data_train[trainingdata][:,:,:3][train_index], target[train_index]
        x_val, target_val = data_train[trainingdata][:,:,:3][val_index], target[val_index]
        
        # if model_index == 1:
        #     break
        
       
#         try:
#             PFN_Model_A1[model] = load_model("./"+str(model)+"_KFold/PFN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_PFN_"+str(model_index)+ ".h5")
#             logging.info(str(model) + " PFN model 1 is loaded!")
#             logging.info("######################################################################################")
#             logging.info("\n")

#         except:
        logging.info("Let's train a PFN model for {}".format(model))
        logging.info("######################################################################################")
        logging.info("\n")
        if os.path.exists("./"+str(model)+"_KFold/PFN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))) == 0:
            os.mkdir("./"+str(model)+"_KFold/PFN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max)))


        ticks_1 = time.time()

        Performance_Frame = {
                        "AUC" : [0],
                        "max_sig" : [0],
                        "r05" : [0],
                        "time": [0]
                        }

        # network architecture parameters
        Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
        # Phi_sizes, F_sizes = (100, 100, 256), (100, 100, 100)

        # network training parameters
        num_epoch = 200
        batch_size = 512

        # build architecture
#         model_PFN = PFN(input_dim=data_train[element][:,:,:3].shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes) #version1
#         model_PFN = PFN(input_dim=data_train[element][:,:,:3].shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, F_dropouts=0.01) #version2 #2022/07/19
#         model_PFN = PFN(input_dim=data_train[element][:,:,:3].shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, F_dropouts=0.1) #version3 #2022/07/21
#         model_PFN = PFN(input_dim=data_train[element][:,:,:3].shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, F_dropouts=0.1, latent_dropout=0.1) #version4 #2022/07/26
        model_PFN = PFN(input_dim=data_train[element][:,:,:3].shape[-1], 
                        Phi_sizes=Phi_sizes, F_sizes=F_sizes, 
                        F_dropouts=0.15, latent_dropout=0.15) #version4 #2022/07/27
        
        # train model
        History = model_PFN.fit(x_train, target_train,
                            epochs=num_epoch,
                            batch_size=batch_size,
                            validation_data=(x_val, target_val),
                            verbose=1
                           )


        model_PFN.model.save("./"+str(model)+"_KFold/PFN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_PFN_"+str(model_index)+ ".h5")
        hist_df = pd.DataFrame(History.history) 
        hist_df.to_csv("./"+str(model)+"_KFold/PFN_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_history_PFN_"+str(model_index)+ ".csv")


        prediction_test =  model_PFN.predict(x_val)
        discriminator_test = prediction_test[:,1]
        discriminator_test = discriminator_test/(max(discriminator_test))

        Performance_Frame["AUC"][0] = metrics.roc_auc_score(np.asarray(target_val)[:,1],discriminator_test)
        FalsePositiveFull, TruePositiveFull, _ = metrics.roc_curve(np.asarray(target_val)[:,1],discriminator_test)
        tmp = np.where(FalsePositiveFull != 0)
        Performance_Frame["max_sig"][0] = max(TruePositiveFull[tmp]/np.sqrt(FalsePositiveFull[tmp])) 
        tmp = np.where(TruePositiveFull >= 0.5)
        Performance_Frame["r05"][0]= 1./FalsePositiveFull[tmp[0][0]]

        Performance_Frame["time"][0] = (time.time() - ticks_1)/60.

        dataframe = pd.DataFrame(Performance_Frame)


        try:
            save_to_csvdata = pd.read_csv("./"+str(model)+"_KFold/PFN_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv")
            DATA = pd.concat([save_to_csvdata, dataframe], ignore_index=True, axis=0,join='inner')
            DATA.to_csv("./"+str(model)+"_KFold/PFN_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv", index = 0)

        except:
            dataframe.to_csv("./"+str(model)+"_KFold/PFN_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv", index = 0)

        ticks_2 = time.time()
        ############################################################################################################################################################
        totaltime =  ticks_2 - ticks_1
        logging.info("\n")
        logging.info("\033[3;33mTime consumption : {:.4f} min for PFN\033[0;m".format(totaltime/60.))
        logging.info("######################################################################################")
        logging.info("\n")
            