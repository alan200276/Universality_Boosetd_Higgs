#!/usr/bin/env python
# encoding: utf-8
# Numerical Packages
import numpy as np
import pandas as pd

#Common packages
import copy
from tqdm import tqdm
 
#Plot's Making  Packages
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

#System Packages
# Import local libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import importlib
import os
from os.path import isdir, isfile, join
import sys
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

# Learning packages
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from joblib import dump, load



os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '64'


def BDT_Model():
    
    rand = np.random.randint(1000000)
    clf_GBDT = GradientBoostingClassifier(
                n_estimators=1000,
#                 n_estimators=10,
                learning_rate=0.02,
                max_depth=5, 
                min_samples_split = 0.25,
                min_samples_leaf = 0.05,
    #             min_impurity_split = 0.00001,
    #             validation_fraction = 0.1,
                random_state= rand,  #np.random,
                verbose = 1
                )

    return clf_GBDT




try:
    pt_min, pt_max = float(sys.argv[1]), float(sys.argv[2])
    n_splits = int(sys.argv[3])
    logging.info("PT min: {} PT max: {} N Splits: {}".format(pt_min,pt_max,n_splits))
    logging.info("\n")
    
    
except:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 BDT_Kfold.py pt_min pt_max n_splits *********")
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
        tmp = pd.read_csv(HOMEPATH + "Notebook/KFold_BDT/" + str(element) + ".csv")
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

BDT_Model_A1 = {
#               "herwig_ang" : 0,
#               "pythia_def" : 0, 
#               "pythia_vin" : 0, 
              "pythia_dip" : 0, 
#               "sherpa_def" : 0,
            }


kf = KFold(n_splits = n_splits)


                         
# skf = StratifiedKFold(n_splits = 5, random_state = 7, shuffle = True) 



for i,(model, trainingdata) in enumerate(zip(BDT_Model_A1, data_train)):

    logging.info("BDT Model: {}  Training Data: {}".format(model, trainingdata))
    
    for model_index, (train_index, val_index) in enumerate(kf.split(data_train[trainingdata]["target"])):
        training_data = data_train[trainingdata].iloc[train_index]
        validation_data = data_train[trainingdata].iloc[val_index]

        try:
            BDT_Model_A1[model] = load_model("./"+str(model)+"_KFold/BDT_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_BDT_"+str(model_index)+ ".h5")
            logging.info(str(model) + " BDT model 1 is loaded!")
            logging.info("######################################################################################")
            logging.info("\n")

        except:
            logging.info("Let's train a BDT model for {}".format(model))
            logging.info("######################################################################################")
            logging.info("\n")
            if os.path.exists("./"+str(model)+"_KFold/BDT_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))) == 0:
                os.mkdir("./"+str(model)+"_KFold/BDT_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max)))
                
                
            ticks_1 = time.time()

            model_BDT = BDT_Model()

            Performance_Frame = {
                            "AUC" : [0],
                            "max_sig" : [0],
                            "r05" : [0],
                            "time": [0]
                            }

            model_BDT.fit(np.asarray(training_data[features]), np.asarray(training_data["target"]))

            dump(model_BDT, "./"+str(model)+"_KFold/BDT_"+str(model)+"_Models_"+str(int(pt_min))+str(int(pt_max))+"/" + str(model) + "_BDT_"+str(model_index)+ ".h5")
            
            
            BDT_Model_A1[model] = model_BDT
            
            prediction_test =  model_BDT.predict(np.asarray(validation_data[features]))
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
                save_to_csvdata = pd.read_csv("./"+str(model)+"_KFold/BDT_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv")
                DATA = pd.concat([save_to_csvdata, dataframe], ignore_index=True, axis=0,join='inner')
                DATA.to_csv("./"+str(model)+"_KFold/BDT_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv", index = 0)
                
            except:
                dataframe.to_csv("./"+str(model)+"_KFold/BDT_"+str(model)+"_to_"+str(trainingdata)+"_Performance_Table_"+str(int(pt_min))+str(int(pt_max))+".csv", index = 0)
                
            
            ticks_2 = time.time()
            ############################################################################################################################################################
            totaltime =  ticks_2 - ticks_1
            logging.info("\n")
            logging.info("\033[3;33mTime consumption : {:.4f} min for BDT\033[0;m".format(totaltime/60.))
            logging.info("######################################################################################")
            logging.info("\n")

