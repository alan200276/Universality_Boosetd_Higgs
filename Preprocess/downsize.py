#!/bin/python3

#%%
import uproot
import pyjet
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import importlib
import time
import re

from BranchClass import *

import Event_List 
import jet_trimming 
import JSS 
from tqdm import tqdm
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)



#%%

# Returns the difference in phi between phi, and phi_center
# as a float between (-PI, PI)
def dphi(phi,phi_c):

    dphi_temp = phi - phi_c
    while dphi_temp > np.pi:
        dphi_temp = dphi_temp - 2*np.pi
    while dphi_temp < -np.pi:
        dphi_temp = dphi_temp + 2*np.pi
    return (dphi_temp)


logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
ticks_1 = time.time()
###################################################################################
"""
Input Check and Setting
"""
###################################################################################
logging.info("Input Check and Setting")
logging.info("\n")
try:
    data_path = str(sys.argv[1])
    
    file_number = int(sys.argv[2])
    
    save_path = str(sys.argv[3])
    
    MCData = uproot.open(data_path)["Delphes;1"]
    
    findlast=  []
    for m in re.finditer("/",data_path):
        findlast.append(m.start())

    l = findlast[-1] 

    file_name = data_path[l:]
    logging.info(file_name)


    if "pythia" in file_name:
        GEN = str("pythia")
    elif "sherpa" in file_name: 
        GEN = str("sherpa")
    elif "herwig" in file_name:
        GEN = str("herwig")

    if "def" in file_name:
        SHO = str("def")
    elif "dip" in file_name: 
        SHO = str("dip")
    elif "ang" in file_name:
        SHO = str("ang")
    elif "vin" in file_name: 
        SHO = str("vin")
    elif "lun" in file_name: 
        SHO = str("lun")

    if "ggHj" in file_name:
        PRO = str("H")
    elif "ppjj" in file_name:
        PRO = str("QCD")


    if "250_500" in file_name:
        PT_SLICE = str("250_500")
    elif "450_700" in file_name:
        PT_SLICE = str("450_700")
    elif "650_900" in file_name:
        PT_SLICE = str("650_900")
    elif "850_1100" in file_name:
        PT_SLICE = str("850_1100")
    else:
        PT_SLICE = str("250_550")

    logging.info("File is loaded!")
    logging.info("Generator (GEN) : {}".format(GEN))
    logging.info("Showering (SHO) : {}".format(SHO))
    logging.info("Process (PRO) : {}".format(PRO))
    logging.info("Pt Slices (PT_SLICE) : {}".format(PT_SLICE))
    
    
except:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 downsize.py <path-of-file>/XXXX.root file_number save_path *********")
    sys.exit(1)
    

    
###################################################################################
"""
Read Data and Jet Clustering 
"""
###################################################################################

logging.info("Read Data and Downsize into h5 format")
logging.info("\n")

# HOMEPATH = "/home/u5/Universality_Boosetd_Higgs/"
# path =  HOMEPATH + "DownsizeData/"
path = save_path + "/"

eventpath = path + "EventList_" + str(PRO)+"_"+str(GEN)+"_"+str(SHO)+"_"+str(PT_SLICE)+"_"+str(file_number)+".h5"
GenParticle = BranchGenParticles(MCData)
Jet10 = BranchParticleFlowJet10(MCData)
EventWeight = Event_Weight(MCData)
EventList = Event_List.Event_List(GenParticle, Jet10, EventWeight, path=eventpath)



logging.info("\n")
ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))