#!/bin/python3

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
from tqdm import tqdm
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

from BranchClass import *

import Event_List 
import jet_trimming 
import JSS 
from make_jet_image import make_jet_image 

import h5py

# Returns the difference in phi between phi, and phi_center
# as a float between (-PI, PI)
def dphi(phi,phi_c):

    dphi_temp = phi - phi_c
    while dphi_temp > np.pi:
        dphi_temp = dphi_temp - 2*np.pi
    while dphi_temp < -np.pi:
        dphi_temp = dphi_temp + 2*np.pi
    return (dphi_temp)

def MJJ(j1,j2):
    pt1, eta1, phi1, m1 = j1.pt,j1.eta,j1.phi,j1.mass
    pt2, eta2, phi2, m2 = j2.pt,j2.eta,j2.phi,j2.mass
    
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    
    return np.sqrt((e1+e2)**2-(px1+px2)**2-(py1+py2)**2-(pz1+pz2)**2)


def ET(jet):
    pt = jet.pt
    m = jet.mass
    ET = np.sqrt(m**2 + pt**2)
    return  ET

def XHH(jet1, jet2):
    m1, m2 = jet1.mass, jet2.mass
    XHH = np.sqrt( (m1-124)**2/(0.1*(m1+1e-5)) + (m2-115)**2/(0.1*(m2+1e-5)))
    return  XHH

###################################################################################
"""
Input Check and Setting
"""
###################################################################################
logging.info("Input Check and Setting")

try:
    data_path = str(sys.argv[1])
    
    index = int(sys.argv[2])
    
    file_number = int(sys.argv[3])
    
    lhe_process_path = str(sys.argv[4])
    
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

    if "H" in file_name:
        PRO = str("H")
    elif "QCD" in file_name:
        PRO = str("QCD")


    if "250_500" in file_name:
        PT_SLICE = str("250_500")
    elif "450_700" in file_name:
        PT_SLICE = str("450_700")
    elif "650_900" in file_name:
        PT_SLICE = str("650_900")
    elif "850_1100" in file_name:
        PT_SLICE = str("850_1100")

    logging.info("File is loaded!")
    logging.info("Generator (GEN) : {}".format(GEN))
    logging.info("Showering (SHO) : {}".format(SHO))
    logging.info("Process (PRO) : {}".format(PRO))
    logging.info("Pt Slices (PT_SLICE) : {}".format(PT_SLICE))
    logging.info("lhe_process_path : {}".format(lhe_process_path))
    
    
except:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 preprocess.py <path-of-file>/XXXX.h5 index file_number lhe_process_path *********")
    sys.exit(1)


    
###################################################################################
"""
Read Data and Jet Clustering 
"""
###################################################################################

logging.info("Read Data and Jet Clustering ")
logging.info("\n")


hf_read = h5py.File(data_path, 'r')

process_list_clustered = []

double_b_tag, Higgs_candidate = [], []
lhe_weight = []
weight = []

logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

logging.info("Get Event Weight For LHE File")
logging.info("=====START=====")
t1 = time.time()
time.sleep(1)

with open(lhe_process_path, 'r') as f:
    for i, line in tqdm(enumerate(f)):

        if len(line.split()) != 6:
                continue

        try:
            if float(line.split()[0]) == 4 or float(line.split()[0]) == 5:
                lhe_weight.append(float(line.split()[2])/100000)
        except:
            continue
            
logging.info("# of events: {}, Average Weight: {} pb".format(len(lhe_weight),sum(lhe_weight)))
logging.info("\n")         

t2 = time.time()
logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
logging.info("=====Finish=====")
logging.info("\n")


logging.info("Jet Clustering")
logging.info("=====START=====")
t1 = time.time()
time.sleep(1)


for i in tqdm(range(len(hf_read["GenParticle"]))):

    """
    Jet clustering 
    Fat jet: R = 1
    Anti-kt
    """
    to_cluster = np.core.records.fromarrays(hf_read.get('GenParticle/dataset_'+ str(i))[:9], 
                                            names="pt, eta, phi, mass, PID, Status, Charge, B_tag, weight",
                                            formats = "f8, f8, f8, f8, f8, f8, f8, f8, f8"
                                           )
    pt_min = 25
    sequence_cluster = pyjet.cluster(to_cluster, R = 1, p = -1) # p = -1: anti-kt , 0: Cambridge-Aachen(C/A), 1: kt
    jets_cluster = sequence_cluster.inclusive_jets(pt_min)
    process_list_clustered.append(jets_cluster)


    """
    Higgs Candidate: have one double-$b$ tagged jet associated with $H$ candidate
    """
    Higgs_candidate_tmp = []
    if len(jets_cluster) >=1:
        for jet in jets_cluster:
            B_tag = 0
            for constituent in jet:
                if constituent.B_tag == 1:
                    B_tag += 1
            if B_tag >= 2:
                Higgs_candidate_tmp.append(jet)

    if len(Higgs_candidate_tmp) >= 1:
        Higgs_candidate.append(Higgs_candidate_tmp[0])
        double_b_tag.append(1)
        weight.append(lhe_weight[i])
        
#         for constituent in Higgs_candidate_tmp[0]:
# #             raw_weight.append(constituent.weight)
#             break

    else:
        double_b_tag.append(0)

t2 = time.time()
logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
logging.info("=====Finish=====")
logging.info("\n")
        
        
#     if i == 10:
#         break

logging.info("There are {} events (process_list_clustered).".format(len(process_list_clustered)))
logging.info("There are {} events (Higgs candidate).".format(len(Higgs_candidate)))




logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

logging.info("\n")
logging.info("For Pandas Data Frame")
logging.info("=====START=====")
t1 = time.time()
time.sleep(1)
    
###################################################################################
"""
Read Data
"""
###################################################################################
logging.info("Read Data")
logging.info("\n")

M_J = []
M_J_trimmed = []
T21 = []
T21_trimmed = []
D21, D22, C21, C22 = [], [], [], []
D21_trimmed, D22_trimmed, C21_trimmed, C22_trimmed = [], [], [], []


###################################################################################
"""
Create Pandas DataFrame
"""
###################################################################################

logging.info("Create Pandas DataFrame")
logging.info("\n")

HOMEPATH = "/home/u5/Universality_Boosetd_Higgs/"
path =  HOMEPATH + "Data_High_Level_Features/"
imagespath =  HOMEPATH + "Jet_Images_untrimmed/"


for j, filename in enumerate(os.listdir(path)):
    if filename == str(GEN) + "_" + str(SHO) + "_" + str(PRO) + "_" + str(PT_SLICE) + "_" + str(file_number) + ".csv":
        index = 1 

if index == 0:
    dataframe = pd.DataFrame()
if index == 1:
    dataframe = pd.DataFrame()
    save_to_csvdata = pd.read_csv(path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) + "_" + str(PT_SLICE) + "_" + str(file_number) + ".csv")

###################################################################################
    
logging.info("Selection and Trimming")
logging.info("\n")    
logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
time.sleep(1)
ticks_1 = time.time()
    
features = ["GEN","SHO","PRO",
            "MJ_0","PTJ_0","eta_0","phi_0",
            "t21_0","D21_0","D22_0","C21_0","C22_0",
            "MJ","PTJ","eta","phi",
            "t21","D21","D22","C21","C22",
            "weight",
            "eventindex"
           ]

untrimmed_jet = []
trimmed_jet = []
k = 0
for N in tqdm(range(len(Higgs_candidate))):
    
#     """
#     Trigger
#     """
#     if ET(process_list_clustered[N][0]) < 420 or process_list_clustered[N][0].mass < 35:
#         continue
        
    """
    >= 1 jet
    """
    if len(Higgs_candidate[N]) < 1:
        continue

    jet_1_untrimmed = Higgs_candidate[N] #leading jet's information
    
    
        
    var = []

    var.append(GEN)
    var.append(SHO)
    var.append(PRO)


   
    t1 = JSS.tn(jet_1_untrimmed, n=1)
    t2 = JSS.tn(jet_1_untrimmed, n=2)
    t21_untrimmed = t2 / t1 if t1 > 0.0 else 0.0

    ee2 = JSS.CalcEECorr(jet_1_untrimmed, n=2, beta=1.0)
    ee3 = JSS.CalcEECorr(jet_1_untrimmed, n=3, beta=1.0)
    d21_untrimmed = ee3/(ee2**3) if ee2>0 else 0
    d22_untrimmed = ee3**2/((ee2**2)**3) if ee2>0 else 0
    c21_untrimmed = ee3/(ee2**2) if ee2>0 else 0
    c22_untrimmed = ee3**2/((ee2**2)**2) if ee2>0 else 0 

    var.append(jet_1_untrimmed.mass)
    var.append(jet_1_untrimmed.pt)
    var.append(jet_1_untrimmed.eta)
    var.append(jet_1_untrimmed.phi)
    var.append(t21_untrimmed)
    var.append(d21_untrimmed)
    var.append(d22_untrimmed)
    var.append(c21_untrimmed)
    var.append(c22_untrimmed)


    """
    Jet Trimming
    """
    jet_1_trimmed = jet_trimming.jet_trim(jet_1_untrimmed)[0]   #trimming jet's information



    t1 = JSS.tn(jet_1_trimmed, n=1)
    t2 = JSS.tn(jet_1_trimmed, n=2)
    t21_trimmed = t2 / t1 if t1 > 0.0 else 0.0

    ee2 = JSS.CalcEECorr(jet_1_trimmed, n=2, beta=1.0)
    ee3 = JSS.CalcEECorr(jet_1_trimmed, n=3, beta=1.0)
    d21_trimmed = ee3/(ee2**3) if ee2>0 else 0
    d22_trimmed = ee3**2/((ee2**2)**3) if ee2>0 else 0
    c21_trimmed = ee3/(ee2**2) if ee2>0 else 0
    c22_trimmed = ee3**2/((ee2**2)**2) if ee2>0 else 0 

    var.append(jet_1_trimmed.mass)
    var.append(jet_1_trimmed.pt)
    var.append(jet_1_trimmed.eta)
    var.append(jet_1_trimmed.phi)
    var.append(t21_trimmed)
    var.append(d21_trimmed)
    var.append(d22_trimmed)
    var.append(c21_trimmed)
    var.append(c22_trimmed)
    
    var.append(weight[N])
        
    var.append(k)

    dataframe_tmp = pd.DataFrame([var],columns=features)
    dataframe = dataframe.append(dataframe_tmp, ignore_index = True)

    k += 1

#     if k >= 1000:
#         break


logging.info("There are {} jets.".format(len(dataframe)))
    
if index == 0:
    dataframe.to_csv( path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) +"_" + str(PT_SLICE) + "_" + str(file_number) + ".csv", index = 0)
elif index == 1:
    DATA = pd.concat([save_to_csvdata, dataframe], ignore_index=True, axis=0,join='inner')
    DATA.to_csv(path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) + "_" + str(PT_SLICE) + "_" + str(file_number) + ".csv", index = 0)

    

t2 = time.time()
logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
logging.info("=====Finish=====")
logging.info("\n")





"""
Leading Jet Images
"""

logging.info("\n")
logging.info("Leading Jet Images")
logging.info("=====START=====")
t1 = time.time()
time.sleep(1)

make_jet_image(Higgs_candidate,imagespath,GEN,SHO,PRO,PT_SLICE,file_number)


t2 = time.time()
logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
logging.info("=====Finish=====")
logging.info("\n")

