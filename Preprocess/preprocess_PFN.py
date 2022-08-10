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
    XHH = np.sqrt( (m1-124)**2/(0.1*(m1+1e-5))**2 + (m2-115)**2/(0.1*(m2+1e-5))**2)
    return  XHH
#%%
###################################################################################
"""
Input Check and Setting
"""
###################################################################################
print("Input Check and Setting")

try:
    data_path = str(sys.argv[1])
    
    index = int(sys.argv[2])
    
    file_number = int(sys.argv[3])
    
    findlast=  []
    for m in re.finditer("/",data_path):
        findlast.append(m.start())

    l = findlast[-1] 

    file_name = data_path[l:]
    print(file_name)


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

    print("File is loaded!")
    print("Generator (GEN) : {}".format(GEN))
    print("Showering (SHO) : {}".format(SHO))
    print("Process (PRO) : {}".format(PRO))
    print("Pt Slices (PT_SLICE) : {}".format(PT_SLICE))
    
    
except:
    print("********* Please Check Input Argunment *********")
    print("********* Usage: python3 preprocess.py <path-of-file>/XXXX.h5 index file_number *********")
    sys.exit(1)


#%%
###################################################################################
"""
Read Data and Jet Clustering 
"""
###################################################################################

print("Read Data and Jet Clustering ")
print("\n")


hf_read = h5py.File(data_path, 'r')

process_list_clustered = []

double_b_tag, Higgs_candidate = [], []

print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
#%%
print("Jet Clustering")
print("=====START=====")
t1_time = time.time()
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

    else:
        double_b_tag.append(0)

    # if i == 1000:
    #     break


t2_time = time.time()
print("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2_time-t1_time)/60.))
print("=====Finish=====")
print("\n")
        
        

print("There are {} events (process_list_clustered).".format(len(process_list_clustered)))
print("There are {} events (Higgs candidate).".format(len(Higgs_candidate)))

#%%


print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))


print("\n")
print("For PFN Data Frame")
print("=====START=====")
t1_time = time.time()
time.sleep(1)
    
###################################################################################
"""
Read Data
"""
###################################################################################
print("Read Data")
print("\n")

HOMEPATH = "/home/u5/Universality_Boosetd_Higgs/"
path =  HOMEPATH + "Data_PFN/"

PFN_data_format = np.zeros((len(Higgs_candidate),300,8))
# (N jets, 300 constituent's space, (cons.eta, cons.phi, cons.pt, cons.e, hj.eta, hj.phi, hj.pt, hj.e))

for i, hj in enumerate(Higgs_candidate):
    for j, cons in enumerate(hj):
        PFN_data_format[i,j,0] = cons.eta
        PFN_data_format[i,j,1] = cons.phi
        PFN_data_format[i,j,2] = cons.pt
        PFN_data_format[i,j,3] = cons.e
        PFN_data_format[i,j,4] = hj.eta
        PFN_data_format[i,j,5] = hj.phi
        PFN_data_format[i,j,6] = hj.pt
        PFN_data_format[i,j,7] = hj.e

np.savez_compressed(path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) + "_" + str(PT_SLICE) + "_" + str(file_number), PFN_data_format = PFN_data_format)


t2_time = time.time()
print("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2_time-t1_time)/60.))
print("=====Finish=====")
print("\n")


# %%
