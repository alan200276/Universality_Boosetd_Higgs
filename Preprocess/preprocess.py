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
    
    
except:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 preprocess.py <path-of-file>/XXXX.h5 index file_number *********")
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
raw_weight = []


logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
time.sleep(1)
ticks_1 = time.time()

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
        
        for constituent in Higgs_candidate_tmp[0]:
            raw_weight.append(constituent.weight)
            break

    else:
        double_b_tag.append(0)

        
#     if i == 10:
#         break

logging.info("There are {} events (process_list_clustered).".format(len(process_list_clustered)))
logging.info("There are {} events (Higgs candidate).".format(len(Higgs_candidate)))

logging.info("\n")
ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
logging.info("\n")

logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
ticks_1 = time.time()


    
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
            "raw_weight",
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
    
    var.append(raw_weight[N])
        
    var.append(k)

    dataframe_tmp = pd.DataFrame([var],columns=features)
    dataframe = dataframe.append(dataframe_tmp, ignore_index = True)

#     k += 1

#     if k >= 1000:
#         break


logging.info("There are {} jets.".format(len(dataframe)))
    
if index == 0:
    dataframe.to_csv( path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) +"_" + str(PT_SLICE) + "_" + str(file_number) + ".csv", index = 0)
elif index == 1:
    DATA = pd.concat([save_to_csvdata, dataframe], ignore_index=True, axis=0,join='inner')
    DATA.to_csv(path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) + "_" + str(PT_SLICE) + "_" + str(file_number) + ".csv", index = 0)

ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
logging.info("\n")



"""
Leading Jet Images
"""
make_jet_image(Higgs_candidate,imagespath,GEN,SHO,PRO,PT_SLICE,file_number)
# ###################################################################################
# logging.info("Make Leading Jet Images")
# logging.info("\n")    
# logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
# time.sleep(1)
# ticks_1 = time.time()

# jetimage_list = []

# for N in tqdm(range(len(process_list_clustered))):
    
# #     """
# #     Trigger
# #     """
# #     if ET(process_list_clustered[N][0]) < 420 or process_list_clustered[N][0].mass < 35:
# #         continue
        
#     """
#     >= 2 jets
#     """
#     if len(process_list_clustered[N]) < 2: # at least two jets in this event.
#         continue

#     jet_1_untrimmed = process_list_clustered[N][0] #leading jet's information
#     jet_2_untrimmed = process_list_clustered[N][1] #subleading jet's information
    
# #     """
# #     Basic Selection
# #     """

# #     if (jet_1_untrimmed.pt < 450) or (jet_2_untrimmed.pt < 250) :
# #         continue

# #     if (abs(jet_1_untrimmed.eta) >= 2) or (jet_1_untrimmed.mass <= 50) :
# #         continue

# #     if (abs(jet_2_untrimmed.eta) >= 2) or (jet_2_untrimmed.mass <= 50) :
# #         continue


# #     """
# #     |\Delta\eta| < 1.3
# #     """        
# #     if (abs(jet_1_untrimmed.eta - jet_2_untrimmed.eta) > 1.3) :
# #         continue
        
# #     """
# #     M(jj) > 700 GeV 
# #     """    
# #     if MJJ(jet_1_untrimmed,jet_2_untrimmed) < 700:
# #         continue
        
        

# #     jet = process_list_clustered[N][0] #leading jet's information
#     jet = jet_1_untrimmed
    

#     width,height = 40,40
#     image_0 = np.zeros((width,height)) #Charged pt 
#     image_1 = np.zeros((width,height)) #Neutral pt
#     image_2 = np.zeros((width,height)) #Charged multiplicity
#     isReflection = 1
#     x_hat = np.array([1,0]) 
#     y_hat = np.array([0,1])
    
#     subjets = pyjet.cluster(jet.constituents_array(), R=0.2, p=-1)
#     subjet_array = subjets.inclusive_jets()
    
        
#     if len(subjet_array) > 1:
#             #First, let's find the direction of the second-hardest jet relative to the first-hardest jet
# #             phi_dir = -(dphi(subjet_array[1].phi,subjet_array[0].phi))
# #             eta_dir = -(subjet_array[1].eta - subjet_array[0].eta)
#             phi_dir = -(dphi(subjet_array[1].phi,jet.phi))
#             eta_dir = -(subjet_array[1].eta - jet.eta)
#             #Norm difference:
#             norm_dir = np.linalg.norm([phi_dir,eta_dir])
#             #This is now the y-hat direction. so we can actually find the unit vector:
#             y_hat = np.divide([phi_dir,eta_dir],np.linalg.norm([phi_dir,eta_dir]))
#             #and we can find the x_hat direction as well
#             x_hat = np.array([y_hat[1],-y_hat[0]]) 
    
#     if len(subjet_array) > 2:
# #         phi_dir_3 = -(dphi(subjet_array[2].phi,subjet_array[0].phi))
# #         eta_dir_3 = -(subjet_array[2].eta - subjet_array[0].eta)
#         phi_dir_3 = -(dphi(subjet_array[2].phi,jet.phi))
#         eta_dir_3 = -(subjet_array[2].eta - jet.eta)

#         isReflection = np.cross(np.array([phi_dir,eta_dir,0]),np.array([phi_dir_3,eta_dir_3,0]))[2]
        

#     R = 1.0
#     for constituent in jet:
        
# #         new_coord = [dphi(constituent.phi,jet.phi),constituent.eta-jet.eta]
# #         indxs = [math.floor(width*new_coord[0]/(R*1.5))+width//2, math.floor(height*(new_coord[1])/(R*1.5))+height//2]


#         if (len(subjet_array) == 1):
#             #In the case that the reclustering only found one hard jet (that seems kind of bad, but hey)
#             #no_two = no_two+1
# #             new_coord = [dphi(constituent.phi,subjet_array[0].phi),constituent.eta-subjet_array[0].eta]
#             new_coord = [dphi(constituent.phi, jet.phi),constituent.eta-jet.eta]
#             indxs = [math.floor(width*new_coord[0]/(R*1))+width//2, math.floor(height*(new_coord[1])/(R*1))+height//2]
            
#         else:
#             #Now, we want to express an incoming particle in this new basis:
# #             part_coord = [dphi(constituent.phi,subjet_array[0].phi),constituent.eta-subjet_array[0].eta]
#             part_coord = [dphi(constituent.phi,jet.phi),constituent.eta-jet.eta]
#             new_coord = np.dot(np.array([x_hat,y_hat]),part_coord)
            
#             #put third-leading subjet on the right-hand side
#             if isReflection < 0: 
#                 new_coord = [-new_coord[0],new_coord[1]]
#             elif isReflection > 0:
#                 new_coord = [new_coord[0],new_coord[1]]
#             #Now, we want to cast these new coordinates into our array
#             #(row,column)
# #             indxs = [math.floor(width*new_coord[0]/(R*1.5))+width//2,math.floor(height*(new_coord[1]+norm_dir/1.5)/(R*1.5))+height//2]
# #             indxs = [math.floor(width*new_coord[0]/(R*1.5))+width//2,math.floor(height*new_coord[1]/(R*1.5))+height//2] #(phi,eta) and the leading subjet at the origin
# #             indxs = [math.floor(height*new_coord[1]/(R*1.5))+height//2,math.floor(width*new_coord[0]/(R*1.5))+width//2] #(eta,phi) and the leading subjet at the origin
#             indxs = [math.floor(height*new_coord[1]/(R*1))+height//2,math.floor(width*new_coord[0]/(R*1))+width//2] #(eta,phi) and the leading subjet at the origin

#         if indxs[0] >= width or indxs[1] >= height or indxs[0] <= 0 or indxs[1] <= 0:
#             continue
            
#         phi_index = int(indxs[0]); eta_index = int(indxs[1])

#         #finally, let's fill
#         if constituent.Charge != 0:
#             image_0[phi_index,eta_index] = image_0[phi_index,eta_index] + constituent.pt
#             image_2[phi_index,eta_index] = image_2[phi_index,eta_index] + 1

#         elif constituent.Charge == 0:
#             image_1[phi_index,eta_index] = image_1[phi_index,eta_index] + constituent.pt

            
#     image_0 = np.divide(image_0,np.sum(image_0)) #Charged pt 
#     image_1 = np.divide(image_1,np.sum(image_1)) #Neutral pt 
#     image_2 = np.divide(image_2,np.sum(image_2)) #Charged multiplicity
#     jetimage_list.append(np.array([image_0,image_1,image_2]))
    
    
# jetimage_list = np.array(jetimage_list)


# logging.info("There are {} jet images.".format(len(jetimage_list)))
# logging.info("\n")
# np.savez(imagespath + str(GEN) + "_" + str(SHO) + "_" + str(PRO) +"_"+str(file_number)+"_untrimmed_leading.npz", 
#            jet_images = jetimage_list)

# ticks_2 = time.time()
# totaltime =  ticks_2 - ticks_1
# logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
# logging.info("\n")  




"""
sub-Leading Jet Images
"""


# ###################################################################################
# logging.info("Make sub-Leading Jet Images")
# logging.info("\n")    
# logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
# time.sleep(1)
# ticks_1 = time.time()

# jetimage_list = []

# for N in tqdm(range(len(process_list_clustered))):
    
# #     """
# #     Trigger
# #     """
# #     if ET(process_list_clustered[N][0]) < 420 or process_list_clustered[N][0].mass < 35:
# #         continue
        
#     """
#     >= 2 jets
#     """
#     if len(process_list_clustered[N]) < 2: # at least two jets in this event.
#         continue

#     jet_1_untrimmed = process_list_clustered[N][0] #leading jet's information
#     jet_2_untrimmed = process_list_clustered[N][1] #subleading jet's information
    
# #     """
# #     Basic Selection
# #     """

# #     if (jet_1_untrimmed.pt < 450) or (jet_2_untrimmed.pt < 250) :
# #         continue

# #     if (abs(jet_1_untrimmed.eta) >= 2) or (jet_1_untrimmed.mass <= 50) :
# #         continue

# #     if (abs(jet_2_untrimmed.eta) >= 2) or (jet_2_untrimmed.mass <= 50) :
# #         continue


# #     """
# #     |\Delta\eta| < 1.3
# #     """        
# #     if (abs(jet_1_untrimmed.eta - jet_2_untrimmed.eta) > 1.3) :
# #         continue
        
# #     """
# #     M(jj) > 700 GeV 
# #     """    
# #     if MJJ(jet_1_untrimmed,jet_2_untrimmed) < 700:
# #         continue
        
        

# #     jet = process_list_clustered[N][0] #leading jet's information
#     jet = jet_2_untrimmed
    

#     width,height = 40,40
#     image_0 = np.zeros((width,height)) #Charged pt 
#     image_1 = np.zeros((width,height)) #Neutral pt
#     image_2 = np.zeros((width,height)) #Charged multiplicity
#     isReflection = 1
#     x_hat = np.array([1,0]) 
#     y_hat = np.array([0,1])
    
#     subjets = pyjet.cluster(jet.constituents_array(), R=0.2, p=-1)
#     subjet_array = subjets.inclusive_jets()
    
        
#     if len(subjet_array) > 1:
#             #First, let's find the direction of the second-hardest jet relative to the first-hardest jet
# #             phi_dir = -(dphi(subjet_array[1].phi,subjet_array[0].phi))
# #             eta_dir = -(subjet_array[1].eta - subjet_array[0].eta)
#             phi_dir = -(dphi(subjet_array[1].phi,jet.phi))
#             eta_dir = -(subjet_array[1].eta - jet.eta)
#             #Norm difference:
#             norm_dir = np.linalg.norm([phi_dir,eta_dir])
#             #This is now the y-hat direction. so we can actually find the unit vector:
#             y_hat = np.divide([phi_dir,eta_dir],np.linalg.norm([phi_dir,eta_dir]))
#             #and we can find the x_hat direction as well
#             x_hat = np.array([y_hat[1],-y_hat[0]]) 
    
#     if len(subjet_array) > 2:
# #         phi_dir_3 = -(dphi(subjet_array[2].phi,subjet_array[0].phi))
# #         eta_dir_3 = -(subjet_array[2].eta - subjet_array[0].eta)
#         phi_dir_3 = -(dphi(subjet_array[2].phi,jet.phi))
#         eta_dir_3 = -(subjet_array[2].eta - jet.eta)

#         isReflection = np.cross(np.array([phi_dir,eta_dir,0]),np.array([phi_dir_3,eta_dir_3,0]))[2]
        

#     R = 1.0
#     for constituent in jet:
        
# #         new_coord = [dphi(constituent.phi,jet.phi),constituent.eta-jet.eta]
# #         indxs = [math.floor(width*new_coord[0]/(R*1.5))+width//2, math.floor(height*(new_coord[1])/(R*1.5))+height//2]


#         if (len(subjet_array) == 1):
#             #In the case that the reclustering only found one hard jet (that seems kind of bad, but hey)
#             #no_two = no_two+1
# #             new_coord = [dphi(constituent.phi,subjet_array[0].phi),constituent.eta-subjet_array[0].eta]
#             new_coord = [dphi(constituent.phi, jet.phi),constituent.eta-jet.eta]
#             indxs = [math.floor(width*new_coord[0]/(R*1))+width//2, math.floor(height*(new_coord[1])/(R*1))+height//2]
            
#         else:
#             #Now, we want to express an incoming particle in this new basis:
# #             part_coord = [dphi(constituent.phi,subjet_array[0].phi),constituent.eta-subjet_array[0].eta]
#             part_coord = [dphi(constituent.phi,jet.phi),constituent.eta-jet.eta]
#             new_coord = np.dot(np.array([x_hat,y_hat]),part_coord)
            
#             #put third-leading subjet on the right-hand side
#             if isReflection < 0: 
#                 new_coord = [-new_coord[0],new_coord[1]]
#             elif isReflection > 0:
#                 new_coord = [new_coord[0],new_coord[1]]
#             #Now, we want to cast these new coordinates into our array
#             #(row,column)
# #             indxs = [math.floor(width*new_coord[0]/(R*1.5))+width//2,math.floor(height*(new_coord[1]+norm_dir/1.5)/(R*1.5))+height//2]
# #             indxs = [math.floor(width*new_coord[0]/(R*1.5))+width//2,math.floor(height*new_coord[1]/(R*1.5))+height//2] #(phi,eta) and the leading subjet at the origin
# #             indxs = [math.floor(height*new_coord[1]/(R*1.5))+height//2,math.floor(width*new_coord[0]/(R*1.5))+width//2] #(eta,phi) and the leading subjet at the origin
#             indxs = [math.floor(height*new_coord[1]/(R*1))+height//2,math.floor(width*new_coord[0]/(R*1))+width//2] #(eta,phi) and the leading subjet at the origin

#         if indxs[0] >= width or indxs[1] >= height or indxs[0] <= 0 or indxs[1] <= 0:
#             continue
            
#         phi_index = int(indxs[0]); eta_index = int(indxs[1])

#         #finally, let's fill
#         if constituent.Charge != 0:
#             image_0[phi_index,eta_index] = image_0[phi_index,eta_index] + constituent.pt
#             image_2[phi_index,eta_index] = image_2[phi_index,eta_index] + 1

#         elif constituent.Charge == 0:
#             image_1[phi_index,eta_index] = image_1[phi_index,eta_index] + constituent.pt

            
#     image_0 = np.divide(image_0,np.sum(image_0)) #Charged pt 
#     image_1 = np.divide(image_1,np.sum(image_1)) #Neutral pt 
#     image_2 = np.divide(image_2,np.sum(image_2)) #Charged multiplicity
#     jetimage_list.append(np.array([image_0,image_1,image_2]))
    
    
# jetimage_list = np.array(jetimage_list)


# logging.info("There are {} jet images.".format(len(jetimage_list)))
# logging.info("\n")
# np.savez(imagespath + str(GEN) + "_" + str(SHO) + "_" + str(PRO) + "_"+str(file_number)+"_untrimmed_subleading.npz", 
#            jet_images = jetimage_list)

# ticks_2 = time.time()
# totaltime =  ticks_2 - ticks_1
# logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))


