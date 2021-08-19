#!/bin/python3
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


# Returns the difference in phi between phi, and phi_center
# as a float between (-PI, PI)
def dphi(phi,phi_c):

    dphi_temp = phi - phi_c
    while dphi_temp > np.pi:
        dphi_temp = dphi_temp - 2*np.pi
    while dphi_temp < -np.pi:
        dphi_temp = dphi_temp + 2*np.pi
    return (dphi_temp)

def make_jet_image(Higgs_candidate=[],imagespath="path",GEN="GEN",SHO="SHO",PRO="PRO", PT_SLICE="PT_SLICE",file_number=0):
    ###################################################################################
    logging.info("Make Jet Images")
    logging.info("\n")    
    logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    time.sleep(1)
    ticks_1 = time.time()

    jetimage_list = []

    for N in tqdm(range(len(Higgs_candidate))):


        """
        >= 1 jet
        """
        if len(Higgs_candidate[N]) < 1: # at least two jets in this event.
            continue

        jet_1_untrimmed = Higgs_candidate[N] #leading jet's information


        jet = jet_1_untrimmed


        width,height = 40,40
        image_0 = np.zeros((width,height)) #Charged pt 
        image_1 = np.zeros((width,height)) #Neutral pt
        image_2 = np.zeros((width,height)) #Charged multiplicity
        isReflection = 1
        x_hat = np.array([1,0]) 
        y_hat = np.array([0,1])

        subjets = pyjet.cluster(jet.constituents_array(), R=0.2, p=-1)
        subjet_array = subjets.inclusive_jets()


        if len(subjet_array) > 1:
                #First, let's find the direction of the second-hardest jet relative to the first-hardest jet
    #             phi_dir = -(dphi(subjet_array[1].phi,subjet_array[0].phi))
    #             eta_dir = -(subjet_array[1].eta - subjet_array[0].eta)
                phi_dir = -(dphi(subjet_array[1].phi,jet.phi))
                eta_dir = -(subjet_array[1].eta - jet.eta)
                #Norm difference:
                norm_dir = np.linalg.norm([phi_dir,eta_dir])
                #This is now the y-hat direction. so we can actually find the unit vector:
                y_hat = np.divide([phi_dir,eta_dir],np.linalg.norm([phi_dir,eta_dir]))
                #and we can find the x_hat direction as well
                x_hat = np.array([y_hat[1],-y_hat[0]]) 

        if len(subjet_array) > 2:
    #         phi_dir_3 = -(dphi(subjet_array[2].phi,subjet_array[0].phi))
    #         eta_dir_3 = -(subjet_array[2].eta - subjet_array[0].eta)
            phi_dir_3 = -(dphi(subjet_array[2].phi,jet.phi))
            eta_dir_3 = -(subjet_array[2].eta - jet.eta)

            isReflection = np.cross(np.array([phi_dir,eta_dir,0]),np.array([phi_dir_3,eta_dir_3,0]))[2]


        R = 1.0
        for constituent in jet:

    #         new_coord = [dphi(constituent.phi,jet.phi),constituent.eta-jet.eta]
    #         indxs = [math.floor(width*new_coord[0]/(R*1.5))+width//2, math.floor(height*(new_coord[1])/(R*1.5))+height//2]


            if (len(subjet_array) == 1):
                #In the case that the reclustering only found one hard jet (that seems kind of bad, but hey)
                #no_two = no_two+1
    #             new_coord = [dphi(constituent.phi,subjet_array[0].phi),constituent.eta-subjet_array[0].eta]
                new_coord = [dphi(constituent.phi, jet.phi),constituent.eta-jet.eta]
                indxs = [math.floor(width*new_coord[0]/(R*1))+width//2, math.floor(height*(new_coord[1])/(R*1))+height//2]

            else:
                #Now, we want to express an incoming particle in this new basis:
    #             part_coord = [dphi(constituent.phi,subjet_array[0].phi),constituent.eta-subjet_array[0].eta]
                part_coord = [dphi(constituent.phi,jet.phi),constituent.eta-jet.eta]
                new_coord = np.dot(np.array([x_hat,y_hat]),part_coord)

                #put third-leading subjet on the right-hand side
                if isReflection < 0: 
                    new_coord = [-new_coord[0],new_coord[1]]
                elif isReflection > 0:
                    new_coord = [new_coord[0],new_coord[1]]
                #Now, we want to cast these new coordinates into our array
                #(row,column)
    #             indxs = [math.floor(width*new_coord[0]/(R*1.5))+width//2,math.floor(height*(new_coord[1]+norm_dir/1.5)/(R*1.5))+height//2]
    #             indxs = [math.floor(width*new_coord[0]/(R*1.5))+width//2,math.floor(height*new_coord[1]/(R*1.5))+height//2] #(phi,eta) and the leading subjet at the origin
    #             indxs = [math.floor(height*new_coord[1]/(R*1.5))+height//2,math.floor(width*new_coord[0]/(R*1.5))+width//2] #(eta,phi) and the leading subjet at the origin
                indxs = [math.floor(height*new_coord[1]/(R*1))+height//2,math.floor(width*new_coord[0]/(R*1))+width//2] #(eta,phi) and the leading subjet at the origin

            if indxs[0] >= width or indxs[1] >= height or indxs[0] <= 0 or indxs[1] <= 0:
                continue

            phi_index = int(indxs[0]); eta_index = int(indxs[1])

            #finally, let's fill
            if constituent.Charge != 0:
                image_0[phi_index,eta_index] = image_0[phi_index,eta_index] + constituent.pt
                image_2[phi_index,eta_index] = image_2[phi_index,eta_index] + 1

            elif constituent.Charge == 0:
                image_1[phi_index,eta_index] = image_1[phi_index,eta_index] + constituent.pt


        image_0 = np.divide(image_0,np.sum(image_0)) #Charged pt 
        image_1 = np.divide(image_1,np.sum(image_1)) #Neutral pt 
        image_2 = np.divide(image_2,np.sum(image_2)) #Charged multiplicity
        jetimage_list.append(np.array([image_0,image_1,image_2]))


    jetimage_list = np.array(jetimage_list)


    logging.info("There are {} jet images.".format(len(jetimage_list)))
    logging.info("\n")
    np.savez(imagespath + str(GEN) + "_" + str(SHO) + "_" + str(PRO) +"_"+str(file_number)+"_untrimmed.npz", 
               jet_images = jetimage_list)

    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
    logging.info("\n")  

