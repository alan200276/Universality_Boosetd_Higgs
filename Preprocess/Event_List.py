#/bin/python3

import time
import pyjet
import numpy as np
import h5py
import importlib
import logging

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)



from BranchClass import *
from tqdm import tqdm


def digit(loc,pid):
    #     //  PID digits (base 10) are: n nr nl nq1 nq2 nq3 nj
    #     //   nj = 1, nq3=2 , nq2=3, nq1, nl, nr, n, n8, n9, n10 
    #     //  the location enum provides a convenient index into the PID
    numerator = 10**(loc-1)
    
    return int((abs(pid)/numerator)%10)

def hasBottom(pid):
    # get B hadron
    # PID for B hadron are 5XX, 5XXX
    # https://gitlab.com/hepcedar/rivet/-/blob/release-3-1-x/analyses/pluginCMS/CMS_2015_I1370682.cc#L390
    # https://rivet.hepforge.org/code/2.1.0/a00827.html#ad4c917595339ea52152c2950ce1225e7
    # https://pdg.lbl.gov/2019/reviews/rpp2019-rev-monte-carlo-numbering.pdf
    if( digit(2,pid) == 5 or digit(3,pid) == 5 or digit(4,pid) == 5 ):
        return True
    else:
        return False


def Event_List(GenParticle, Jet, EventWeight, path="./data_gzip.h5"):
    logging.info("Make Event List")
    logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    time.sleep(1)
    ticks_1 = time.time()
    
    k = 0
    bbbb_list = []
    event_list = []
    Mbbbb = []
    bhadron_list = []
    
    hf = h5py.File(path, 'w')
    for N in tqdm(range(GenParticle.length)):

        event_list_tmp = []
        
        event_list_tmp.append([GenParticle.PT_At(N),  GenParticle.Eta_At(N), \
                               GenParticle.Phi_At(N), GenParticle.Mass_At(N), \
                               GenParticle.PID_At(N), GenParticle.Status_At(N),\
                               GenParticle.Charge_At(N),\
                               np.full(len(GenParticle.PT_At(N)),0), #for B hadron tag
                               np.full(len(GenParticle.PT_At(N)), EventWeight.Event_Weight_At(N)[0]),
                               GenParticle.M1_At(N), GenParticle.M2_At(N),\
                               GenParticle.D1_At(N), GenParticle.D2_At(N),\
                               

                              ])
        
        event_list_tmp = np.array(event_list_tmp)
        
    
        """
        Find All B Hadrons
        """
        unstable_hadron = event_list_tmp[0][:,np.abs(event_list_tmp[0][5,:])==2]
        bhadron_index = np.where(np.vectorize(hasBottom)(unstable_hadron[4,:])==True)[0] 
        bhadron = unstable_hadron[:,bhadron_index]
        bhadron_list_tmp = []
        
        """
        Find B Hadrons before decay
        """
        for i in range(len(bhadron[0])):   
            if hasBottom(GenParticle.PID_At(N)[int(bhadron[11][i])]) == False and hasBottom(GenParticle.PID_At(N)[int(bhadron[12][i])]) == False:
                bhadron_list_tmp.append(bhadron[:,i])
        
        """
        Ghost Association Method: create ghost-associated B Hadrons
        """
        Ghostparam = 1E-20
        for i, element in enumerate(bhadron_list_tmp):
            bhadron_list_tmp[i][0] = element[0]*Ghostparam  # PT*Ghostparam
            bhadron_list_tmp[i][3] = element[3]*Ghostparam  # Mass*Ghostparam
            bhadron_list_tmp[i][7] = 1                      #B Hadron tag 
        
        """
        Pick stable final state particel (status = 1) and filter ou neutrinos (|PID| = 12, 14, 16)
        """
        event_list_tmp = event_list_tmp[0][:,event_list_tmp[0][5,:]==1] 
        event_list_tmp = event_list_tmp[:,np.abs(event_list_tmp[4,:])!=12]
        event_list_tmp = event_list_tmp[:,np.abs(event_list_tmp[4,:])!=14]
        event_list_tmp = event_list_tmp[:,np.abs(event_list_tmp[4,:])!=16]
        
        """
        Ghost Association Method: add ghost-associated B Hadrons
        """
        if len(bhadron_list_tmp) != 0:
            event_list_tmp = np.concatenate((np.array(event_list_tmp).transpose(),bhadron_list_tmp)).transpose()
            
        else:
            event_list_tmp = np.array(event_list_tmp)
        
        event_list.append(event_list_tmp)
        
        hf.create_dataset("GenParticle/dataset_" + str(N), data=event_list_tmp, compression="gzip", compression_opts=5)
    
#         k += 1
        
#         if k > 10000:
#             break
        
        
        
    k = 0
    jet_list = []
    
    for N in tqdm(range(Jet.length)):
        
        jet_list_tmp = []
        jet_list_tmp_tmp = []


        jet_list_tmp.append([Jet.PT_At(N),  Jet.Eta_At(N), \
                               Jet.Phi_At(N), Jet.Mass_At(N), \
                               Jet.Charge_At(N), Jet.BTag_At(N),\
                               np.full(len(Jet.PT_At(N)), EventWeight.Event_Weight_At(N)[0]),
                              ])


        
        jet_list_tmp = np.array(jet_list_tmp)

        jet_list_tmp = jet_list_tmp[0]#[:,jet_list_tmp[0][5,:]==1]
#         jet_list_tmp = jet_list_tmp[:,np.abs(jet_list_tmp[4,:])!=12]
        
        jet_list.append(jet_list_tmp)
        
        hf.create_dataset("Jet/dataset_" + str(N), data=jet_list_tmp, compression="gzip", compression_opts=5)
        
#         k += 1
        
#         if k > 10000:
#             break
        
        
    hf.close()

    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
    
    return event_list