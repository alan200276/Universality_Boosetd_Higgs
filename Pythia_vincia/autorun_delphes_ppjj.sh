#!/bin/bash

echo "Start Running"

outpath="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Log/Pythia_vincia_QCD"
hepmcpath="/home/u5/Universality_Boosetd_Higgs/Pythia_vincia"
savepath="/home/u5/Universality_Boosetd_Higgs/Pythia_vincia"
rootfilename="ppjj_pythia_vin"
hepmcfilename="ppjj_pythia_vin"




# Iterate 10 LHE File for each Pt Slice
i=1
while [ $i != 2 ]
do
#========================================================================================
    

       date +"%Y %b %m"
       date +"%r"
       echo "ggHj"
       
       
        # Iterate Pt Slices
        ###################################################################
        for pt_range in "250_500" "450_700" "650_900" "850_1100" 
        do
       
            echo "pt_range=$pt_range"
            echo "i =  $i "
            echo "============================================"
            cd /root/Delphes-3.4.2

            nohup ./DelphesHepMC /home/alan/ML_Analysis/Universality_Boosetd_Higgs/Cards/delphes_card_HLLHC.tcl $savepath/"$rootfilename"_"$pt_range"_$i.root $hepmcpath/"$hepmcfilename"_"$pt_range"_"$i".hepmc > $outpath/"$rootfilename"_"$pt_range"_"$i"_log.txt &


            date +"%Y %b %m"
            date +"%r"
            echo "============================================"
            
        done  
        ################################################################### 
   
   
   
        i=$(($i+1))

done
#========================================================================================



echo "Finish"

date
