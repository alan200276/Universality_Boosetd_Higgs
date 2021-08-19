#/bin/bash

PROCESS="Herwig_angular"
HOMEPATH="/home/alan/ML_Analysis/Universality_Boosetd_Higgs"
datapath="/home/u5/Universality_Boosetd_Higgs/"$PROCESS
outpath_H="Log/"$PROCESS"_H"
outpath_QCD="Log/"$PROCESS"_QCD"
process="herwig_ang"

date

echo "Start Running"


date +"%Y %b %m"



# Iterate 10 LHE File for each Pt Slice
i=1
while [ $i != 2 ]
do
#========================================================================================
    

        # Iterate Pt Slices
        ###################################################################
        for pt_range in "250_500" "450_700" "650_900" "850_1100" 
        do
        
        nohup python3 $HOMEPATH/Preprocess/preprocess.py $datapath/EventList_H_"$process"_"$pt_range"_"$i".h5 0 $i > $HOMEPATH/$outpath_H/preprocess_ggHj_"$process"_"$pt_range"_"$i".log  &


        nohup python3 $HOMEPATH/Preprocess/preprocess.py $datapath/EventList_QCD_"$process"_"$pt_range"_"$i".h5 0 $i > $HOMEPATH/$outpath_QCD/preprocess_ppjj_"$process"_"$pt_range"_"$i".log  &
    

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

