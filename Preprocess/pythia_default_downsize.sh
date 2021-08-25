#/bin/bash

PROCESS="Pythia_default"
HOMEPATH="/home/alan/ML_Analysis/Universality_Boosetd_Higgs"
datapath="/home/u5/Universality_Boosetd_Higgs/"$PROCESS
outpath_H="Log/"$PROCESS"_H"
outpath_QCD="Log/"$PROCESS"_QCD"
process="pythia_def"

date

echo "Start Running"


date +"%Y %b %m"


# Iterate 10 LHE File for each Pt Slice
i=2
while [ $i != 3 ]
do
#========================================================================================
    

        # Iterate Pt Slices
        ###################################################################
        for pt_range in "250_500" "450_700" "650_900" "850_1100" 
        do
       
            echo "pt_range=$pt_range"
            echo "i =  $i "
            echo "============================================"

            echo "H $PROCESS"
            nohup python3 $HOMEPATH/Preprocess/downsize.py $datapath/ggHj_"$process"_"$pt_range"_"$i".root $i "$datapath" > $HOMEPATH/$outpath_H/downsize_ggHj_"$process"_"$pt_range"_"$i".log &


            echo "QCD $PROCESS"
            nohup python3 $HOMEPATH/Preprocess/downsize.py $datapath/ppjj_"$process"_"$pt_range"_"$i".root $i $datapath > $HOMEPATH/$outpath_QCD/downsize_ppjj_"$process"_"$pt_range"_"$i".log &



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
