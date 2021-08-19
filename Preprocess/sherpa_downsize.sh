#/bin/bash

PROCESS="Sherpa"
HOMEPATH="/home/alan/ML_Analysis/Universality_DNN_DiHiggs"
datapath="/home/u5/Universality_DiHiggs/"$PROCESS
outpath_H="Out/"$PROCESS"_H"
outpath_QCD="Out/"$PROCESS"_QCD"
process="sherpa_def"

date

echo "Start Running"


date +"%Y %b %m"


i=1
while [ $i != 2 ]
do 


    echo "H $PROCESS"
    nohup python3 $HOMEPATH/Preprocess/downsize_separate.py $datapath"_ppHhh"/ppHhh_"$process"_"$i".root 0 $i > $HOMEPATH/$outpath_H/downsize_sherpa_def_"$process"_"$i".log &

   
    echo "QCD $PROCESS"
    nohup python3 $HOMEPATH/Preprocess/downsize_separate.py $datapath"_ppbbbb"/ppbbbb_"$process"_"$i".root 0 $i > $HOMEPATH/$outpath_QCD/downsize_sherpa_def_"$process"_"$i".log &





    date +"%Y %b %m"
    date +"%r"
    i=$(($i+1))

done



echo "Finish"

date