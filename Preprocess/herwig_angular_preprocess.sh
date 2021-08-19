#/bin/bash

# PROCESS="Herwig_angular"
PROCESS="Herwig_angular"
HOMEPATH="/home/alan/ML_Analysis/Universality_DNN_DiHiggs"
datapath="/home/u5/Universality_DiHiggs/"$PROCESS
outpath_wz="Out/"$PROCESS"_H"
outpath_jj="Out/"$PROCESS"_QCD"
process="herwig_ang"

date

echo "Start Running"


date +"%Y %b %m"


i=1
while [ $i != 11 ]
do 

#     nohup python3 $HOMEPATH/Preprocess/preprocess.py $datapath/EventList_H_herwig_ang_"$i".h5 0 $i > $HOMEPATH/$outpath_wz/preprocess_ppHhh_"$process"_"$i".log  &
    
#     nohup python3 $HOMEPATH/Preprocess/preprocess.py $datapath/EventList_QCD_herwig_ang_"$i".h5 0 $i > $HOMEPATH/$outpath_jj/preprocess_ppbbbb_"$process"_"$i".log  &

    nohup python3 $HOMEPATH/Preprocess/preprocess.py $datapath/EventList_QCD_herwig_ang_"$i".h5 0 $i > $HOMEPATH/$outpath_jj/preprocess_ppjjjj_"$process"_"$i"_high_pt_500.log  &


    date +"%Y %b %m"
    date +"%r"
    i=$(($i+1))

done



echo "Finish"

date
