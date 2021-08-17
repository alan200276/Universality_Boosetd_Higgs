#!/bin/bash

cardpath="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Cards"

outpath="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Log"

mcdatapath="/home/u5"



echo "Start Running"

i=1
while [ $i != 2 ]
do
   echo i=$i

   date +"%Y %b %m"
   date +"%r"
   
# PT(H): 250 GeV ~ 500 GeV 
   
    echo "PP H hh"
    python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ggHj.txt > $outpath/proc_ggHj_"$i".log 

    echo "PP jjjj"
    python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ppjj.txt > $outpath/proc_ppjj_"$i".log


   
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))

done


gzip -d $mcdatapath/proc_*/Events/run_*/unweighted_events.lhe.gz


echo "Finish"

date
