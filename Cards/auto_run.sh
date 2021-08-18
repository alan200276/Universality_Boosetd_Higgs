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
   
# # PT(H): 250 GeV ~ 500 GeV 
   
#     echo "PP H hh"
#     python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ggHj_250_500.txt > $outpath/proc_ggHj_250_500_"$i".log 

#     echo "PP jj"
#     python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ppjj_250_500.txt > $outpath/proc_ppjj_250_500_"$i".log
    
    
# PT(H): 450 GeV ~ 700 GeV 
   
    echo "PP H hh"
    python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ggHj_450_700.txt > $outpath/proc_ggHj_450_700_"$i".log 

    echo "PP jj"
    python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ppjj_450_700.txt > $outpath/proc_ppjj_450_700_"$i".log


# PT(H): 650 GeV ~ 900 GeV 
   
    echo "PP H hh"
    python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ggHj_650_900.txt > $outpath/proc_ggHj_650_900_"$i".log 

    echo "PP jj"
    python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ppjj_650_900.txt > $outpath/proc_ppjj_650_900_"$i".log
    
    
    
# PT(H): 850 GeV ~ 1100 GeV 
   
    echo "PP H hh"
    python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ggHj_850_1100.txt > $outpath/proc_ggHj_850_1100_"$i".log 

    echo "PP jj"
    python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/proc_ppjj_850_1100.txt > $outpath/proc_ppjj_850_1100_"$i".log

   
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))

done


gzip -d $mcdatapath/proc_*/Events/run_*/unweighted_events.lhe.gz
gzip -d $mcdatapath/proc_*/Events/run_*/events.lhe.gz


echo "Finish"

date
