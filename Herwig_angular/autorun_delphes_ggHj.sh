#!/bin/bash

echo "Start Running"

outpath="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Log/Herwig_angular_H"
hepmcpath="/home/u5/Universality_Boosetd_Higgs/Herwig_angular"
savepath="/home/u5/Universality_Boosetd_Higgs/Herwig_angular"
rootfilename="ggHj_herwig_ang"
hepmcfilename="ggHj_angular"


i=1

while [ $i != 2 ]
do

       date +"%Y %b %m"
       date +"%r"
       echo "ggHj"
       
       cd /root/Delphes-3.4.2
       
        nohup ./DelphesHepMC /home/alan/ML_Analysis/Universality_Boosetd_Higgs/Cards/delphes_card_HLLHC.tcl $savepath/"$rootfilename"_$i.root $hepmcpath/"$hepmcfilename"_"$i".hepmc > $outpath/"$rootfilename"_"$i"_log.txt &
        
        
       date +"%Y %b %m"
       date +"%r"

   
   i=$(($i+1))

done

echo "Finish"

date
