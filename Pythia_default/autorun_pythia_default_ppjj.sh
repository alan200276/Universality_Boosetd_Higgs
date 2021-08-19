#!/bin/bash

pythiapath="/root/pythia8303/examples"
pythiacmndpath="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Pythia_default"
mcdatapath="/home/u5/Universality_Boosetd_Higgs"
outpath="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Log/Pythia_default_QCD"


echo "Start Running"
echo "============================================"
date

# Iterate 10 LHE File for each Pt Slice
i=1
i_tmp=$i
while [ $i != 2 ]
do
#========================================================================================
    
    
    # Iterate Pt Slices
    ###################################################################
    for pt_range in "250_500" "450_700" "650_900" "850_1100" 
    do
    
    echo "pt_range=$pt_range"
    
    rand=$RANDOM
    
    echo "i= $i"
    
    echo "Random Seed =  $rand "    
    
    sed -i -e "s/250_500/"$pt_range"/g" $pythiacmndpath/ppjj.cmnd 
    sed -i -e "s/randomseed/"$rand"/g" $pythiacmndpath/ppjj.cmnd 
    
    
    cd $pythiapath
    
        nohup ./main42 $pythiacmndpath/ppjj.cmnd $mcdatapath/Pythia_default/ppjj_pythia_def_"$pt_range"_"$i".hepmc > $outpath/ppjj_"$pt_range"_"$i".log &
    
    
    sleep 5
    

    
    sed -i -e "s/"$pt_range"/250_500/g" $pythiacmndpath/ppjj.cmnd 
    sed -i -e "s/"$rand"/randomseed/g"  $pythiacmndpath/ppjj.cmnd 
    
    echo "============================================"
    
    

     
    done  
    ###################################################################



   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))
   
#    echo "i=$i"
   
   if [ "$i" -lt "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_0"$(($i))"/g" $pythiacmndpath/ppjj.cmnd  
#         echo "i<10 $i"
       
    elif [ "$i" -eq "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_"$(($i))"/g" $pythiacmndpath/ppjj.cmnd 
#          echo "i==10 $i"

    elif [ "$i" -gt "10" ];then

       sed -i -e "s/run_"$(($i-1))"/run_"$(($i))"/g" $pythiacmndpath/ppjj.cmnd 
#          echo "i>10 $i"

    fi



done
#========================================================================================




if [ "$i" -lt "10" ];then

   sed -i -e "s/run_0"$(($i))"/run_01/g" $pythiacmndpath/ppjj.cmnd  
#         echo "i<10 $i"

elif [ "$i" -eq "10" ];then

   sed -i -e "s/run_"$(($i))"/run_01/g" $pythiacmndpath/ppjj.cmnd 
#          echo "i==10 $i"

elif [ "$i" -gt "10" ];then

   sed -i -e "s/run_"$(($i))"/run_01/g" $pythiacmndpath/ppjj.cmnd 
#          echo "i>10 $i"

fi


echo "Finish"

date