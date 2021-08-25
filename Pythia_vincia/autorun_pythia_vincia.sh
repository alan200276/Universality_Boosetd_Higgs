#!/bin/bash

PROCESS="Pythia_vincia"
pythiapath="/root/pythia8303/examples"
HOMEPATH="/home/alan/ML_Analysis/Universality_Boosetd_Higgs"
mcdatapath="/home/u5/Universality_Boosetd_Higgs"
outpath_H="Log/"$PROCESS"_H"
outpath_QCD="Log/"$PROCESS"_QCD"
process="pythia_vin"

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

        sed -i -e "s/250_500/"$pt_range"/g" $HOMEPATH/$PROCESS/ggHj.cmnd 
        sed -i -e "s/randomseed/"$rand"/g"  $HOMEPATH/$PROCESS/ggHj.cmnd 
        
        sed -i -e "s/250_500/"$pt_range"/g" $HOMEPATH/$PROCESS/ppjj.cmnd 
        sed -i -e "s/randomseed/"$rand"/g"  $HOMEPATH/$PROCESS/ppjj.cmnd 



        cd $pythiapath
    
        nohup ./main42 $HOMEPATH/$PROCESS/ggHj.cmnd $mcdatapath/$PROCESS/ggHj_"$process"_"$pt_range"_"$i".hepmc > $HOMEPATH/$outpath_H/ggHj_"$pt_range"_"$i".log &
        
        nohup ./main42 $HOMEPATH/$PROCESS/ppjj.cmnd $mcdatapath/$PROCESS/ppjj_"$process"_"$pt_range"_"$i".hepmc > $HOMEPATH/$outpath_QCD/ppjj_"$pt_range"_"$i".log &
        
    
    
        sleep 5
    

        sed -i -e "s/"$pt_range"/250_500/g" $HOMEPATH/$PROCESS/ggHj.cmnd 
        sed -i -e "s/"$rand"/randomseed/g"  $HOMEPATH/$PROCESS/ggHj.cmnd 
        
        sed -i -e "s/"$pt_range"/250_500/g" $HOMEPATH/$PROCESS/ppjj.cmnd 
        sed -i -e "s/"$rand"/randomseed/g"  $HOMEPATH/$PROCESS/ppjj.cmnd 
    
    echo "============================================"
    
    

     
    done  
    ###################################################################



   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))
   
#    echo "i=$i"
   
   if [ "$i" -lt "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_0"$(($i))"/g" $HOMEPATH/$PROCESS/ggHj.cmnd  
       sed -i -e "s/run_0"$(($i-1))"/run_0"$(($i))"/g" $HOMEPATH/$PROCESS/ppjj.cmnd  
#         echo "i<10 $i"
       
    elif [ "$i" -eq "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_"$(($i))"/g" $HOMEPATH/$PROCESS/ggHj.cmnd 
       sed -i -e "s/run_0"$(($i-1))"/run_"$(($i))"/g" $HOMEPATH/$PROCESS/ppjj.cmnd 
#          echo "i==10 $i"

    elif [ "$i" -gt "10" ];then

       sed -i -e "s/run_"$(($i-1))"/run_"$(($i))"/g" $HOMEPATH/$PROCESS/ggHj.cmnd 
       sed -i -e "s/run_"$(($i-1))"/run_"$(($i))"/g" $HOMEPATH/$PROCESS/ppjj.cmnd 
#          echo "i>10 $i"

    fi



done
#========================================================================================


#========================================================================================
#Reset to Origin


if [ "$i" -lt "10" ];then

   sed -i -e "s/run_0"$(($i))"/run_01/g" $HOMEPATH/$PROCESS/ggHj.cmnd 
   sed -i -e "s/run_0"$(($i))"/run_01/g" $HOMEPATH/$PROCESS/ppjj.cmnd 
#         echo "i<10 $i"

elif [ "$i" -eq "10" ];then

   sed -i -e "s/run_"$(($i))"/run_01/g" $$HOMEPATH/$PROCESS/ggHj.cmnd 
   sed -i -e "s/run_"$(($i))"/run_01/g" $$HOMEPATH/$PROCESS/ppjj.cmnd 
#          echo "i==10 $i"

elif [ "$i" -gt "10" ];then

   sed -i -e "s/run_"$(($i))"/run_01/g" $HOMEPATH/$PROCESS/ggHj.cmnd 
   sed -i -e "s/run_"$(($i))"/run_01/g" $HOMEPATH/$PROCESS/ppjj.cmnd 
#          echo "i>10 $i"

fi


echo "Finish"

date