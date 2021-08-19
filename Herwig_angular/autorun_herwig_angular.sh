#!/bin/bash

herwigpath="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Herwig_angular"
outpath_H="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Log/Herwig_angular_H"
outpath_QCD="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Log/Herwig_angular_QCD"
mcdatapath="/home/u5"


echo "Start Running"
echo "============================================"
date

#lhapdf install NNPDF23_nlo_as_0118



nevent=10000

# Iterate 10 LHE File for each Pt Slice
i=1
while [ $i != 2 ]
do
#========================================================================================
    


    # Iterate Pt Slices
    ###################################################################
    for pt_range in "250_500" "450_700" "650_900" "850_1100" 
    do
    
        rand=$RANDOM

        echo "pt_range=$pt_range"
        echo "i =  $i "
        echo "Random Seed =  $rand "   

        cd  $herwigpath/ggHj
        sed -i -e "s/250_500/"$pt_range"/g" ./ggHj_angular.in

        cd  $herwigpath/ppjj
        sed -i -e "s/250_500/"$pt_range"/g" ./ppjj_angular.in


        echo "Reading"   
        echo "============================================"

        cd  $herwigpath/ggHj
        nohup /root/Herwig_7_2_2/bin/Herwig read ggHj_angular.in > read_"$pt_range"_"$i"_log &

        cd  $herwigpath/ppjj
        nohup /root/Herwig_7_2_2/bin/Herwig read ppjj_angular.in > read_"$pt_range"_"$i"_log &

        sleep 10

        echo "Running"  
        echo "============================================"

        cd  $herwigpath/ggHj
        nohup /root/Herwig_7_2_2/bin/Herwig run ggHj_angular.run -N $nevent -s $rand -d 1 > $outpath_H/ggHj_angular_"$pt_range"_"$i".log &

        cd  $herwigpath/ppjj
        nohup /root/Herwig_7_2_2/bin/Herwig run ppjj_angular.run -N $nevent -s $rand -d 1 > $outpath_QCD/ppjj_angular_"$pt_range"_"$i".log &



        cd  $herwigpath/ggHj
        sed -i -e "s/"$pt_range"/250_500/g" ./ggHj_angular.in

        cd  $herwigpath/ppjj
        sed -i -e "s/"$pt_range"/250_500/g" ./ppjj_angular.in


        echo "============================================"

        sleep 5
    
    done  
    ################################################################### 
    
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))
   
   
   if [ "$i" -lt "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_0"$(($i))"/g" ./ppjjjj_angular.in  

       sed -i -e "s/ppjjjj_angular_250_500_"$(($i-1))"/ppjjjj_angular_250_500_"$(($i))"/g" ./ppjjjj_angular.in
       
    elif [ "$i" -eq "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_"$(($i))"/g" ./ppjjjj_angular.in  

       sed -i -e "s/ppjjjj_angular_250_500_"$(($i-1))"/ppjjjj_angular_250_500_"$(($i))"/g" ./ppjjjj_angular.in

    elif [ "$i" -gt "10" ];then

       sed -i -e "s/run_"$(($i-1))"/run_"$(($i))"/g" ./ppjjjj_angular.in  

       sed -i -e "s/ppjjjj_angular_250_500_"$(($i-1))"/ppjjjj_angular_250_500_"$(($i))"/g" ./ppjjjj_angular.in
        
    fi

  

done
#========================================================================================




if [ "$i" -lt "10" ];then

    sed -i -e "s/run_0"$(($i))"/run_01/g" ./ppjjjj_angular.in 
    
elif [ "$i" -eq "10" ];then

    sed -i -e "s/run_"$(($i))"/run_01/g" ./ppjjjj_angular.in
    
elif [ "$i" -gt "10" ];then

    sed -i -e "s/run_"$(($i))"/run_01/g" ./ppjjjj_angular.in
    
fi

cd  $herwigpath/ggHj
sed -i -e "s/ggHj_angular_250_500_"$(($i))"/ggHj_angular_"$pt_range"_1/g" ./ggHj_angular.in

cd  $herwigpath/ppjj
sed -i -e "s/ppjj_angular_250_500_"$(($i))"/ppjj_angular_"$pt_range"_1/g" ./ppjj_angular.in


echo "Finish"

date