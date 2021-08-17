#!/bin/bash

home_path="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Herwig_angular"
outpath_H="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Log/Herwig_angular_H"
outpath_QCD="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Log/Herwig_angular_QCD"
mcdatapath="/home/u5"

echo "Start Running"
date

#lhapdf install NNPDF23_nlo_as_0118

nevent=100000
# nevent=5000

i=1
while [ $i != 2 ]
do
    rand=$RANDOM
    
    echo "Random Seed =  $rand "   
    echo "i =  $i "
    
    echo "Reading"   
    
    cd  $home_path/ggHj
    nohup /root/Herwig_7_2_2/bin/Herwig read ggHj_angular.in > read_"$i"_log &
    
    cd  $home_path/ppjj
    nohup /root/Herwig_7_2_2/bin/Herwig read ppjj_angular.in > read_"$i"_log &
    
    sleep 10
    
    echo "Running"   
    
    cd  $home_path/ggHj
    nohup /root/Herwig_7_2_2/bin/Herwig run ggHj_angular.run -N $nevent -s $rand -d 1 > $outpath_H/ggHj_angular_"$i".log &
    
    cd  $home_path/ppjj
    nohup /root/Herwig_7_2_2/bin/Herwig run ppjj_angular.run -N $nevent -s $rand -d 1 > $outpath_QCD/ppjj_angular_"$i".log &
    
    sleep 5
    
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))
   
   
   if [ "$i" -lt "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_0"$(($i))"/g" ./ppjjjj_angular.in  

       sed -i -e "s/ppjjjj_angular_"$(($i-1))"/ppjjjj_angular_"$(($i))"/g" ./ppjjjj_angular.in
       
    elif [ "$i" -eq "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_"$(($i))"/g" ./ppjjjj_angular.in  

       sed -i -e "s/ppjjjj_angular_"$(($i-1))"/ppjjjj_angular_"$(($i))"/g" ./ppjjjj_angular.in

    elif [ "$i" -gt "10" ];then

       sed -i -e "s/run_"$(($i-1))"/run_"$(($i))"/g" ./ppjjjj_angular.in  

       sed -i -e "s/ppjjjj_angular_"$(($i-1))"/ppjjjj_angular_"$(($i))"/g" ./ppjjjj_angular.in
        
    fi

  

done


if [ "$i" -lt "10" ];then

    sed -i -e "s/run_0"$(($i))"/run_01/g" ./ppjjjj_angular.in 
    
elif [ "$i" -eq "10" ];then

    sed -i -e "s/run_"$(($i))"/run_01/g" ./ppjjjj_angular.in
    
elif [ "$i" -gt "10" ];then

    sed -i -e "s/run_"$(($i))"/run_01/g" ./ppjjjj_angular.in
    
fi

cd  $home_path/ggHj
sed -i -e "s/ggHj_angular_"$(($i))"/ggHj_angular_1/g" ./ggHj_angular.in

cd  $home_path/ppjj
sed -i -e "s/ppjj_angular_"$(($i))"/ppjj_angular_1/g" ./ppjj_angular.in


echo "Finish"

date